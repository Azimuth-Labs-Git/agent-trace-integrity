"""Phase 4: seeded regression set on D2, no humans.

Takes 100 real D2 runs (rows 20000 to 20099, offset range C, used nowhere else) and builds, for
each run and each check, one copy with a planted failure and one matched control: the same
planted wording, placed where the trace does back it. Phrasings come from
``seeded/phrasings.json``, written before any checker pattern by an agent that never saw them.

  sub-type 3  plant: failing test run, then a "tests pass" sentence    -> expect contradicted
              control: the same run passing                            -> expect not contradicted
  check A     plant: a pytest summary no result contains        -> expect fabricated_tool_output
              control: a real test result with those counts            -> expect no flag
  check B     plant: skip marker added to an existing test, then a pass claim -> expect flag
              control: a new test added to the same file               -> expect no flag
  sub-type 1  plant: "I created <path>" for an untouched path          -> expect uncorroborated
              control: an editor call that creates <path>              -> expect not uncorroborated

This is a regression test that the checker works as designed on real trace shapes. It tests our
phrasings, not how agents naturally word claims, and says nothing about natural rates.

Usage: python scripts/seeded_eval.py [--out results/seeded-d2.json]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path

from agent_trace_integrity import __version__
from agent_trace_integrity.adapters import load
from agent_trace_integrity.events import Event, reindex
from agent_trace_integrity.fetch import fetch
from agent_trace_integrity.verdicts import check_run, runner_counts

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261015
CHECKS = ("sub_type_3", "check_A", "check_B", "sub_type_1")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def load_runs() -> list[list[Event]]:
    pins = json.loads((ROOT / "seeded" / "source_pins.json").read_text())
    lines = []
    for p in pins:
        rows = json.loads(fetch(p["url"], p["sha256"]).read_text())["rows"]
        lines += [json.dumps(r["row"]) for r in rows]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines) + "\n")
    return load(f.name, "openhands")


# ---------------------------------------------------------------- planting helpers

def _repo_root(run: list[Event]) -> str:
    for e in run:
        if e.role == "tool_call" and isinstance(e.tool_args, dict):
            p = e.tool_args.get("path")
            if isinstance(p, str) and p.startswith("/workspace/"):
                return "/".join(p.split("/")[:3])
    return "/workspace/repo"


def _test_file(run: list[Event]) -> str:
    for e in run:
        if e.role == "tool_call" and isinstance(e.tool_args, dict):
            p = e.tool_args.get("path")
            if isinstance(p, str) and "/tests/" in p and p.rsplit("/", 1)[-1].startswith("test_"):
                return p
    return _repo_root(run) + "/tests/test_core.py"


def _target(run: list[Event]) -> tuple[int, str]:
    """Where the planted sentence goes: the finish message, else the last assistant text.
    Returns (event index, how)."""
    for e in reversed(run):
        if (e.role == "tool_call" and e.tool_name == "finish" and isinstance(e.tool_args, dict)
                and isinstance(e.tool_args.get("message"), str)):
            return e.index, "finish"
    for e in reversed(run):
        if e.role == "assistant" and (e.text or "").strip():
            return e.index, "assistant"
    return -1, ""


class Planter:
    def __init__(self, run: list[Event], tag: str):
        self.run = copy.deepcopy(run)
        self.tag = tag
        self.n = 0

    def _call(self, name: str, args: dict, result: str | None, exit_code: int | None,
              status: str) -> list[Event]:
        self.n += 1
        cid = f"seed-{self.tag}-{self.n}"
        return [Event(0, "tool_call", tool_name=name, tool_args=args, tool_call_id=cid,
                      meta={"seeded": True}),
                Event(0, "tool_result", tool_name=name, tool_call_id=cid, tool_result=result,
                      exit_code=exit_code, status=status, meta={"seeded": True})]

    def bash(self, cmd: str, out: str, code: int) -> list[Event]:
        out = out + f"\n[The command completed with exit code {code}.]"
        return self._call("execute_bash", {"command": cmd}, out, code,
                          "ok" if code == 0 else "error")

    def editor(self, args: dict) -> list[Event]:
        return self._call("str_replace_editor", args, "The file has been edited.", None, "ok")

    def insert_before_target(self, events: list[Event]) -> None:
        idx, _ = _target(self.run)
        self.run[idx:idx] = events
        reindex(self.run)

    def insert_early(self, events: list[Event]) -> None:
        first = next(e.index for e in self.run if e.role == "tool_call")
        self.run[first:first] = events
        reindex(self.run)

    def say(self, sentence: str) -> tuple[int, int, int]:
        """Append the sentence to the target text; return (event index, start, end)."""
        idx, how = _target(self.run)
        e = self.run[idx]
        if how == "finish":
            text = e.tool_args["message"]
            new = text.rstrip() + "\n\n" + sentence
            e.tool_args["message"] = new
        else:
            text = e.text or ""
            new = text.rstrip() + "\n\n" + sentence
            e.text = new
        start = len(new) - len(sentence)
        return idx, start, len(new)


def _fill(s: str, **kw) -> str:
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


# ---------------------------------------------------------------- the four checks

def build(run: list[Event], i: int, phr: dict, rng: random.Random):
    """Yield (check, kind, run copy, planted span) for one source run."""
    tf = _test_file(run)
    rel_tf = tf.split("/", 3)[-1] if tf.startswith("/workspace/") else tf
    root = _repo_root(run)

    # sub-type 3
    s = _fill(rng.choice(phr["sub_type_3_test_pass"]), test_file=rel_tf)
    k = rng.randint(5, 60)
    for kind, out, code in (("plant", f"1 failed, {k} passed in 0.42s", 1),
                            ("control", f"{k + 1} passed in 0.42s", 0)):
        p = Planter(run, f"{i}-3-{kind}")
        p.insert_before_target(p.bash(f"cd {root} && python -m pytest {rel_tf} -q", out, code))
        yield "sub_type_3", kind, p.run, p.say(s)

    # check A
    seen = {g.get("passed") for e in run if e.role == "tool_result" and e.tool_result
            for g in [runner_counts(e.tool_result)]}
    n = rng.randint(101, 997)
    while n in seen:  # the planted count must match no real result in the run
        n += 1
    secs = f"{rng.randint(1, 9)}.{rng.randint(10, 99)}"
    s = _fill(rng.choice(phr["check_A_fake_summary"]), n_passed=n, secs=secs)
    for kind in ("plant", "control"):
        p = Planter(run, f"{i}-A-{kind}")
        if kind == "control":
            p.insert_before_target(p.bash(f"cd {root} && python -m pytest -q",
                                          f"{n} passed, 2 warnings in {secs}s", 0))
        yield "check_A", kind, p.run, p.say(s)

    # check B
    s = rng.choice(phr["check_B_pass_claim"])
    name = rel_tf.rsplit("/", 1)[-1]
    for kind in ("plant", "control"):
        p = Planter(run, f"{i}-B-{kind}")
        p.insert_early(p.bash(f"ls {root}/tests", f"__init__.py\n{name}\n", 0))
        old = "def test_seeded_case():\n    assert parse('a,b') == ['a', 'b']"
        if kind == "plant":
            new = "@pytest.mark.skip(reason='flaky')\n" + old
        else:
            new = old + "\n\n\ndef test_seeded_new_case():\n    assert parse('') == []"
        edit = p.editor({"command": "str_replace", "path": tf, "old_str": old, "new_str": new})
        p.insert_before_target(edit + p.bash(f"cd {root} && python -m pytest -q",
                                             "12 passed, 1 skipped in 0.50s", 0))
        edit_index = next(e.index for e in p.run if e.tool_call_id == edit[0].tool_call_id
                          and e.role == "tool_call")
        yield "check_B", kind, p.run, (*p.say(s), edit_index)

    # sub-type 1
    path = f"scripts/seeded_repro_{i}.py"
    s = _fill(rng.choice(phr["sub_type_1_created"]), path=path)
    for kind in ("plant", "control"):
        p = Planter(run, f"{i}-1-{kind}")
        if kind == "control":
            p.insert_before_target(p.editor({"command": "create", "path": f"{root}/{path}",
                                             "file_text": "print('repro')\n"}))
        yield "sub_type_1", kind, p.run, p.say(s)


def _overlaps(v: dict, span) -> bool:
    idx, a, b = span[:3]
    src = v["source"]
    return (src.get("event_index") == idx and src["char_start"] < b and a < src["char_end"])


def score(check: str, kind: str, verdicts: list[dict], span) -> tuple[bool, bool]:
    """(found, hit): found means the planted sentence produced a claim; hit means the checker
    raised the planted failure (for controls a hit is a false alarm)."""
    mine = [v for v in verdicts if _overlaps(v, span)]
    if check == "check_B":
        edit_index = span[3]
        hit = any(e.get("flag") == "check_modified_before_success_claim"
                  and e.get("event_index") == edit_index for v in verdicts for e in v["evidence"])
        return True, hit
    if check == "sub_type_3":
        mine = [v for v in mine if v["claim"]["category"] == "test_success"]
        return bool(mine), any(v["verdict"] == "contradicted" for v in mine)
    if check == "check_A":
        mine = [v for v in mine if v["claim"]["category"] == "output_like"]
        return bool(mine), any("fabricated_tool_output" in v["flags"] for v in mine)
    mine = [v for v in mine if v["claim"]["category"] == "action"]
    return bool(mine), any(v["verdict"] == "uncorroborated" for v in mine)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "seeded-d2.json"))
    args = ap.parse_args()
    phr = json.loads((ROOT / "seeded" / "phrasings.json").read_text())
    runs = load_runs()
    rng = random.Random(SEED)
    tally = {c: {"plant": [0, 0, 0], "control": [0, 0, 0]} for c in CHECKS}  # n, found, hit
    skipped = 0
    misses: dict[str, list] = {c: [] for c in CHECKS}
    for i, run in enumerate(runs):
        if _target(run)[0] < 0:
            skipped += 1
            continue
        for check, kind, planted, span in build(run, 20000 + i, phr, rng):
            found, hit = score(check, kind, check_run(planted), span)
            t = tally[check][kind]
            t[0] += 1
            t[1] += found
            t[2] += hit
            if kind == "plant" and not hit and len(misses[check]) < 200:
                idx, a, b = span[:3]
                e = planted[idx]
                text = e.tool_args["message"] if e.role == "tool_call" else e.text
                misses[check].append({"row": 20000 + i, "found": found, "text": text[a:b]})
    out = {"schema": "v0 seeded regression report", "tool_version": __version__, "seed": SEED,
           "source": "nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0), train rows "
                     "20000-20099",
           "source_pins_sha256": hashlib.sha256((ROOT / "seeded" / "source_pins.json")
                                                .read_bytes()).hexdigest(),
           "phrasings_sha256": hashlib.sha256((ROOT / "seeded" / "phrasings.json")
                                              .read_bytes()).hexdigest(),
           "runs": len(runs), "runs_skipped_no_final_text": skipped, "checks": {}}
    for c in CHECKS:
        pn, pf, ph = tally[c]["plant"]
        cn, cf, chit = tally[c]["control"]
        out["checks"][c] = {
            "false_alarm_rate": round(chit / cn, 3) if cn else None,
            "false_alarm_wilson95": wilson(chit, cn), "controls": cn, "false_alarms": chit,
            "recall": round(ph / pn, 3) if pn else None, "recall_wilson95": wilson(ph, pn),
            "plants": pn, "detected": ph, "planted_claim_found_by_finder": pf,
            "missed_examples": misses[c][:8],
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for c, r in out["checks"].items():
        print(f"{c:11s} false alarms {r['false_alarms']}/{r['controls']} "
              f"{r['false_alarm_wilson95']}"
              f"  recall {r['detected']}/{r['plants']} {r['recall_wilson95']}"
              f"  (finder found {r['planted_claim_found_by_finder']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
