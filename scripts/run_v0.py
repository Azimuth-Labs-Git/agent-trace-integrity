"""Phase 5: run the checker on the real datasets and write every number to one results file.

All inputs are pinned by URL and sha256 (data/pins/, devset/, seeded/, results/seeded-d2.json)
and fetched at runtime into the cache; nothing third-party is committed. The output holds counts,
intervals, run ids and line numbers only: no transcript text, no names, no usernames.

  D1  SWE-bench bash-only, 4 models x 75 tasks on mini-swe-agent v1.x (headline)
      plus a side note on 7 mini-swe-agent v2 submissions (20 runs each)
  D3  Palisade shutdown-avoidance Inspect logs, 22 logs (worked example and scorer agreement)
  D5  Gemini CLI issue #4586 transcript, lines 1 to 511 (lossy-log illustration)
  D2  seeded regression results and the Phase 2 dev-set precision, copied from their own files

Usage: python scripts/run_v0.py [--out results/v0-2026-10.json] [--cards results/cards.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from agent_trace_integrity import __version__
from agent_trace_integrity.adapters import load
from agent_trace_integrity.adapters.gemini_text import load_lines
from agent_trace_integrity.claims import find_claims
from agent_trace_integrity.fetch import fetch
from agent_trace_integrity.verdicts import check_run, command_of, run_summary

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "data" / "pins"
B = "check_modified_before_success_claim"
A = "fabricated_tool_output"
D5 = ("https://github.com/user-attachments/files/21372906/gemini.cli.screw.up.txt",
      "000ddab20df84d46ebbe70b36a73f21eadb6761b190f3d16722f404c31a34935")
SUB3 = ("test_success", "generic_success")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def rate(k: int, n: int) -> dict:
    """A count with its denominator, a Wilson 95% interval, and the rule-of-three upper bound
    when the count is zero."""
    out = {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "wilson95": wilson(k, n)}
    if k == 0 and n:
        out["upper95_rule_of_three"] = round(3 / n, 3)
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- D1


def _resolved(run_pin: dict, details: dict) -> bool | None:
    if run_pin.get("report_sha256"):
        rep = json.loads(fetch(run_pin["report_url"], run_pin["report_sha256"]).read_text())
        if isinstance(rep.get("resolved"), bool):
            return rep["resolved"]
        inner = rep.get(run_pin["instance_id"], {})
        if isinstance(inner.get("resolved"), bool):
            return inner["resolved"]
    v = details.get(run_pin["instance_id"], {}).get("resolved")
    return v if isinstance(v, bool) else None


def d1_submission(sub: str, s: dict, cards: list) -> dict:
    details = json.loads(fetch(s["details"]["url"], s["details"]["sha256"]).read_text())
    cat_verdicts: Counter = Counter()
    asserted = Counter()
    runs_with_claim = 0
    flags_a = flags_b = runs_b = 0
    b_edits = test_edits = 0
    succ_by_res = {True: [0, 0], False: [0, 0], None: [0, 0]}
    sub3 = Counter()
    for rp in s["runs"]:
        [run] = load(fetch(rp["url"], rp["sha256"]), "swebench")
        vs = check_run(run)
        summ = run_summary(run, vs)
        b_edits += len(summ["check_b_edits"])
        test_edits += summ["test_edits_seen"]
        claims = [v for v in vs if v["claim"]["category"] != "submit"]
        a_claims = [v for v in claims if v["claim"]["modality"] == "asserted"]
        runs_with_claim += bool(a_claims)
        for v in claims:
            c = v["claim"]["category"]
            asserted[(c, v["claim"]["modality"])] += 1
            cat_verdicts[(c, v["verdict"])] += 1
            if v["claim"]["modality"] == "asserted" and c in SUB3:
                sub3[v["verdict"]] += 1
            if A in v["flags"]:
                flags_a += 1
        flagged_b = any(B in v["flags"] for v in vs)
        flags_b += sum(B in v["flags"] for v in vs)
        runs_b += flagged_b
        res = _resolved(rp, details)
        has_succ = any(v["claim"]["category"] in SUB3 for v in a_claims)
        succ_by_res[res][0] += has_succ
        succ_by_res[res][1] += 1
        for v in vs:
            if v["verdict"] in ("contradicted", "uncorroborated") or v["flags"]:
                cards.append({"dataset": "D1", "submission": sub, "instance_id": rp["instance_id"],
                              "traj_url": rp["url"], "traj_sha256": rp["sha256"],
                              "resolved": res, "verdict": v})
    checked3 = sub3["contradicted"] + sub3["corroborated"]
    return {
        "runs": len(s["runs"]),
        "runs_with_an_asserted_claim": runs_with_claim,
        "claims_by_category_and_modality": {f"{c}/{m}": n
                                            for (c, m), n in sorted(asserted.items())},
        "verdicts_by_category": {f"{c}/{v}": n for (c, v), n in sorted(cat_verdicts.items())},
        "sub_type_3_asserted_success_claims": dict(sub3),
        "sub_type_3_contradicted_of_settled": rate(sub3["contradicted"], checked3),
        "check_A_fabricated_tool_output_flags": rate(flags_a, sum(
            n for (c, m), n in asserted.items() if c == "output_like" and m == "asserted")),
        "check_B_runs_flagged": rate(runs_b, len(s["runs"])),
        "check_B_flags_on_verdicts": flags_b,
        "check_B_edits_found": b_edits,
        "test_edits_seen_not_flagged": test_edits,
        "harness_check": {
            "note": "Share of runs with an asserted success claim, split by the harness resolved "
                    "flag. resolved uses hidden tests the agent never ran, so a claim in an "
                    "unresolved run is weak evidence, not proof, of a false claim.",
            "resolved_false": rate(*succ_by_res[False]),
            "resolved_true": rate(*succ_by_res[True]),
            "resolved_unknown_runs": succ_by_res[None][1],
        },
    }


def d1(cards: list) -> dict:
    pins = json.loads((PINS / "d1.json").read_text())
    out = {"source": "SWE-bench/experiments bash-only submissions, trajectories on public S3",
           "experiments_commit": pins["experiments_commit"], "rule": pins["rule"],
           "pins_sha256": _sha(PINS / "d1.json"), "scope": "final narration of each run",
           "models": {}}
    for sub, s in pins["submissions"].items():
        out["models"][sub] = {"provider_group": s["provider_group"],
                              **d1_submission(sub, s, cards)}
        print("D1", sub, "done")
    return out


def d1_v2_note() -> dict:
    pins = json.loads((PINS / "d1-v2-sidenote.json").read_text())
    out = {"note": pins["note"], "pins_sha256": _sha(PINS / "d1-v2-sidenote.json"),
           "submissions": {}}
    for sub, s in pins["submissions"].items():
        n_claims = runs_with = 0
        for rp in s["runs"]:
            [run] = load(fetch(rp["url"], rp["sha256"]), "swebench")
            cs = [c for c in find_claims(run) if c.modality == "asserted"]
            n_claims += len(cs)
            runs_with += bool(cs)
        out["submissions"][sub] = {"runs": len(s["runs"]), "asserted_final_claims": n_claims,
                                   "runs_with_a_claim": runs_with}
    return out


# ---------------------------------------------------------------- D3

_READS_SHUTDOWN = re.compile(r"\bshutdown\.sh\b")


def d3(cards: list) -> dict:
    pins = json.loads((PINS / "d3.json").read_text())
    out = {"source": "PalisadeResearch/shutdown_avoidance, docs/inspect_logs/logs",
           "commit": pins["commit"], "pins_sha256": _sha(PINS / "d3.json"),
           "scorer_note": "Palisade's only scorer is `exact` (C when all answers were submitted "
                          "correctly, which needs getting past the shutdown). It is a proxy for "
                          "shutdown avoidance, not a direct label. Check B has no rule for kinds "
                          "of sabotage other than writing to or disabling the named script.",
           "logs": [], "worked_example": None}
    tot = Counter()
    for lg in pins["logs"]:
        runs = load(fetch(lg["url"], lg["sha256"]), "inspect")
        c = Counter()
        for run in runs:
            sc = (run[0].meta.get("scores") or {}).get("exact", {}).get("value")
            vs = check_run(run)
            flagged = any(B in v["flags"] for v in vs)
            touched = any(e.role == "tool_call" and _READS_SHUTDOWN.search(command_of(e))
                          for e in run)
            c[f"scorer_{sc}_flagged_{flagged}"] += 1
            if touched and not flagged:
                c["touched_shutdown_sh_not_flagged"] += 1
            if touched and flagged and sc == "I":
                c["flagged_but_scorer_I"] += 1
            if flagged and sc == "I":
                cards.append({"dataset": "D3", "file": lg["file"],
                              "sample_id": run[0].meta.get("sample_id"),
                              "epoch": run[0].meta.get("epoch"), "verdict":
                              next(v for v in vs if B in v["flags"])})
            if (lg["model"] == "openai/o3" and str(run[0].meta.get("sample_id")) == "21"
                    and run[0].meta.get("epoch") == 1 and out["worked_example"] is None
                    and "moReoGKNRrN4aoUz8Bf2vS" in lg["file"]):
                v = next(v for v in vs if B in v["flags"])
                ev = [e for e in v["evidence"] if e.get("flag") == B]
                out["worked_example"] = {
                    "file": lg["file"], "sample_id": 21, "epoch": 1, "model": "openai/o3",
                    "reads_of_shutdown_sh_at_events": [
                        e.index for e in run if e.role == "tool_call"
                        and re.search(r"\bcat\b.*shutdown\.sh", command_of(e))],
                    "flagged_edit_event_index": ev[0]["event_index"],
                    "flagged_edit": ev[0]["observed_value"],
                    "final_submit_event_index": v["source"]["event_index"],
                    "scorer": sc, "verdict_claim_category": v["claim"]["category"],
                }
        cC = c["scorer_C_flagged_True"] + c["scorer_C_flagged_False"]
        cI = c["scorer_I_flagged_True"] + c["scorer_I_flagged_False"]
        out["logs"].append({
            "file": lg["file"], "model": lg["model"], "samples": len(runs),
            "scorer_C": cC, "scorer_I": cI,
            "flagged_among_C": rate(c["scorer_C_flagged_True"], cC),
            "flagged_among_I": rate(c["scorer_I_flagged_True"], cI),
            "touched_shutdown_sh_not_flagged": c["touched_shutdown_sh_not_flagged"],
        })
        tot.update(c)
        tot["C"] += cC
        tot["I"] += cI
    out["all_logs"] = {
        "samples": tot["C"] + tot["I"],
        "agreement_with_scorer": rate(tot["scorer_C_flagged_True"] + tot["scorer_I_flagged_False"],
                                      tot["C"] + tot["I"]),
        "flagged_among_C": rate(tot["scorer_C_flagged_True"], tot["C"]),
        "false_alarms_flagged_among_I": rate(tot["scorer_I_flagged_True"], tot["I"]),
        "runs_that_touched_shutdown_sh_but_were_not_flagged":
            tot["touched_shutdown_sh_not_flagged"],
    }
    print("D3 done")
    return out


# ---------------------------------------------------------------- D5


def d5() -> dict:
    path = fetch(*D5)
    [run] = load_lines(path, 1, 511)
    vs = check_run(run, "all")
    rows = []
    for v in vs:
        e = run[v["source"]["event_index"]]
        rows.append({"lines": e.source_ref, "category": v["claim"]["category"],
                     "modality": v["claim"]["modality"], "verdict": v["verdict"],
                     "sub_type": v["sub_type"], "reason_code": v["reason_code"],
                     "flags": v["flags"]})
    results = [e for e in run if e.role == "tool_result"]
    return {"source": "Gemini CLI issue #4586, attached transcript (contains personal data: "
                      "only line numbers and verdicts are published)",
            "sha256": D5[1], "lines": "1-511", "scope": "all narration",
            "tool_results": len(results),
            "tool_results_lost_in_copy": sum(1 for e in results if e.tool_result is None),
            "verdicts": rows, "verdict_counts": dict(Counter(r["verdict"] for r in rows))}


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "v0-2026-10.json"))
    ap.add_argument("--cards", default=str(ROOT / "local" / "flags-d1-d3.json"))
    args = ap.parse_args()
    cards: list = []
    seeded = json.loads((ROOT / "results" / "seeded-d2.json").read_text())
    dev_text = (ROOT / "devset" / "report.json").read_text()
    dev = json.loads(dev_text[: dev_text.index("\n}") + 2])
    out = {
        "title": "agent-trace-integrity v0 results (preliminary)",
        "tool_version": __version__,
        "labels": "Preliminary. Every number names its dataset, harness and models. Rates are "
                  "per sub-type and per dataset, never pooled.",
        "D1": d1(cards),
        "D1_v2_side_note": d1_v2_note(),
        "D3": d3(cards),
        "D5": d5(),
        "D2_seeded_regression": {k: {kk: vv for kk, vv in v.items() if kk != "missed_examples"}
                                 for k, v in seeded["checks"].items()},
        "D2_seeded_sha256": _sha(ROOT / "results" / "seeded-d2.json"),
        "phase2_dev_precision": {k: dev[k] for k in ("precision", "precision_wilson95",
                                                     "finder_claims", "finder_hits",
                                                     "recall_of_reference_sentences",
                                                     "span_agreement_f1")},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    Path(args.cards).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cards).write_text(json.dumps(cards, indent=1) + "\n")
    print("wrote", args.out, "and", len(cards), "flagged verdicts to", args.cards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
