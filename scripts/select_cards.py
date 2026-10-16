"""Select Matt's 20 hand-check cards from D1 (Phase 5), by a fixed rule, and pull the evidence
lines each card needs. Writes local/cards-input.json (gitignored: it holds agent text).

Rule: every check A and check B flag first; then contradicted and uncorroborated claims in
sha256(instance_id + claim_id) order; then, if fewer than 20, unflagged corroborated claims in
the same order. Matt judges only the language side (is this an asserted claim, read correctly);
the evidence side is mechanical.
"""

import hashlib
import json
from pathlib import Path

from agent_trace_integrity.adapters import load
from agent_trace_integrity.fetch import fetch
from agent_trace_integrity.verdicts import check_run, command_of

ROOT = Path(__file__).resolve().parents[1]
N = 20


def _key(rp, v):
    return hashlib.sha256((rp["instance_id"] + v["claim_id"] + rp["url"]).encode()).hexdigest()


def main() -> None:
    pins = json.loads((ROOT / "data" / "pins" / "d1.json").read_text())
    flagged, bad, good = [], [], []
    runs = {}
    for sub, s in pins["submissions"].items():
        for rp in s["runs"]:
            [run] = load(fetch(rp["url"], rp["sha256"]), "swebench")
            runs[rp["url"]] = run
            for v in check_run(run):
                item = (sub, rp, v)
                if v["flags"]:
                    flagged.append(item)
                elif v["verdict"] in ("contradicted", "uncorroborated"):
                    bad.append(item)
                elif v["verdict"] == "corroborated" and v["claim"]["category"] != "submit":
                    good.append(item)
    pick = flagged + sorted(bad, key=lambda x: _key(x[1], x[2]))
    pick = pick[:N] + sorted(good, key=lambda x: _key(x[1], x[2]))[: max(0, N - len(pick))]
    cards = []
    for n, (sub, rp, v) in enumerate(pick, 1):
        run = runs[rp["url"]]
        ev = []
        for e in v["evidence"]:
            i = e.get("event_index")
            if i is None:
                continue
            call = run[i]
            res = next((x for x in run if x.role == "tool_result"
                        and x.tool_call_id == call.tool_call_id), None)
            tail = (res.tool_result or "").strip().splitlines()[-12:] if res else []
            ev.append({"event_index": i, "role": e["role"], "flag": e.get("flag"),
                       "command": command_of(call)[:600],
                       "exit_code": res.exit_code if res else None,
                       "result_tail": "\n".join(tail)[:1500], "note": e.get("note")})
        src = run[v["source"]["event_index"]]
        text = src.text or ""
        a, b = v["source"]["char_start"], v["source"]["char_end"]
        cards.append({"card": n, "submission": sub, "instance_id": rp["instance_id"],
                      "claim_sentence": v["source"]["text"],
                      "context": text[max(0, a - 300): b + 300],
                      "category": v["claim"]["category"], "verdict": v["verdict"],
                      "sub_type": v["sub_type"], "flags": v["flags"],
                      "reason_code": v["reason_code"], "evidence": ev,
                      "selected_because": "flag" if v["flags"] else (
                          "contradicted/uncorroborated" if v["verdict"] in (
                              "contradicted", "uncorroborated") else "random corroborated")})
    out = ROOT / "local" / "cards-input.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(cards, indent=1))
    print(len(flagged), "flagged,", len(bad), "contradicted/uncorroborated,", len(cards), "cards")


if __name__ == "__main__":
    main()
