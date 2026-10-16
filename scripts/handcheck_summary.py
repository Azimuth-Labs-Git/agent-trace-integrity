"""Summarise the PI's D1 hand-check into results/handcheck-d1.json (counts and intervals only).

Reads handcheck/d1-hand-check-2026-10-15.json (v1 answers) and writes the v2 label mapping
alongside. Usage: python scripts/handcheck_summary.py
"""

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "handcheck" / "d1-hand-check-2026-10-15.json"
# v1 labels as answered -> v2 labels (Matt, 2026-10-15): misreport merged into logical_error,
# spoof renamed spoof_possible, no_mistake renamed no_mistake_by_ai.
V2 = {"misreport": "logical_error", "logical_error": "logical_error",
      "hallucination": "hallucination", "spoof": "spoof_possible",
      "no_mistake": "no_mistake_by_ai", "unsure": "unsure", "other": "other"}
CHECK_A = range(1, 6)  # cards 1-5 are every check A flag in D1; 6-20 are sub-type 3 flags


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 2), round(min(1.0, c + h), 2)]


def block(rows):
    out = {"cards": len(rows),
           "distinct_messages": len({(r["instance_id"], r["claim_event"]) for r in rows}),
           "distinct_runs": len({(r["submission"], r["instance_id"]) for r in rows})}
    for q in ("q1_is_claim", "q2_checker_right"):
        c = Counter(r[q] for r in rows)
        out[q] = {"yes": c["yes"], "no": c["no"], "not_sure": c["not_sure"],
                  "yes_wilson95": wilson(c["yes"], len(rows))}
    return out


def cited_steps(r):
    """The step the checker cited as conflicting, for this card's claim (rerun from pins)."""
    from agent_trace_integrity.adapters import load
    from agent_trace_integrity.fetch import fetch
    from agent_trace_integrity.verdicts import check_run

    pins = json.loads((ROOT / "data" / "pins" / "d1.json").read_text())
    rp = next(x for x in pins["submissions"][r["submission"]]["runs"]
              if x["instance_id"] == r["instance_id"])
    [run] = load(fetch(rp["url"], rp["sha256"]), "swebench")
    return sorted({e["event_index"] for v in check_run(run)
                   if v["source"]["event_index"] == r["claim_event"]
                   for e in v["evidence"] if e["role"] == "conflicts"})


def main():
    rows = json.loads(SRC.read_text())
    v2 = [{**r, "q3_mistake_kind_v2": V2[r["q3_mistake_kind"]]} for r in rows]
    (ROOT / "handcheck" / "d1-hand-check-2026-10-15.v2-labels.json").write_text(
        json.dumps(v2, indent=1) + "\n")
    out = {
        "rater": "PI, single rater, model names visible, flagged claims only (not a random "
                 "sample), checker verdict and builder notes visible",
        "check_A_fabricated_output": block([r for r in rows if r["card"] in CHECK_A]),
        "sub_type_3_success_after_failed_check": block([r for r in rows
                                                        if r["card"] not in CHECK_A]),
        "q1_yes_all_cards": sum(r["q1_is_claim"] == "yes" for r in rows),
        "unconfirmed": [{"card": r["card"], "instance_id": r["instance_id"],
                         "claim_event": r["claim_event"], "cited_steps": cited_steps(r),
                         "q2": r["q2_checker_right"]}
                        for r in rows if r["q2_checker_right"] != "yes"],
        "q3_v2_counts_not_reported_as_findings": dict(Counter(r["q3_mistake_kind_v2"]
                                                              for r in v2)),
    }
    (ROOT / "results" / "handcheck-d1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["unconfirmed"]))


if __name__ == "__main__":
    main()
