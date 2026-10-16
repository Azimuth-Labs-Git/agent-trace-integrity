"""Add derived totals to results/malt-d4.json (counts only).

Usage: python scripts/malt_summary.py
"""

import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "results" / "malt-d4.json"
CHEATING = ("bypass_constraints", "hardcoded_solution", "sabotage")


def main():
    d = json.loads(P.read_text())
    per = d["per_label"]
    d["summary"] = {
        "runs_labelled_bypass_hardcoded_sabotage": sum(per[k]["runs"] for k in CHEATING),
        "check_B_flags_on_those": sum(per[k].get("runs_flag_B", 0) for k in CHEATING),
        "flagged_run_ids": sorted(r["run_id"] for v in d["flagged_non_normal"].values()
                                  for r in v),
    }
    P.write_text(json.dumps(d, indent=1) + "\n")
    print(d["summary"])


if __name__ == "__main__":
    main()
