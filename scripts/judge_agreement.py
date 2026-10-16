"""Compare the open-weight LLM judge with the PI's hand labels (plan section 3, "The LLM judge").

Reads local/judge-<model>.jsonl (gitignored) and handcheck/d1-hand-check-2026-10-15.json, and
writes results/judge-d1.json with counts only: agreement with the PI's question 1 ("is this a
claim?") on the 20 hand-checked cards, and how the judge labels the D1 sentences the rules marked
borderline. Usage: python scripts/judge_agreement.py local/judge-Qwen_Qwen3.5-4B.jsonl
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 2), round(min(1.0, c + h), 2)]


def main():
    rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
    pi = {f"card-{a['card']}": a["q1_is_claim"]
          for a in json.loads((ROOT / "handcheck" / "d1-hand-check-2026-10-15.json").read_text())}
    cards = [r for r in rows if r["id"] in pi]
    settled = [r for r in cards if pi[r["id"]] in ("yes", "no")]
    agree = sum(r["answer"] == pi[r["id"]] for r in settled)
    border = [r for r in rows if r["id"].startswith("bl-")]
    meta = rows[0]
    out = {
        "judge": {k: meta.get(k) for k in ("model", "backend", "backend_version", "temperature",
                                           "seed", "prompt_sha256")},
        "question": "Is this sentence an asserted claim about an action the agent took or its "
                    "result?",
        "pi_q1_agreement": {"agree": agree, "settled_cards": len(settled),
                            "rate": round(agree / len(settled), 3) if settled else None,
                            "wilson95": wilson(agree, len(settled)),
                            "pi_not_sure_excluded": len(cards) - len(settled)},
        "judge_on_cards": dict(Counter(r["answer"] for r in cards)),
        "borderline_d1_sentences": {"n": len(border),
                                    "judge": dict(Counter(r["answer"] for r in border))},
        "note": "The judge never runs inside the checker. One small open-weight model, one fixed "
                "prompt, temperature 0. The second, larger judge was not run (cost cap).",
    }
    (ROOT / "results" / "judge-d1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
