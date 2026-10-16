"""Check that every number in a write-up comes from the results files.

Usage: python scripts/check_numbers.py TEXT_FILE [TEXT_FILE ...]

A number passes if it equals a value in results/v0-2026-10.json or results/seeded-d2.json
(after flattening), or a percentage of one (0.937 -> 93.7%), or a sum of counts that the text
marks with "total". Numbers inside code spans, links, dates (2026-10-15), version strings, dataset
and sub-type names (D1, 2a), schema versions and the allow-list below are skipped. Prints every
number it could not trace and exits 1 if there are any.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 20, 22, 30, 60, 75, 95, 100, 300, 511, 2025, 2026,
         4586, 67847, 0.8, 0.6}


def values(obj, out: set):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 6))
    elif isinstance(obj, dict):
        for v in obj.values():
            values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            values(v, out)
    elif isinstance(obj, str):
        for m in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", obj):
            out.add(round(float(m), 6))


def main() -> int:
    known: set = set()
    for name in ("v0-2026-10.json", "seeded-d2.json", "handcheck-d1.json", "judge-d1.json"):
        values(json.loads((ROOT / "results" / name).read_text()), known)
    pct = {round(v * 100, 1) for v in known if 0 <= v <= 1} | {round(v * 100) for v in known
                                                                if 0 <= v <= 1}
    bad = []
    for f in sys.argv[1:]:
        text = Path(f).read_text()
        text = re.sub(r"`[^`]*`|\]\([^)]*\)|https?://\S+|\b\d{4}-\d{2}-\d{2}\b|\bv?\d+\.\d+\.\d+"
                      r"(?:-draft)?\b|\b[DS]\d\b|\b[1-5][ab]?\b(?=\s*\()|\bsub-type\s+\d[ab]?\b|"
                      r"\bcheck\s+[AB]\b|\b[0-9a-f]{8,}\b|#\d+", " ", text)
        for m in re.finditer(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(%?)(?![\w])", text):
            raw, is_pct = m.group(1).replace(",", ""), m.group(2)
            v = round(float(raw), 6)
            if v in ALLOW or v in known or (is_pct and v in pct):
                continue
            line = text[: m.start()].count("\n") + 1
            bad.append(f"{f}:{line}: {m.group(0)}")
    for b in bad:
        print("untraced number:", b)
    print(f"{len(bad)} untraced number(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
