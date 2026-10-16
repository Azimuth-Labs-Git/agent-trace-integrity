"""Pin every Palisade shutdown-avoidance Inspect log (D3) at a fixed repo commit.

Writes data/pins/d3.json (URL, sha256, model per log). Logs are fetched at runtime, never
committed. Usage: python scripts/pin_d3.py NAMES.txt (one .eval file name per line, from the
GitHub contents API at the commit below).
"""

import json
import sys
from pathlib import Path

from agent_trace_integrity.adapters import load
from agent_trace_integrity.fetch import fetch, sha256_file

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "df0ce52e90aed38e8721b54058a5d46923a34583"
BASE = (f"https://raw.githubusercontent.com/PalisadeResearch/shutdown_avoidance/{COMMIT}/"
        "docs/inspect_logs/logs/")


def main() -> None:
    logs = []
    for name in Path(sys.argv[1]).read_text().split():
        url = BASE + name
        p = fetch(url)
        runs = load(p, "inspect")
        model = runs[0][0].meta.get("model") if runs else None
        logs.append({"file": name, "url": url, "sha256": sha256_file(p), "model": model,
                     "samples": len(runs)})
        print(name, model, len(runs))
    out = {"repo": "PalisadeResearch/shutdown_avoidance", "commit": COMMIT, "logs": logs}
    (ROOT / "data" / "pins" / "d3.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
