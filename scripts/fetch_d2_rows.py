"""Fetch D2 rows by offset into a local JSONL file (outside the repo) and write their pins.

Usage: python scripts/fetch_d2_rows.py OFFSET COUNT OUT.jsonl
"""

import json
import sys

from agent_trace_integrity.fetch import fetch, sha256_file

URL = ("https://datasets-server.huggingface.co/rows?dataset=nebius/"
       "SWE-rebench-openhands-trajectories&config=default&split=train&offset={o}&length={n}")


def main() -> None:
    off, count, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    pins, rows = [], []
    for o in range(off, off + count, 20):
        url = URL.format(o=o, n=min(20, off + count - o))
        p = fetch(url)
        pins.append({"url": url, "sha256": sha256_file(p)})
        rows += json.load(open(p))["rows"]
    with open(out, "w") as f:
        for r in rows:
            row = r["row"]
            row["_row_idx"] = r["row_idx"]
            f.write(json.dumps(row) + "\n")
    json.dump(pins, open(out + ".pins.json", "w"), indent=1)
    print(len(rows))


if __name__ == "__main__":
    main()
