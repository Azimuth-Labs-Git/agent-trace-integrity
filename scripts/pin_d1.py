"""Pin the D1 headline sample: 4 SWE-bench bash-only submissions on mini-swe-agent v1.x (one
template), 75 tasks each, chosen before any checker result was seen.

Rule (fixed 2026-10-15, before results): intersect the instance ids in the four submissions'
per_instance_details.json, order by sha256(instance_id) hex, take the first 75.

Writes data/pins/d1.json with URLs and sha256 only (trajectories are fetched at runtime and never
committed). Usage: python scripts/pin_d1.py
"""

import hashlib
import json
import urllib.error
from pathlib import Path

from agent_trace_integrity.fetch import fetch, sha256_file

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"  # SWE-bench/experiments main, 2026-09-03
SUBMISSIONS = {
    "Anthropic": "20251124_mini-v1.16.0_claude-opus-4-5-20251101",
    "OpenAI": "20251211_mini-v1.17.2_gpt-5.2-2025-12-11",
    "Google": "20251118_mini-v1.15.0_gemini-3-pro-preview-20251118",
    "open-weight": "20251201_mini-v1.17.1_deepseek-v3.2-reasoner",
}
N = 75
RAW = f"https://raw.githubusercontent.com/SWE-bench/experiments/{COMMIT}/evaluation/verified"
S3 = "https://swe-bench-submissions.s3.amazonaws.com/bash-only"


def main() -> None:
    details, ids = {}, None
    for sub in SUBMISSIONS.values():
        url = f"{RAW}/{sub}/per_instance_details.json"
        p = fetch(url)
        details[sub] = {"url": url, "sha256": sha256_file(p)}
        these = set(json.loads(p.read_text()))
        ids = these if ids is None else ids & these
    chosen = sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:N]
    out = {"rule": "intersect per_instance_details ids; order by sha256(instance_id); first 75",
           "experiments_commit": COMMIT, "n_per_model": N, "instance_ids": chosen,
           "submissions": {}}
    for prov, sub in SUBMISSIONS.items():
        runs = []
        for i in chosen:
            t = f"{S3}/{sub}/trajs/{i}/{i}.traj.json"
            r = f"{S3}/{sub}/logs/{i}/report.json"
            try:
                rsha = sha256_file(fetch(r))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                rsha = None  # no harness report (e.g. empty patch); resolved comes from details
            runs.append({"instance_id": i, "url": t, "sha256": sha256_file(fetch(t)),
                         "report_url": r if rsha else None, "report_sha256": rsha})
        out["submissions"][sub] = {"provider_group": prov, "details": details[sub], "runs": runs}
        print(sub, len(runs))
    (ROOT / "data" / "pins" / "d1.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
