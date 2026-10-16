"""Terminate the Lambda instance recorded in local/lambda-instance.json and verify it is gone.

Calls POST /instance-operations/terminate for that id, then polls GET /instances until the id is
no longer listed, then prints "terminated, verified". --dry-run prints the calls and sends nothing.
A real call needs --i-have-matts-yes and the LAMBDA_API_KEY env var.

Usage:
  python scripts/lambda/teardown.py --dry-run
  python scripts/lambda/teardown.py --i-have-matts-yes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lambda_api as api

PLACEHOLDER = "<instance id from local/lambda-instance.json>"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="print the calls, send nothing")
    p.add_argument("--i-have-matts-yes", action="store_true",
                   help="Matt said yes to stopping this instance")
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--timeout-seconds", type=float, default=900.0)
    p.add_argument("--local-dir", type=Path, default=api.LOCAL_DIR, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def read_instance_id(record_path: Path) -> str | None:
    if not record_path.exists():
        return None
    return json.loads(record_path.read_text()).get("instance_id")


def listed_ids(resp: dict) -> set[str]:
    return {inst.get("id") for inst in resp.get("data", [])}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record_path = args.local_dir / api.INSTANCE_FILE_NAME
    instance_id = read_instance_id(record_path)
    shown = instance_id or PLACEHOLDER
    print(f"POST {api.API_BASE}/instance-operations/terminate")
    print("body: " + json.dumps({"instance_ids": [shown]}))
    print(f"then GET {api.API_BASE}/instances every {args.poll_seconds:g}s "
          f"(up to {args.timeout_seconds:g}s) until {shown} is absent")
    if args.dry_run:
        if instance_id is None:
            print(f"note: {record_path} not found; a real run would refuse")
        print("dry run: no request sent")
        return 0
    if not args.i_have_matts_yes:
        print(api.NO_YES_MESSAGE, file=sys.stderr)
        return api.EXIT_NO_YES
    if instance_id is None:
        print(f"refusing: no instance id in {record_path}. Check the Lambda dashboard by hand.",
              file=sys.stderr)
        return api.EXIT_BAD_INPUT
    key = api.api_key()
    if key is None:
        print(f"refusing: set {api.KEY_ENV} in the environment (never in the repo)",
              file=sys.stderr)
        return api.EXIT_NO_KEY

    api.api_call("POST", "/instance-operations/terminate", key, {"instance_ids": [instance_id]})
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        if instance_id not in listed_ids(api.api_call("GET", "/instances", key)):
            break
        if time.monotonic() > deadline:
            print(f"NOT VERIFIED: {instance_id} still listed after {args.timeout_seconds:g}s. "
                  "Check the Lambda dashboard now; it may still be billing.", file=sys.stderr)
            return 1
        time.sleep(args.poll_seconds)
    done = record_path.with_name("lambda-instance.terminated.json")
    record_path.replace(done)
    print(f"{instance_id}: terminated, verified")
    print(f"record moved to {done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
