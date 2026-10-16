"""Launch one Lambda Cloud GPU instance for the open-weight LLM judge (plan sections 3 and 4).

Always run with --dry-run first. It prints the exact request body, instance type, region, hours
and estimated cost (hours x price per hour), and refuses with exit 2 if the cost is over the cap.

A real launch also needs --i-have-matts-yes (Matt has said yes to this instance type and these
hours) and the LAMBDA_API_KEY env var. It prints the same summary, then calls
POST /instance-operations/launch, and writes the instance id to local/lambda-instance.json
(local/ is gitignored).

The price per hour must be passed: read it off Lambda's pricing page at launch and write it into
docs/build-log.md. This script does not look it up.

Usage:
  python scripts/lambda/launch.py --instance-type gpu_1x_a10 --region us-east-1 \
      --ssh-key matt-laptop --hours 3 --price-per-hour 0.75 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import lambda_api as api

DEFAULT_CAP = 150.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--instance-type", required=True, help="e.g. gpu_1x_a10 (check at launch)")
    p.add_argument("--region", required=True, help="e.g. us-east-1 (check capacity at launch)")
    p.add_argument("--ssh-key", required=True, help="name of an SSH key already on the account")
    p.add_argument("--hours", type=float, required=True, help="expected hours, Matt approves")
    p.add_argument("--price-per-hour", type=float, required=True,
                   help="USD per hour from Lambda's pricing page, checked at launch")
    p.add_argument("--cap", type=float, default=DEFAULT_CAP, help="USD cap (default 150)")
    p.add_argument("--dry-run", action="store_true", help="print the request, call nothing")
    p.add_argument("--i-have-matts-yes", action="store_true",
                   help="Matt said yes to this instance type and these hours")
    p.add_argument("--local-dir", type=Path, default=api.LOCAL_DIR, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def request_body(args: argparse.Namespace) -> dict:
    return {
        "instance_type_name": args.instance_type,
        "region_name": args.region,
        "ssh_key_names": [args.ssh_key],
        "quantity": 1,
    }


def summary(args: argparse.Namespace, body: dict, cost: float) -> str:
    return "\n".join([
        f"POST {api.API_BASE}/instance-operations/launch",
        "body: " + json.dumps(body, indent=2),
        f"instance type:   {args.instance_type}",
        f"region:          {args.region}",
        f"hours:           {args.hours:g}",
        f"price per hour:  ${args.price_per_hour:.2f} (from Lambda's pricing page; "
        "write it in docs/build-log.md)",
        f"estimated cost:  ${cost:.2f} = {args.hours:g} h x ${args.price_per_hour:.2f}/h",
        f"cap:             ${args.cap:.2f}",
    ])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.hours <= 0 or args.price_per_hour <= 0:
        print("refusing: --hours and --price-per-hour must be positive", file=sys.stderr)
        return api.EXIT_BAD_INPUT
    body = request_body(args)
    cost = args.hours * args.price_per_hour
    print(summary(args, body, cost))
    if cost > args.cap:
        print(f"refusing: estimated cost ${cost:.2f} is over the ${args.cap:.2f} cap",
              file=sys.stderr)
        return api.EXIT_OVER_CAP
    if args.dry_run:
        print("dry run: no request sent")
        return 0
    if not args.i_have_matts_yes:
        print(api.NO_YES_MESSAGE, file=sys.stderr)
        return api.EXIT_NO_YES
    key = api.api_key()
    if key is None:
        print(f"refusing: set {api.KEY_ENV} in the environment (never in the repo)",
              file=sys.stderr)
        return api.EXIT_NO_KEY
    record_path = args.local_dir / api.INSTANCE_FILE_NAME
    if record_path.exists():
        print(f"refusing: {record_path} already exists; run teardown.py first so two "
              "instances are never running", file=sys.stderr)
        return api.EXIT_BAD_INPUT

    resp = api.api_call("POST", "/instance-operations/launch", key, body)
    ids = resp.get("data", {}).get("instance_ids", [])
    if len(ids) != 1:
        print("unexpected launch response: " + json.dumps(resp), file=sys.stderr)
        return 1
    args.local_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "instance_id": ids[0],
        "instance_type": args.instance_type,
        "region": args.region,
        "hours": args.hours,
        "price_per_hour": args.price_per_hour,
        "estimated_cost": round(cost, 2),
        "launched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"launched {ids[0]}; recorded in {record_path}")
    print("next: get the instance IP from the Lambda dashboard, then serve_vllm.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
