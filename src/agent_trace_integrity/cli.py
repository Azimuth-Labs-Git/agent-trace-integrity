"""Command line: ``ati check``, ``ati events``, ``ati fetch``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .adapters import FORMATS, load
from .events import check_pairing


def _cmd_events(args: argparse.Namespace) -> int:
    runs = load(args.input, args.format)
    out = open(args.out, "w") if args.out else sys.stdout
    bad = 0
    for run in runs:
        problems = check_pairing(run)
        bad += bool(problems)
        for e in run:
            out.write(json.dumps(e.to_dict()) + "\n")
        for p in problems:
            print(f"pairing: {p}", file=sys.stderr)
    if out is not sys.stdout:
        out.close()
    print(f"{len(runs)} run(s), {sum(map(len, runs))} events, {bad} with pairing problems",
          file=sys.stderr)
    return 1 if bad else 0


def _cmd_check(args: argparse.Namespace) -> int:
    from collections import Counter

    from .verdicts import check_run

    runs = load(args.input, args.format)
    out = open(args.out, "w") if args.out else sys.stdout
    tally: Counter = Counter()
    for run in runs:
        for v in check_run(run, args.scope):
            tally[v["verdict"]] += 1
            for f in v["flags"]:
                tally[f] += 1
            out.write(json.dumps(v) + "\n")
    if out is not sys.stdout:
        out.close()
    print(f"{len(runs)} run(s); " + ", ".join(f"{k} {n}" for k, n in sorted(tally.items())),
          file=sys.stderr)
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .fetch import fetch, sha256_file

    path = fetch(args.url, args.sha256)
    print(f"{path}\t{sha256_file(path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ati", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", _cmd_check), ("events", _cmd_events)):
        s = sub.add_parser(name)
        s.add_argument("input")
        s.add_argument("--format", required=True, choices=sorted(FORMATS))
        s.add_argument("--out")
        if name == "check":
            s.add_argument("--scope", choices=("final", "all"), default="final",
                           help="read only the closing narration (default) or every message")
        s.set_defaults(fn=fn)
    s = sub.add_parser("fetch")
    s.add_argument("url")
    s.add_argument("--sha256")
    s.set_defaults(fn=_cmd_fetch)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
