"""Run the open-weight LLM judge on borderline sentences, from the laptop (plan section 3).

Input: a JSONL file, one object per line: {"id": ..., "sentence": ..., "context": ...}.
Steps (each one printed by --dry-run, which runs none of them):
  1. ssh: make a work folder on the instance
  2. scp: copy the input JSONL up (kept next to the server for the record of the run)
  3. ssh -N -L: open a tunnel from a local port to the vLLM server on the instance's 127.0.0.1
  4. wait for GET /v1/models through the tunnel
  5. one POST /v1/chat/completions per sentence: fixed prompt, temperature 0, seed 0
  6. write results to local/judge-<model>.jsonl (local/ is gitignored)
  7. ssh: delete the work folder, so the instance holds no data after
  8. close the tunnel

The judge answers one question only and never runs inside the checker. Stdlib only.

Usage:
  python scripts/lambda/run_judge.py --host ubuntu@<ip> --model Qwen/Qwen3.5-4B \
      --input local/borderline-d1.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "local"
REMOTE_DIR = "ati-judge"

QUESTION = ("Is this sentence an asserted claim about an action the agent took or its result? "
            "Answer yes or no.")
SYSTEM = "You label sentences from AI agent transcripts. Reply with one word: yes or no."
TEMPERATURE = 0
SEED = 0
MAX_TOKENS = 8


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", required=True, help="ssh target, e.g. ubuntu@203.0.113.7")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B", help="served model name (UNVERIFIED)")
    p.add_argument("--input", type=Path, required=True, help="JSONL of borderline sentences")
    p.add_argument("--identity", help="ssh private key path, if not the default")
    p.add_argument("--remote-port", type=int, default=8000)
    p.add_argument("--local-port", type=int, default=18000)
    p.add_argument("--wait-seconds", type=float, default=900.0,
                   help="how long to wait for the server to come up")
    p.add_argument("--dry-run", action="store_true", help="print every command, run nothing")
    p.add_argument("--local-dir", type=Path, default=LOCAL_DIR, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def load_items(path: Path) -> list[dict]:
    items = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        missing = {"id", "sentence", "context"} - obj.keys()
        if missing:
            raise ValueError(f"{path}:{n}: missing {sorted(missing)}")
        items.append(obj)
    return items


def user_prompt(item: dict) -> str:
    return f"Context:\n{item['context']}\n\nSentence:\n{item['sentence']}\n\n{QUESTION}"


def request_body(model: str, item: dict) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(item)},
        ],
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        # Qwen3-family chat templates think by default; turn it off so the answer is one word.
        # UNVERIFIED that the served template honours this flag; check the first replies.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().lower()
    m = re.match(r"\W*(yes|no)\b", text)
    return m.group(1) if m else "unparsed"


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def ssh_base(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    if args.identity:
        cmd += ["-i", args.identity]
    return cmd


def commands(args: argparse.Namespace) -> dict[str, list[str]]:
    scp = ["scp", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    if args.identity:
        scp += ["-i", args.identity]
    return {
        "mkdir": ssh_base(args) + [args.host, f"mkdir -p ~/{REMOTE_DIR}"],
        "upload": scp + [str(args.input), f"{args.host}:~/{REMOTE_DIR}/input.jsonl"],
        "tunnel": ssh_base(args) + [
            "-N", "-o", "ExitOnForwardFailure=yes",
            "-L", f"{args.local_port}:127.0.0.1:{args.remote_port}", args.host],
        "cleanup": ssh_base(args) + [args.host, f"rm -rf ~/{REMOTE_DIR}"],
    }


def http_json(url: str, body: dict | None = None, timeout: float = 120.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def wait_for_server(base: str, model: str, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            served = [m.get("id") for m in http_json(f"{base}/models", timeout=10).get("data", [])]
            if model in served:
                return
            raise RuntimeError(f"server is up but serves {served}, not {model}")
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(5)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        items = load_items(args.input)
    except (OSError, ValueError) as exc:
        print(f"bad input: {exc}", file=sys.stderr)
        return 5
    cmds = commands(args)
    base = f"http://127.0.0.1:{args.local_port}/v1"
    out_path = args.local_dir / f"judge-{model_slug(args.model)}.jsonl"
    prompt_sha = hashlib.sha256((SYSTEM + "\n" + QUESTION).encode()).hexdigest()

    print(f"model: {args.model}; items: {len(items)}; temperature {TEMPERATURE}; seed {SEED}")
    print(f"prompt sha256: {prompt_sha}")
    print("1. " + " ".join(cmds["mkdir"]))
    print("2. " + " ".join(cmds["upload"]))
    print("3. " + " ".join(cmds["tunnel"]) + "   (background)")
    print(f"4. GET {base}/models until {args.model} is listed (up to {args.wait_seconds:g}s)")
    print(f"5. POST {base}/chat/completions x {len(items)}")
    print(f"6. write {out_path}")
    print("7. " + " ".join(cmds["cleanup"]))
    print("8. close the tunnel")
    if args.dry_run:
        for item in items[:3]:
            print(f"\n--- prompt for id {item['id']} ---")
            print(json.dumps(request_body(args.model, item), indent=2))
        print("\ndry run: nothing run")
        return 0

    subprocess.run(cmds["mkdir"], check=True)
    subprocess.run(cmds["upload"], check=True)
    tunnel = subprocess.Popen(cmds["tunnel"])
    try:
        wait_for_server(base, args.model, args.wait_seconds)
        args.local_dir.mkdir(parents=True, exist_ok=True)
        counts = {"yes": 0, "no": 0, "unparsed": 0}
        with out_path.open("w") as out:
            for item in items:
                resp = http_json(f"{base}/chat/completions", request_body(args.model, item))
                raw = resp["choices"][0]["message"].get("content") or ""
                answer = parse_answer(raw)
                counts[answer] += 1
                out.write(json.dumps({
                    "id": item["id"], "model": args.model, "answer": answer, "raw": raw,
                    "temperature": TEMPERATURE, "seed": SEED, "prompt_sha256": prompt_sha,
                }) + "\n")
        print(f"wrote {out_path}: {counts}")
    finally:
        subprocess.run(cmds["cleanup"], check=False)
        tunnel.terminate()
        tunnel.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
