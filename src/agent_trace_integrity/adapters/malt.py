"""METR MALT transcripts (D4): https://huggingface.co/datasets/metr-evals/malt-transcripts-public

MIT-licensed, gated, with a do-not-train canary: rows are read at runtime (on Modal in this
repo) and never copied into the repo. Each row is one run with a message DAG in ``nodes``
(``node_id``, ``parent_node_id``, ``branch_id``, ``node_data.message`` with ``role``,
``content``, ``name``, ``function_call``).

Mapping:

- The run's path is the root-to-leaf path (through ``parent_node_id``) with the most actions,
  then the most nodes. Side branches (token-budget notes, side questions such as "which file
  should I write to?", advisor turns) are not on it.
- An assistant message becomes an ``assistant`` event (its text), then one ``tool_call`` per
  action, in this order of preference:
  - a structured ``function_call`` (``name``, JSON ``arguments``);
  - a ``function_call: {...}`` JSON object written in the text;
  - ``<bash>...</bash>`` or ``<python>...</python>`` blocks in the text.
  Bash commands go in ``tool_args["command"]``, Python code in ``tool_args["code"]``.
- A tool result is a ``function``-role message, or a user message that starts with
  ``<NAME-output>``. It is paired with the oldest open call. ``Exit code: N`` sets ``exit_code``.
- A call with no result on the path gets a ``not_logged`` placeholder.

The first event's ``meta`` holds ``run_id``, ``task_id``, ``model``, ``labels``,
``manually_reviewed`` and ``run_source``. Load with ``load(path)`` (a JSON list of rows or
JSONL) or ``parse_row(row)`` for a row dict already in memory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..events import Event, reindex, status_from_exit_code

FORMAT = "malt"

_TAG = re.compile(r"<(bash|python)>(.*?)</\1>", re.S)
_TEXT_FC = re.compile(r"function_call:\s*(\{.*\})\s*$", re.S)
_OUTPUT = re.compile(r"^\s*<([\w-]+)-output>", re.S)
_EXIT = re.compile(r"Exit code:\s*(-?\d+)")


def _path(nodes: list[dict]) -> list[dict]:
    """The run's main trajectory: the root-to-leaf path with the most actions, then the most
    nodes. (Branch 0 is not always the agent: in some runs it is a short side question.)"""
    by_id = {n["node_id"]: n for n in nodes}
    parents = {n.get("parent_node_id") for n in nodes}
    leaves = [n for n in nodes if n["node_id"] not in parents] or nodes[-1:]

    def walk(leaf):
        out, cur, seen = [], leaf, set()
        while cur is not None and cur["node_id"] not in seen:
            seen.add(cur["node_id"])
            out.append(cur)
            cur = by_id.get(cur.get("parent_node_id"))
        return out[::-1]

    def score(path):
        acts = sum(bool(_actions((n.get("node_data") or {}).get("message") or {}))
                   for n in path if ((n.get("node_data") or {}).get("message") or {})
                   .get("role") == "assistant")
        return (acts, len(path))

    return max((walk(leaf) for leaf in leaves), key=score) if leaves else []


def _args(name: str, raw: Any) -> dict | str:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw if isinstance(raw, dict) else {"value": raw}


def _actions(msg: dict) -> list[tuple[str, dict | str]]:
    fc = msg.get("function_call") or {}
    if fc.get("name"):
        return [(fc["name"], _args(fc["name"], fc.get("arguments")))]
    text = msg.get("content") or ""
    m = _TEXT_FC.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                return [(obj["name"], _args(obj["name"], obj.get("arguments")))]
        except json.JSONDecodeError:
            pass
    acts = []
    for kind, body in _TAG.findall(text):
        acts.append(("bash" if kind == "bash" else "python",
                     {"command": body.strip()} if kind == "bash" else {"code": body.strip()}))
    return acts


def parse_row(row: dict, run: str | None = None) -> list[Event]:
    run = run or f"malt-{row.get('run_id')}"
    events: list[Event] = []
    pending: list[Event] = []
    n_calls = 0
    for node in _path(row.get("nodes") or []):
        msg = (node.get("node_data") or {}).get("message") or {}
        role, text = msg.get("role"), msg.get("content") or ""
        ref = f"node {node.get('node_id')}"
        out = _OUTPUT.match(text) if role == "user" else None
        if role == "function" or out:
            if pending:
                call = pending.pop(0)
                code = _EXIT.search(text)
                exit_code = int(code.group(1)) if code else None
                events.append(Event(0, "tool_result", tool_name=call.tool_name,
                                    tool_call_id=call.tool_call_id, tool_result=text,
                                    exit_code=exit_code,
                                    status=status_from_exit_code(exit_code) if code else "unknown",
                                    source_ref=ref))
            continue
        if role in ("system", "developer"):
            events.append(Event(0, "system", text=text, source_ref=ref))
        elif role == "user":
            events.append(Event(0, "user", text=text, source_ref=ref))
        elif role == "assistant":
            for call in pending:  # unanswered calls before the next turn
                events.append(Event(0, "tool_result", tool_name=call.tool_name,
                                    tool_call_id=call.tool_call_id, status="unknown",
                                    meta={"not_logged": True}))
            pending = []
            visible = _TEXT_FC.sub("", text).strip()
            events.append(Event(0, "assistant", text=visible, source_ref=ref))
            for name, args in _actions(msg):
                n_calls += 1
                call = Event(0, "tool_call", tool_name=name, tool_args=args,
                             tool_call_id=f"{run}-call-{n_calls}", source_ref=ref)
                events.append(call)
                pending.append(call)
    for call in pending:
        events.append(Event(0, "tool_result", tool_name=call.tool_name,
                            tool_call_id=call.tool_call_id, status="unknown",
                            meta={"not_logged": True}))
    if events:
        events[0].meta.update({k: row.get(k) for k in ("run_id", "task_id", "model", "labels",
                                                        "manually_reviewed", "run_source")})
        events[0].meta["format"] = FORMAT
    return reindex(events)


def load(path: Path) -> list[list[Event]]:
    text = Path(path).read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(x) for x in text.splitlines() if x.strip()]
    return [parse_row(r) for r in rows]
