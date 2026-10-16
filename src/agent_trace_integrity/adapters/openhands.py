"""Adapter for OpenHands rows of nebius/SWE-rebench-openhands-trajectories (D2, CC-BY-4.0).

Dataset: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories, checked at
revision 35455389ab51bf5e2306bfd436ef72d0f98bf882 (one ``train`` split, 67,074 rows, OpenHands
v0.54.0 driving Qwen3-Coder-480B-A35B-Instruct). Each row has ``trajectory_id``, ``instance_id``,
``repo``, ``trajectory`` (OpenAI-style messages), ``tools``, ``model_patch``, ``exit_status``,
``resolved``, ``gen_tests_correct`` and ``pred_passes_gen_tests``.

Accepted inputs, one run per row:

- a JSONL file, one row per line;
- a single JSON row (an object with a ``trajectory`` list), or a JSON list of rows;
- a datasets-server ``/rows`` response (``{"rows": [{"row_idx": n, "row": {...}}, ...]}``).

Mapping:

- ``system`` and ``user`` messages become ``system`` and ``user`` events.
- An ``assistant`` message becomes one ``assistant`` event when its text is not blank (or when it
  has no tool calls), then one ``tool_call`` event per entry in ``tool_calls``, in order.
  ``tool_args`` is the parsed JSON of ``function.arguments``, or the raw string if it does not
  parse. Assistant text that only *looks* like a tool call (a malformed ``<function=...>`` block
  the harness did not parse) stays text: the harness never ran it.
- A ``tool`` message becomes a ``tool_result`` paired by ``tool_call_id``. If the source lacks
  ids, calls get ``"{run}-call-{n}"`` and results pair to the oldest open call.
- Status:
  - ``execute_bash``: ``exit_code`` from the last ``[Command finished with exit code N]`` line
    (fallback ``[The command completed with exit code N...]``). Non-zero is ``error``. If the
    output carries ``[... Observation truncated due to length ...]`` and the exit code is not
    non-zero, status is ``truncated``. A command still running (``no new output after N
    seconds`` or ``timed out after``) has no exit code and status ``unknown``.
  - ``str_replace_editor``, ``think``, ``task_tracker``: ``error`` if the text starts with
    ``ERROR:``; ``truncated`` if it contains ``<response clipped>``; otherwise ``ok``.
  - Any result containing ``[Error occurred in processing last action]`` is ``error``.
  - Other tools: ``unknown`` unless the text starts with ``ERROR:``.
  Whenever a truncation marker is present, ``meta["truncated"] = True`` is also set, so a
  truncated result with exit code 1 still reads as ``error`` but keeps the flag.
- Calls with no observation in the source get a placeholder ``tool_result`` directly after the
  call with ``tool_result=None`` (not logged), ``status="unknown"`` and
  ``meta={"placeholder": "no_observation_in_source"}``. In D2 this is every ``finish`` call (the
  harness stops and logs nothing) and the last call of runs that hit the iteration limit.
- ``source_ref`` is ``"<where>:trajectory[i]"`` or ``"<where>:trajectory[i].tool_calls[k]"``,
  where ``<where>`` is ``line N`` (JSONL, 1-based), ``row_idx N`` (datasets-server) or ``row N``
  (position in a JSON list).

The first event's ``meta`` holds the row fields that identify and score the run, when present:
``format``, ``trajectory_id``, ``instance_id``, ``repo``, ``exit_status``, ``resolved``,
``gen_tests_correct``, ``pred_passes_gen_tests`` and ``row_idx``. ``model_patch`` and ``tools``
are not copied (large, and not part of the event record).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..events import Event, reindex

FORMAT = "openhands"

META_FIELDS = (
    "trajectory_id",
    "instance_id",
    "repo",
    "exit_status",
    "resolved",
    "gen_tests_correct",
    "pred_passes_gen_tests",
)

_EXIT_FINISHED = re.compile(r"^\[Command finished with exit code (-?\d+)\]\s*$", re.M)
_EXIT_COMPLETED = re.compile(r"^\[The command completed with exit code (-?\d+)\b", re.M)
_STILL_RUNNING = re.compile(
    r"\[The command has no new output after \d+ seconds|\[The command timed out after"
)
_BASH_TRUNC = "[... Observation truncated due to length ...]"
_EDITOR_TRUNC = "<response clipped>"
_ACTION_ERROR = "[Error occurred in processing last action]"
_TEXT_TOOLS = ("str_replace_editor", "think", "task_tracker")


def _read_rows(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return (where, row) pairs from any accepted input shape."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            doc = None  # several objects, one per line: JSONL
        if isinstance(doc, dict):
            if isinstance(doc.get("rows"), list):
                return [
                    (f"row_idx {r.get('row_idx', i)}", {**r["row"], "row_idx": r.get("row_idx", i)})
                    for i, r in enumerate(doc["rows"])
                ]
            return [("row 0", doc)]
        if isinstance(doc, list):
            return [(f"row {i}", r) for i, r in enumerate(doc)]
    rows = []
    for n, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            rows.append((f"line {n}", json.loads(line)))
    return rows


def _parse_args(raw: Any) -> dict[str, Any] | str | None:
    if raw is None or isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw
    return parsed if isinstance(parsed, dict) else raw


def classify(tool_name: str | None, text: str | None) -> tuple[int | None, str, bool]:
    """Return (exit_code, status, truncated) for one observation."""
    if text is None:
        return None, "unknown", False
    if tool_name == "execute_bash":
        found = _EXIT_FINISHED.findall(text) or _EXIT_COMPLETED.findall(text)
        code = int(found[-1]) if found else None
        truncated = _BASH_TRUNC in text
        if _ACTION_ERROR in text or (code is not None and code != 0):
            return code, "error", truncated
        if truncated:
            return code, "truncated", True
        if code is None or _STILL_RUNNING.search(text):
            return code, "unknown", False
        return code, "ok", False
    truncated = _EDITOR_TRUNC in text or _BASH_TRUNC in text
    if text.startswith("ERROR:") or _ACTION_ERROR in text:
        return None, "error", truncated
    if tool_name in _TEXT_TOOLS:
        return None, ("truncated" if truncated else "ok"), truncated
    return None, ("truncated" if truncated else "unknown"), truncated


def row_to_events(row: dict[str, Any], where: str = "row 0") -> list[Event]:
    run = str(row.get("trajectory_id") or row.get("instance_id") or where)
    events: list[Event] = []
    open_calls: list[str] = []  # ids awaiting a result, oldest first
    call_names: dict[str, str | None] = {}
    answered: set[str] = set()
    n_calls = 0
    for i, msg in enumerate(row.get("trajectory") or []):
        role = msg.get("role")
        content = msg.get("content")
        ref = f"{where}:trajectory[{i}]"
        if role in ("system", "user"):
            events.append(Event(0, role, text=content, source_ref=ref))
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            if (content is not None and content.strip()) or not calls:
                events.append(Event(0, "assistant", text=content, source_ref=ref))
            for k, call in enumerate(calls):
                fn = call.get("function") or {}
                cid = call.get("id") or f"{run}-call-{n_calls}"
                n_calls += 1
                open_calls.append(cid)
                call_names[cid] = fn.get("name")
                events.append(Event(
                    0, "tool_call", tool_name=fn.get("name"),
                    tool_args=_parse_args(fn.get("arguments")), tool_call_id=cid,
                    source_ref=f"{ref}.tool_calls[{k}]",
                ))
        elif role == "tool":
            cid = msg.get("tool_call_id")
            if not cid and open_calls:
                cid = open_calls[0]
            if cid in open_calls:
                open_calls.remove(cid)
            name = msg.get("name") or call_names.get(cid)
            code, status, truncated = classify(name, content)
            answered.add(cid)
            events.append(Event(
                0, "tool_result", tool_name=name, tool_call_id=cid, tool_result=content,
                exit_code=code, status=status, source_ref=ref,
                meta={"truncated": True} if truncated else {},
            ))
        else:
            raise ValueError(f"{where}: unknown message role {role!r} at trajectory[{i}]")

    # Placeholder results for calls the source never answered (finish, iteration limit).
    out: list[Event] = []
    for e in events:
        out.append(e)
        if e.role == "tool_call" and e.tool_call_id not in answered:
            out.append(Event(
                0, "tool_result", tool_name=e.tool_name, tool_call_id=e.tool_call_id,
                tool_result=None, status="unknown", source_ref=e.source_ref,
                meta={"not_logged": True, "placeholder": "no_observation_in_source"},
            ))
    reindex(out)
    if out:
        meta = {"format": FORMAT}
        meta.update({k: row[k] for k in (*META_FIELDS, "row_idx") if k in row})
        out[0].meta = {**meta, **out[0].meta}
    return out


def load(path: str | Path) -> list[list[Event]]:
    return [row_to_events(row, where) for where, row in _read_rows(Path(path))]
