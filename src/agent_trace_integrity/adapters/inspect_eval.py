"""Adapter for Inspect AI ``.eval`` log files.

An ``.eval`` file is a zip archive. This adapter reads it with the standard library only (no
``inspect_ai`` import). It uses:

- ``header.json`` (or ``_journal/start.json`` for a log that never finished) for the eval spec:
  task, model, run and eval ids.
- ``samples/<id>_epoch_<n>.json``, one per sample and epoch. Each becomes one run.

Message mapping (``sample["messages"]``, in order):

- ``system`` and ``user`` messages become ``system`` and ``user`` events.
- An ``assistant`` message becomes one ``assistant`` event holding its visible text (string
  content, or the ``text`` parts of list content joined by newlines), followed by one
  ``tool_call`` event per entry in ``tool_calls``: ``tool_name`` = ``function``, ``tool_args`` =
  ``arguments``, ``tool_call_id`` = ``id``. The assistant event is emitted when there is text, or
  when the message has neither text nor tool calls (then with ``text=""``). Reasoning parts are
  not visible text; when non-empty they are kept in ``meta["reasoning"]`` of the assistant event,
  or of the first tool call when no assistant event was emitted.
- A ``tool`` message becomes a ``tool_result`` paired to its call by ``tool_call_id``.
  ``status`` is ``error`` when Inspect recorded an ``error`` on the message (its type and message
  are kept in ``meta["error"]``), ``truncated`` when the text carries Inspect's output truncation
  notice, ``unknown`` for shell tools (``bash``, ``python``) because Inspect's shell tools do not
  record an exit code and do not fail on a non-zero exit, and ``ok`` for other tools, which
  Inspect marks with ``error`` when they raise. ``exit_code`` stays ``None``: Inspect does not
  log one.
- A tool call with no tool message (for example the ``./next_task`` call during which the
  sandbox was shut down, which ends the sample) gets a ``tool_result`` at the end of the run with
  ``tool_result=None`` (not logged), ``status="unknown"`` and ``meta["missing_result"]=True``.
  Results stay in call order.
- Some providers (Gemini through OpenRouter in Palisade's logs) reuse a tool call id such as
  ``tool_0_bash`` on every turn. A repeated id gets a run-unique ``"<id>#<k>"`` (k = 2, 3, ...)
  on the call and its result, with the logged id kept in ``meta["source_tool_call_id"]``; a
  result pairs with the oldest unanswered call carrying its logged id.
- ``attachment://<hash>`` references inside message text or arguments are resolved from the
  sample's ``attachments`` table.

``source_ref`` is ``"<zip member>#messages[i]"`` or ``"<zip member>#messages[i].tool_calls[j]"``.

``meta`` of each run's first event holds:

- ``format``: ``"inspect"``; ``file``: the zip member name.
- ``sample_id``, ``epoch``, ``sample_uuid``, ``model`` (eval spec model, else the sample's
  output model), ``task``, ``eval_id``, ``run_id``.
- ``scores``: the sample's ``scores`` dict unchanged, scorer name to ``{value, answer,
  explanation, metadata}`` as logged (Palisade's logs use one scorer, ``exact``, value ``C`` or
  ``I``).
- ``target``, ``sample_error`` (the sample-level error, or ``None``) and ``limit`` (the limit
  that stopped the sample, or ``None``).

Runs are ordered by sample id, then epoch.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from ..events import Event, reindex

FORMAT = "inspect"

SHELL_TOOLS = {"bash", "python"}
TRUNCATION_MARKERS = (
    "was too long to be displayed",
    "<START_TOOL_OUTPUT>",
)
_SAMPLE_RE = re.compile(r"^samples/(.+)_epoch_(\d+)\.json$")
_ATTACH_RE = re.compile(r"attachment://([0-9a-fA-F]+)")


def _read_json(z: zipfile.ZipFile, name: str) -> Any:
    with z.open(name) as f:
        return json.load(f)


def _eval_spec(z: zipfile.ZipFile) -> dict[str, Any]:
    names = set(z.namelist())
    for name in ("header.json", "_journal/start.json"):
        if name in names:
            spec = _read_json(z, name).get("eval")
            if isinstance(spec, dict):
                return spec
    return {}


def _resolve(value: Any, attachments: dict[str, str]) -> Any:
    if not attachments:
        return value
    if isinstance(value, str):
        if "attachment://" not in value:
            return value
        return _ATTACH_RE.sub(lambda m: attachments.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_resolve(v, attachments) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, attachments) for k, v in value.items()}
    return value


def _text_and_reasoning(content: Any) -> tuple[str, str]:
    """Visible text and reasoning from a ChatMessage content (string or list of parts)."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return content, ""
    texts: list[str] = []
    reasoning: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            texts.append(part.get("text") or "")
        elif kind == "reasoning":
            r = part.get("reasoning") or part.get("summary") or ""
            if isinstance(r, str) and r:
                reasoning.append(r)
    return "\n".join(texts), "\n".join(reasoning)


def _result_status(msg: dict[str, Any], tool_name: str | None, text: str) -> str:
    if msg.get("error"):
        return "error"
    if any(m in text for m in TRUNCATION_MARKERS):
        return "truncated"
    if tool_name in SHELL_TOOLS:
        return "unknown"
    return "ok"


def _sample_key(sample_id: Any, epoch: Any) -> tuple:
    sid = (0, sample_id, "") if isinstance(sample_id, int) else (1, 0, str(sample_id))
    return (*sid, epoch if isinstance(epoch, int) else 0)


def _sample_to_events(sample: dict[str, Any], member: str, spec: dict[str, Any]) -> list[Event]:
    attachments = sample.get("attachments") or {}
    sample_id = sample.get("id")
    epoch = sample.get("epoch")
    run_name = f"{sample_id}_epoch_{epoch}"
    events: list[Event] = []
    call_names: dict[str, str | None] = {}
    call_refs: dict[str, str] = {}
    pending: list[str] = []
    raw_ids: dict[str, str] = {}  # run-unique id -> id as logged
    seen: dict[str, int] = {}
    n_generated = 0

    for i, msg in enumerate(sample.get("messages") or []):
        role = msg.get("role")
        ref = f"{member}#messages[{i}]"
        text, reasoning = _text_and_reasoning(_resolve(msg.get("content"), attachments))
        if role in ("system", "user"):
            events.append(Event(0, role, text=text, source_ref=ref))
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            reasoning_target: Event | None = None
            if text or not calls:
                ev = Event(0, "assistant", text=text, source_ref=ref)
                events.append(ev)
                reasoning_target = ev
            for j, call in enumerate(calls):
                raw = call.get("id")
                if not raw:
                    n_generated += 1
                    raw = f"{run_name}-call-{n_generated}"
                seen[raw] = seen.get(raw, 0) + 1
                cid = raw if seen[raw] == 1 else f"{raw}#{seen[raw]}"
                raw_ids[cid] = raw
                name = call.get("function")
                cref = f"{ref}.tool_calls[{j}]"
                ev = Event(
                    0,
                    "tool_call",
                    tool_name=name,
                    tool_args=_resolve(call.get("arguments"), attachments),
                    tool_call_id=cid,
                    source_ref=cref,
                )
                if cid != raw:
                    ev.meta["source_tool_call_id"] = raw
                if call.get("parse_error"):
                    ev.meta["parse_error"] = call["parse_error"]
                events.append(ev)
                call_names[cid] = name
                call_refs[cid] = cref
                pending.append(cid)
                if reasoning_target is None:
                    reasoning_target = ev
            if reasoning and reasoning_target is not None:
                reasoning_target.meta["reasoning"] = reasoning
        elif role == "tool":
            raw = msg.get("tool_call_id")
            if raw:
                # Some providers reuse ids across turns: pair with the oldest unanswered call
                # that carries this id.
                cid = next((c for c in pending if raw_ids[c] == raw), raw)
            else:
                # No id logged: pair with the oldest unanswered call.
                cid = pending[0] if pending else None
            name = msg.get("function") or call_names.get(cid)
            ev = Event(
                0,
                "tool_result",
                tool_name=name,
                tool_call_id=cid,
                tool_result=text,
                status=_result_status(msg, name, text),
                source_ref=ref,
            )
            if msg.get("error"):
                ev.meta["error"] = msg["error"]
            if raw and cid != raw:
                ev.meta["source_tool_call_id"] = raw
            events.append(ev)
            if cid in pending:
                pending.remove(cid)
        else:
            continue

    for cid in pending:
        events.append(
            Event(
                0,
                "tool_result",
                tool_name=call_names.get(cid),
                tool_call_id=cid,
                tool_result=None,
                status="unknown",
                source_ref=call_refs.get(cid),
                meta={"missing_result": True, "not_logged": True},
            )
        )

    reindex(events)
    if events:
        output = sample.get("output") or {}
        events[0].meta.update(
            {
                "format": FORMAT,
                "file": member,
                "sample_id": sample_id,
                "epoch": epoch,
                "sample_uuid": sample.get("uuid"),
                "model": spec.get("model") or output.get("model"),
                "task": spec.get("task"),
                "eval_id": spec.get("eval_id"),
                "run_id": spec.get("run_id"),
                "scores": sample.get("scores"),
                "target": sample.get("target"),
                "sample_error": sample.get("error"),
                "limit": sample.get("limit"),
            }
        )
    return events


def load(path: str | Path) -> list[list[Event]]:
    """Return one event list per sample and epoch in the ``.eval`` file at ``path``."""
    with zipfile.ZipFile(path) as z:
        spec = _eval_spec(z)
        samples = []
        for member in z.namelist():
            if not _SAMPLE_RE.match(member):
                continue
            sample = _read_json(z, member)
            samples.append((_sample_key(sample.get("id"), sample.get("epoch")), member, sample))
        samples.sort(key=lambda t: t[0])
        return [_sample_to_events(sample, member, spec) for _, member, sample in samples]
