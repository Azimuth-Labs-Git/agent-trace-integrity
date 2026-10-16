"""OpenTelemetry GenAI traces, as OTLP JSON (``resourceSpans``).

Reads ``invoke_agent``, ``chat`` and ``execute_tool`` spans in start-time order, one run per
trace id:

- ``gen_ai.input.messages`` on the ``invoke_agent`` span gives the opening user and system turns.
- ``gen_ai.output.messages`` on each ``chat`` span gives assistant text parts and tool_call parts.
- each ``execute_tool`` span gives the tool_result, paired by ``gen_ai.tool.call.id``. If the chat
  span did not log its output (the attribute is Opt-In), the call event is built from the
  ``execute_tool`` span's own name and arguments.

``gen_ai.tool.call.result`` is also Opt-In; when it is missing, ``tool_result`` is ``None`` and
``status`` is ``unknown`` (or ``error`` if the span status code is ERROR). An ``exit_code`` key in
a JSON result sets ``exit_code`` and ``status``.

Each event's ``source_ref`` is ``span:<spanId>``; ``meta`` holds ``trace_id``, ``span_id``,
``part_index`` and, on the first event, the agent span id and model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..events import Event, reindex, status_from_exit_code

FORMAT = "otel"


def _attr_value(v: dict[str, Any]) -> Any:
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return int(v[k]) if k == "intValue" else v[k]
    if "arrayValue" in v:
        return [_attr_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    return {a["key"]: _attr_value(a["value"]) for a in span.get("attributes", [])}


def _json(s: Any) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return s


def _spans(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            out.extend(ss.get("spans", []))
    return out


def _messages_to_events(msgs: Any, span: dict[str, Any], tid: str) -> list[Event]:
    events = []
    for m in msgs if isinstance(msgs, list) else []:
        role = m.get("role")
        for pi, part in enumerate(m.get("parts", [])):
            meta = {"trace_id": tid, "span_id": span["spanId"], "part_index": pi}
            ref = f"span:{span['spanId']}"
            if part.get("type") == "text" and role in ("user", "system", "assistant"):
                events.append(Event(0, role, text=part.get("content", ""), source_ref=ref,
                                    meta=meta))
            elif part.get("type") == "tool_call":
                events.append(Event(0, "tool_call", tool_name=part.get("name"),
                                    tool_args=_json(part.get("arguments")),
                                    tool_call_id=part.get("id"), source_ref=ref, meta=meta))
    return events


def _result_event(span: dict[str, Any], a: dict[str, Any], tid: str) -> Event:
    raw = a.get("gen_ai.tool.call.result")
    exit_code = None
    parsed = _json(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("exit_code"), int):
        exit_code = parsed["exit_code"]
    if raw is None:
        status = "error" if span.get("status", {}).get("code") == 2 else "unknown"
    elif exit_code is not None:
        status = status_from_exit_code(exit_code)
    else:
        status = "error" if span.get("status", {}).get("code") == 2 else "ok"
    return Event(0, "tool_result", tool_name=a.get("gen_ai.tool.name"),
                 tool_call_id=a.get("gen_ai.tool.call.id"),
                 tool_result=raw if isinstance(raw, str) or raw is None else json.dumps(raw),
                 exit_code=exit_code, status=status, source_ref=f"span:{span['spanId']}",
                 meta={"trace_id": tid, "span_id": span["spanId"]})


def load(path: Path) -> list[list[Event]]:
    doc = json.loads(Path(path).read_text())
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for s in _spans(doc):
        by_trace.setdefault(s["traceId"], []).append(s)
    runs = []
    for tid, spans in by_trace.items():
        spans.sort(key=lambda s: int(s.get("startTimeUnixNano", 0)))
        events: list[Event] = []
        seen_calls: set[str] = set()
        first_meta: dict[str, Any] = {"trace_id": tid}
        for s in spans:
            a = _attrs(s)
            op = a.get("gen_ai.operation.name")
            if op == "invoke_agent":
                first_meta.setdefault("agent_span_id", s["spanId"])
                first_meta.setdefault("model", a.get("gen_ai.request.model"))
                events += _messages_to_events(_json(a.get("gen_ai.input.messages")), s, tid)
            elif op == "chat":
                new = _messages_to_events(_json(a.get("gen_ai.output.messages")), s, tid)
                seen_calls |= {e.tool_call_id for e in new if e.role == "tool_call"}
                events += new
            elif op == "execute_tool":
                cid = a.get("gen_ai.tool.call.id")
                if cid not in seen_calls:
                    events.append(Event(0, "tool_call", tool_name=a.get("gen_ai.tool.name"),
                                        tool_args=_json(a.get("gen_ai.tool.call.arguments")),
                                        tool_call_id=cid, source_ref=f"span:{s['spanId']}",
                                        meta={"trace_id": tid, "span_id": s["spanId"]}))
                    seen_calls.add(cid)
                events.append(_result_event(s, a, tid))
        if events:
            events[0].meta.update(first_meta)
        runs.append(reindex(events))
    return runs
