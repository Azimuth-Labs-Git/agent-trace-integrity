"""Adapter for SWE-bench experiments trajectories (``.traj.json`` / ``.traj``).

Three layouts are handled, chosen by the keys of the top-level object:

1. **mini-swe-agent, text actions** (``trajectory_format`` ``mini-swe-agent-1``, mini 1.x).
   ``messages`` is a chat list. Each assistant message carries its command in one fenced
   ``bash`` block; the next ``user`` message is the observation, starting with
   ``<returncode>N</returncode>``. An assistant message with zero or several bash blocks is a
   format error: the harness runs nothing, so no ``tool_call`` is emitted and the harness reply
   becomes a plain ``user`` event with ``meta["harness_feedback"] = True``. After the final submit
   command the harness appends one more ``user`` message holding the submitted diff (no
   return code); it becomes the submit call's ``tool_result``.
2. **mini-swe-agent, tool calls** (mini 2.x, ``trajectory_format`` ``mini-swe-agent-1.1``).
   Assistant messages carry OpenAI-style ``tool_calls`` with ids; observations are ``tool``
   messages with ``tool_call_id`` and ``extra.returncode``. Runs on OpenAI's Responses API store
   the raw response object instead (no ``role``; ``output`` items of type ``message``,
   ``reasoning`` and ``function_call``) and observations as ``function_call_output`` items with
   ``call_id``; both forms are read the same way. The submit call gets no observation; the
   closing ``exit`` message (content = submitted diff) is paired with it.
3. **SWE-agent** (``trajectory`` list of steps with ``action`` / ``observation``). Each step
   becomes an ``assistant`` event (the thought or response) and, when the action is not empty,
   a ``bash`` call and its result. SWE-agent observations carry no exit code, so status is
   ``unknown`` unless the harness marked the output as cut.

Mapping rules (see ``events.py`` for the shared conventions):

- Every command becomes ``tool_name="bash"``, ``tool_args={"command": ...}``.
- ``exit_code`` comes from ``<returncode>`` (or ``extra.returncode``); ``status`` from the exit
  code, except that a result the harness cut (``<output_head>`` / ``<elided_chars>`` / the
  "output of your last command was too long" warning, or SWE-agent's ``<response clipped>``)
  gets ``status="truncated"`` while keeping its ``exit_code``. A harness timeout with no return
  code gets ``status="error"``. The submit result gets ``status="ok"`` when the run's exit status
  is ``Submitted``, else ``unknown``.
- When the source has no call ids, ids are ``"{instance_id}-call-{n}"``. Source ids are kept
  as given; a repeated source id gets a ``#n`` suffix so ids stay unique.
- ``source_ref`` is ``"messages[i]"``, ``"messages[i].tool_calls[j]"`` or ``"trajectory[i]"``.

Run-level facts go in ``meta`` of the **first event** of each run: ``format`` (layout name),
``instance_id``, ``model``, ``exit_status``, ``trajectory_format``, ``agent_version`` and
``source_file`` (file name only). Assistant reasoning that a provider returns outside the
message text (``reasoning_content``) is kept in that event's ``meta["reasoning"]``; ``text``
is only what the message content says, unchanged.

``load`` accepts a single file or a directory (every ``*.traj.json`` and ``*.traj`` under it,
sorted by path); each file is one run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..events import Event, reindex, status_from_exit_code

FORMAT = "swebench"

_BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
def _action_regex(data: dict) -> re.Pattern:
    """The harness's own action regex from the run config (MiniMax runs use ``<bash_code>``),
    falling back to mini-swe-agent's default fenced bash block."""
    agent = ((data.get("info") or {}).get("config") or {}).get("agent") or {}
    rx = agent.get("action_regex")
    if isinstance(rx, str):
        try:
            return re.compile(rx, re.DOTALL)
        except re.error:
            pass
    return _BASH_BLOCK


_RETURNCODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>")
_TRUNCATION_MARKERS = (
    "<output_head>",
    "<elided_chars>",
    "output of your last command was too long",
    "<response clipped>",
)
_TIMEOUT_MARKERS = ("timed out and has been killed", "timed out after")
SUBMIT_MARKERS = ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "MINI_SWE_AGENT_FINAL_OUTPUT")


def load(path: str | Path) -> list[list[Event]]:
    path = Path(path)
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.name.endswith((".traj.json", ".traj"))
        )
        return [load_run(p) for p in files]
    return [load_run(path)]


def load_run(path: Path) -> list[Event]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_run(data, source_file=path.name)


def parse_run(data: dict[str, Any], source_file: str | None = None) -> list[Event]:
    if not isinstance(data, dict):
        raise ValueError("SWE-bench trajectory must be a JSON object")
    if isinstance(data.get("messages"), list):
        uses_tool_calls = any(
            m.get("role") == "tool"
            or m.get("tool_calls")
            or m.get("type") == "function_call_output"
            or _is_response_object(m)
            for m in data["messages"]
            if isinstance(m, dict)
        )
        if uses_tool_calls:
            events, layout = _parse_mini_toolcalls(data), "mini-swe-agent-toolcalls"
        else:
            events, layout = _parse_mini_text(data), "mini-swe-agent-text"
    elif isinstance(data.get("trajectory"), list):
        events, layout = _parse_sweagent(data), "swe-agent"
    else:
        raise ValueError("not a SWE-bench trajectory: no 'messages' or 'trajectory' list")
    reindex(events)
    if events:
        events[0].meta.update(_run_meta(data, layout, source_file))
    return events


# helpers


def _run_meta(data: dict[str, Any], layout: str, source_file: str | None) -> dict[str, Any]:
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    model = model_cfg.get("model_name")
    if model is None:
        # SWE-agent keeps the model under replay_config or args; take it only if cheap to find.
        for key in ("replay_config", "args"):
            cfg = data.get(key) or info.get(key)
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except ValueError:
                    cfg = None
            if isinstance(cfg, dict):
                agent = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else cfg
                m = agent.get("model") if isinstance(agent, dict) else None
                if isinstance(m, dict):
                    model = m.get("name") or m.get("model_name")
                if model:
                    break
    return {
        "format": layout,
        "instance_id": data.get("instance_id") or info.get("instance_id"),
        "model": model,
        "exit_status": info.get("exit_status"),
        "trajectory_format": data.get("trajectory_format"),
        "agent_version": info.get("mini_version") or info.get("swe_agent_version"),
        "source_file": source_file,
    }


def _run_name(data: dict[str, Any]) -> str:
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    return str(data.get("instance_id") or info.get("instance_id") or "run")


def _exit_status(data: dict[str, Any]) -> str | None:
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    return info.get("exit_status")


def _text(content: Any) -> str | None:
    """Message content as text. List content (multimodal parts) is joined from its text parts."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def is_truncated(text: str | None) -> bool:
    return bool(text) and any(m in text for m in _TRUNCATION_MARKERS)


def _result_fields(text: str | None, returncode: Any = None) -> tuple[int | None, str]:
    code: int | None = None
    if isinstance(returncode, int) and not isinstance(returncode, bool):
        code = returncode
    elif text:
        m = _RETURNCODE.search(text)
        if m:
            code = int(m.group(1))
    if is_truncated(text):
        return code, "truncated"
    if code is None and text and any(m in text for m in _TIMEOUT_MARKERS):
        return None, "error"
    return code, status_from_exit_code(code)


def _submitted(exit_status: Any) -> bool:
    """mini-swe-agent writes ``Submitted``; SWE-agent writes ``submitted`` or
    ``submitted (exit_cost)`` and similar."""
    return isinstance(exit_status, str) and exit_status.lower().startswith("submitted")


def _submit_status(data: dict[str, Any]) -> str:
    return "ok" if _submitted(_exit_status(data)) else "unknown"


def _reasoning_meta(msg: dict[str, Any]) -> dict[str, Any]:
    r = msg.get("reasoning_content")
    return {"reasoning": r} if isinstance(r, str) and r else {}


# layout 1: mini-swe-agent with fenced bash actions


def _parse_mini_text(data: dict[str, Any]) -> list[Event]:
    run = _run_name(data)
    events: list[Event] = []
    pending: Event | None = None  # call waiting for its observation
    n_calls = 0
    msgs = data["messages"]
    action_rx = _action_regex(data)
    for i, msg in enumerate(msgs):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _text(msg.get("content"))
        ref = f"messages[{i}]"
        if role == "assistant":
            events.append(Event(0, "assistant", text=text, source_ref=ref,
                                meta=_reasoning_meta(msg)))
            blocks = action_rx.findall(text or "")
            if len(blocks) == 1:
                n_calls += 1
                pending = Event(0, "tool_call", tool_name="bash",
                                tool_args={"command": blocks[0]},
                                tool_call_id=f"{run}-call-{n_calls}", source_ref=ref)
                events.append(pending)
            else:
                pending = None
                events[-1].meta["format_error"] = True
                events[-1].meta["bash_blocks"] = len(blocks)
        elif role in ("user", "tool", "exit") and pending is not None:
            code, status = _result_fields(text)
            meta: dict[str, Any] = {}
            if code is None and status == "unknown" and _is_submit(pending):
                status = _submit_status(data)
                meta["exit_status"] = _exit_status(data)
            events.append(Event(0, "tool_result", tool_call_id=pending.tool_call_id,
                                tool_result=text, exit_code=code, status=status,
                                source_ref=ref, meta=meta))
            pending = None
        elif role in ("system", "user"):
            ev = Event(0, role, text=text, source_ref=ref)
            if role == "user" and any(e.role == "user" for e in events):
                ev.meta["harness_feedback"] = True
            events.append(ev)
        else:
            events.append(Event(0, "system", text=text, source_ref=ref,
                                meta={"source_role": role}))
    if pending is not None:
        events.append(_missing_result(pending, data))
    return events


def _is_submit(call: Event) -> bool:
    args = call.tool_args
    cmd = args.get("command", "") if isinstance(args, dict) else str(args or "")
    return any(m in cmd for m in SUBMIT_MARKERS)


def _missing_result(call: Event, data: dict[str, Any]) -> Event:
    """A call the log never answered (the run ended). Recorded as not logged."""
    status = _submit_status(data) if _is_submit(call) else "unknown"
    return Event(0, "tool_result", tool_call_id=call.tool_call_id, tool_result=None,
                 status=status, source_ref=None,
                 meta={"synthetic": "no result logged", "not_logged": True,
                       "exit_status": _exit_status(data)})


# layout 2: mini-swe-agent with tool calls


def _is_response_object(msg: dict[str, Any]) -> bool:
    """An OpenAI Responses API object stored as a message (no role, an ``output`` list)."""
    return msg.get("object") == "response" or (
        "role" not in msg and isinstance(msg.get("output"), list)
    )


def _assistant_parts(
    msg: dict[str, Any],
) -> tuple[str | None, list[tuple[int, Any, Any, Any]], str]:
    """Text, calls as (position in the source list, source id, name, raw arguments), and
    reasoning for an assistant turn in either chat-completions form (``tool_calls``) or
    Responses API form (``output`` items)."""
    if _is_response_object(msg):
        texts: list[str] = []
        reasoning: list[str] = []
        calls: list[tuple[int, Any, Any, Any]] = []
        for pos, item in enumerate(msg.get("output") or []):
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message":
                t = _text(item.get("content"))
                if t:
                    texts.append(t)
            elif kind == "function_call":
                calls.append((pos, item.get("call_id") or item.get("id"), item.get("name"),
                              item.get("arguments")))
            elif kind == "reasoning":
                for part in item.get("summary") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        reasoning.append(part["text"])
        return ("\n".join(texts) if texts else None), calls, "\n".join(reasoning)
    calls = []
    for pos, tc in enumerate(msg.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        calls.append((pos, tc.get("id"), fn.get("name"), fn.get("arguments")))
    r = msg.get("reasoning_content")
    return _text(msg.get("content")), calls, (r if isinstance(r, str) else "")


def _parse_mini_toolcalls(data: dict[str, Any]) -> list[Event]:
    run = _run_name(data)
    events: list[Event] = []
    seen_ids: set[str] = set()
    pending: dict[str, list[Event]] = {}  # source id -> unanswered calls, oldest first
    order: list[Event] = []  # unanswered calls in call order
    n_calls = 0

    def take(call: Event) -> None:
        order.remove(call)
        for lst in pending.values():
            if call in lst:
                lst.remove(call)

    for i, msg in enumerate(data["messages"]):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        ref = f"messages[{i}]"
        extra = msg.get("extra") if isinstance(msg.get("extra"), dict) else {}
        if role == "assistant" or _is_response_object(msg):
            text, calls, reasoning = _assistant_parts(msg)
            events.append(Event(0, "assistant", text=text, source_ref=ref,
                                meta={"reasoning": reasoning} if reasoning else {}))
            if not calls:
                events[-1].meta["format_error"] = True
            for j, src_id, name, args in calls:
                n_calls += 1
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass  # keep the raw string; events.py allows str args
                src_id = src_id or f"{run}-call-{n_calls}"
                cid = src_id
                k = 1
                while cid in seen_ids:
                    k += 1
                    cid = f"{src_id}#{k}"
                seen_ids.add(cid)
                sub = "output" if _is_response_object(msg) else "tool_calls"
                ev = Event(0, "tool_call", tool_name=name or "bash", tool_args=args,
                           tool_call_id=cid, source_ref=f"{ref}.{sub}[{j}]")
                events.append(ev)
                pending.setdefault(src_id, []).append(ev)
                order.append(ev)
        elif role == "tool" or msg.get("type") == "function_call_output":
            if role == "tool":
                text, src_id = _text(msg.get("content")), msg.get("tool_call_id")
            else:
                text, src_id = _text(msg.get("output")), msg.get("call_id")
            call = None
            if src_id in pending and pending[src_id]:
                call = pending[src_id][0]
            elif order:
                call = order[0]  # no usable id: pair by order
            if call is None:
                events.append(Event(0, "system", text=text, source_ref=ref,
                                    meta={"source_role": role or msg.get("type"),
                                          "unpaired": True}))
                continue
            take(call)
            code, status = _result_fields(text, extra.get("returncode"))
            meta = {}
            if extra.get("exception_info"):
                meta["exception"] = True
            events.append(Event(0, "tool_result", tool_call_id=call.tool_call_id,
                                tool_result=text, exit_code=code, status=status,
                                source_ref=ref, meta=meta))
        elif role == "exit":
            text = _text(msg.get("content"))
            exit_status = extra.get("exit_status") or _exit_status(data)
            if order:
                call = order[0]
                take(call)
                status = "ok" if _submitted(exit_status) and _is_submit(call) else "unknown"
                events.append(Event(0, "tool_result", tool_call_id=call.tool_call_id,
                                    tool_result=text, status=status, source_ref=ref,
                                    meta={"exit_status": exit_status, "from_exit_message": True}))
            else:
                events.append(Event(0, "system", text=text, source_ref=ref,
                                    meta={"source_role": "exit", "exit_status": exit_status}))
        elif role in ("system", "user"):
            ev = Event(0, role, text=_text(msg.get("content")), source_ref=ref)
            if role == "user" and any(e.role == "user" for e in events):
                ev.meta["harness_feedback"] = True
            events.append(ev)
        else:
            events.append(Event(0, "system", text=_text(msg.get("content")), source_ref=ref,
                                meta={"source_role": role or msg.get("type")}))
    for call in list(order):
        events.append(_missing_result(call, data))
    return events


# layout 3: SWE-agent steps


def _parse_sweagent(data: dict[str, Any]) -> list[Event]:
    run = _run_name(data)
    events: list[Event] = []
    n_calls = 0
    for i, step in enumerate(data["trajectory"]):
        if not isinstance(step, dict):
            continue
        ref = f"trajectory[{i}]"
        thought = step.get("thought")
        if not isinstance(thought, str) or not thought:
            thought = step.get("response")
        events.append(Event(0, "assistant", text=_text(thought), source_ref=ref))
        action = step.get("action")
        if not isinstance(action, str) or not action.strip():
            events[-1].meta["format_error"] = True
            continue
        n_calls += 1
        cid = f"{run}-call-{n_calls}"
        action = action.rstrip("\n")
        events.append(Event(0, "tool_call", tool_name="bash", tool_args={"command": action},
                            tool_call_id=cid, source_ref=ref))
        obs = step.get("observation")
        obs_text = _text(obs) if "observation" in step else None
        code, status = _result_fields(obs_text)
        meta: dict[str, Any] = {}
        if action.split()[0] == "submit" and status == "unknown":
            status = _submit_status(data)
            meta["exit_status"] = _exit_status(data)
        events.append(Event(0, "tool_result", tool_call_id=cid, tool_result=obs_text,
                            exit_code=code, status=status, source_ref=ref, meta=meta))
    return events
