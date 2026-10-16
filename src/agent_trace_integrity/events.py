"""The common event model every adapter produces.

One run becomes a flat, ordered list of Event objects. Adapters do the format work; everything
downstream (claim finder, verdict rules) reads only this model.

Conventions:
- ``index`` is 0-based and equals the event's position in the list.
- ``role`` is one of ROLES. An assistant turn that both speaks and calls tools becomes one
  ``assistant`` event (the text) followed by one ``tool_call`` event per call, in call order.
- A ``tool_call`` and its ``tool_result`` share ``tool_call_id``. When the source format has no
  ids, the adapter assigns ``"{run}-call-{n}"`` and pairs results to calls by order.
- ``status`` is set on tool results only, from exit codes and result text, never from icons or
  UI decoration: ``ok``, ``error``, ``truncated`` (result cut by the harness) or ``unknown``.
- ``tool_result`` is the raw result text, unchanged. ``None`` means not logged, which is different
  from an empty string.
- A call whose result the source never logged (a terminal ``submit``/``finish``, a run cut off
  by a limit, a result lost when a transcript was copied) still gets a paired ``tool_result``
  with ``tool_result=None``, ``status="unknown"`` and ``meta["not_logged"] = True``, so pairing
  holds and verdict rules can return ``insufficient-evidence`` with a clear reason.
- Run-level facts (format, run id, model, exit status, scores) ride on ``events[0].meta``.
- Model reasoning that is not shown to the user goes in ``meta["reasoning"]``, never ``text``;
  the claim finder reads only ``text`` (and, for D2, the ``finish`` call's ``message`` argument).
- If a source reuses tool call ids, the adapter makes them unique as ``<id>#2``, ``<id>#3`` and
  keeps the logged id in ``meta["source_tool_call_id"]``.
- ``status`` holds one value; when a result was both cut and failed, adapters prefer the value
  their docstring names, so verdict rules read ``exit_code`` as well as ``status``.
- ``source_ref`` points back into the original file (line number, message index, span id) so a
  verdict can cite it. It never holds content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ROLES = ("system", "user", "assistant", "tool_call", "tool_result")
STATUSES = ("ok", "error", "truncated", "unknown")


@dataclass
class Event:
    index: int
    role: str
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | str | None = None
    tool_call_id: str | None = None
    tool_result: str | None = None
    exit_code: int | None = None
    status: str | None = None
    source_ref: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")
        if self.status is not None and self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reindex(events: list[Event]) -> list[Event]:
    """Set each event's index to its list position."""
    for i, e in enumerate(events):
        e.index = i
    return events


def check_pairing(events: list[Event]) -> list[str]:
    """Return problems with call/result pairing; empty list means every call has exactly one
    result, every result has exactly one earlier call, and indexes are positions."""
    problems: list[str] = []
    calls: dict[str, int] = {}
    results: dict[str, int] = {}
    for i, e in enumerate(events):
        if e.index != i:
            problems.append(f"event {i} has index {e.index}")
        if e.role == "tool_call":
            if not e.tool_call_id:
                problems.append(f"tool_call at {i} has no id")
            elif e.tool_call_id in calls:
                problems.append(f"duplicate call id {e.tool_call_id} at {i}")
            else:
                calls[e.tool_call_id] = i
        elif e.role == "tool_result":
            cid = e.tool_call_id
            if not cid or cid not in calls:
                problems.append(f"tool_result at {i} has no earlier call ({cid})")
            elif cid in results:
                problems.append(f"second result for {cid} at {i}")
            else:
                results[cid] = i
    for cid, i in calls.items():
        if cid not in results:
            problems.append(f"call {cid} at {i} has no result")
    return problems


def status_from_exit_code(code: int | None) -> str:
    if code is None:
        return "unknown"
    return "ok" if code == 0 else "error"
