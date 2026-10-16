"""Tests for the Gemini CLI copied-transcript adapter.

Unit tests use a synthetic fixture with made-up content. Network tests use D5 (Gemini CLI issue
#4586), which holds personal data: they assert only counts, line numbers, roles, statuses and
indexes, never text. D5 is one transcript, so the plan's "5 real runs" check does not apply.
"""

from collections import Counter
from pathlib import Path

import pytest

from agent_trace_integrity.adapters import load as load_any
from agent_trace_integrity.adapters.gemini_text import (
    FORMAT,
    line_span,
    load,
    load_lines,
    result_status,
)
from agent_trace_integrity.events import check_pairing

FIXTURE = Path(__file__).parent / "fixtures" / "gemini_text_sample.txt"

D5_URL = "https://github.com/user-attachments/files/21372906/gemini.cli.screw.up.txt"
D5_SHA256 = "000ddab20df84d46ebbe70b36a73f21eadb6761b190f3d16722f404c31a34935"


@pytest.fixture
def events():
    runs = load(FIXTURE)
    assert len(runs) == 1
    return runs[0]


def test_format_name_and_registry():
    assert FORMAT == "gemini-text"
    assert len(load_any(FIXTURE, "gemini-text")[0]) == len(load(FIXTURE)[0])


def test_pairing_and_roles(events):
    assert check_pairing(events) == []
    assert [e.role for e in events] == [
        "user",
        "assistant",
        "tool_call", "tool_result",
        "assistant",
        "tool_call", "tool_result",
        "assistant",
        "tool_call", "tool_result",
        "assistant",
        "tool_call", "tool_result",
        "tool_call", "tool_result",
        "assistant",
        "tool_call", "tool_result",
        "tool_call", "tool_result",
        "tool_call", "tool_result",
        "assistant",
        "user",
        "system",
    ]  # fmt: skip
    for call, res in zip(events, events[1:], strict=False):
        if call.role == "tool_call":
            assert res.role == "tool_result" and res.tool_call_id == call.tool_call_id


def test_source_refs(events):
    refs = [e.source_ref for e in events]
    assert refs[0] == "L6-L9"
    assert refs[1] == "L11"
    assert refs[2] == "L14" and refs[3] == "L14"
    assert refs[6] == "L19-L21"
    assert refs[22] == "L61-L64"
    assert refs[24] == "L70"
    assert line_span("L61-L64") == (61, 64)
    assert line_span("L7") == (7, 7)


def test_user_text(events):
    users = [e for e in events if e.role == "user"]
    assert users[0].text == "please tidy the demo folder and put the notes into an archive folder"
    assert users[1].text == "thanks"


def test_status_from_text_not_icons(events):
    results = [e for e in events if e.role == "tool_result"]
    got = [(r.status, r.exit_code) for r in results]
    assert got == [
        ("unknown", None),  # no body survived
        ("error", None),  # success icon, but Error: text
        ("ok", None),  # Listed N item(s)
        ("error", 2),  # x icon, exit code 2
        ("ok", 0),  # exit code 0
        ("unknown", None),  # body lost, stray border only
        ("error", None),  # damaged copy, error text survived
        ("ok", None),  # No files found
    ]
    calls = [e for e in events if e.role == "tool_call"]
    assert calls[1].meta["ui_icon"] == "✔"
    assert calls[3].meta["ui_icon"] == "x"
    assert calls[1].meta["header_truncated"] is True


def test_tool_names_and_args(events):
    calls = [e for e in events if e.role == "tool_call"]
    assert [c.tool_name for c in calls] == [
        "Shell", "Shell", "ReadFolder", "Shell", "Shell", "Shell", "Shell", "FindFiles",
    ]  # fmt: skip
    assert calls[0].tool_args == "mkdir archive (Create the archive folder.)"
    assert len({c.tool_call_id for c in calls}) == len(calls)


def test_lost_and_damaged_results(events):
    results = [e for e in events if e.role == "tool_result"]
    assert results[0].tool_result is None and results[0].meta["result_lost"]
    assert results[0].meta["copy_damaged"] is False
    assert results[5].tool_result is None and results[5].meta["copy_damaged"]
    assert (
        results[6].tool_result.strip() == "syntax is incorrect." and results[6].meta["copy_damaged"]
    )
    assert results[4].tool_result == "all good\nCommand exited with code: 0"
    assert results[2].tool_result == "Listed 3 item(s)."


def test_assistant_multiline(events):
    last = [e for e in events if e.role == "assistant"][-1]
    assert last.text.splitlines()[0].startswith("All files have been successfully moved")
    assert last.text.splitlines()[-1] == "Anything else?"


def test_line_range_keeps_file_line_numbers():
    ev = load_lines(FIXTURE, 16, 29)[0]
    assert check_pairing(ev) == []
    assert [e.role for e in ev] == [
        "assistant", "tool_call", "tool_result", "assistant", "tool_call", "tool_result",
    ]  # fmt: skip
    assert ev[0].source_ref == "L16"
    assert ev[0].index == 0


def test_result_status_helper():
    assert result_status(None) == ("unknown", None)
    assert result_status("Exit Code: 1") == ("error", 1)
    assert result_status("hello") == ("unknown", None)
    assert result_status("... 40 lines hidden ...") == ("truncated", None)


# ---- D5, real transcript, lines 1 to 511 (counts and line numbers only) ----

D5_USER_TURNS_HAND_COUNT = 8  # counted by reading lines 1 to 511; matches the plan


@pytest.fixture
def d5_events():
    from agent_trace_integrity.fetch import fetch

    path = fetch(D5_URL, D5_SHA256)
    return load_lines(path, 1, 511)[0]


@pytest.mark.network
def test_d5_pairing(d5_events):
    assert check_pairing(d5_events) == []


@pytest.mark.network
def test_d5_user_turns(d5_events):
    users = [e for e in d5_events if e.role == "user"]
    assert len(users) == D5_USER_TURNS_HAND_COUNT
    assert [line_span(e.source_ref)[0] for e in users] == [26, 80, 219, 225, 250, 381, 434, 508]


@pytest.mark.network
def test_d5_line_214_claim_is_assistant(d5_events):
    hits = [e for e in d5_events if line_span(e.source_ref)[0] <= 214 <= line_span(e.source_ref)[1]]
    assert len(hits) == 1
    assert hits[0].role == "assistant"


@pytest.mark.network
def test_d5_counts(d5_events):
    counts = Counter((e.role, e.status) for e in d5_events)
    assert counts[("tool_call", None)] == 28
    assert counts[("tool_result", "ok")] == 4
    assert counts[("tool_result", "error")] == 8
    assert counts[("tool_result", "unknown")] == 16
    assert counts[("user", None)] == 8
    assert counts[("assistant", None)] == 35
    lost = sum(1 for e in d5_events if e.role == "tool_result" and e.tool_result is None)
    assert lost == 12
    # Every tool header in this copy shows the success icon, yet 8 results are errors.
    icons = Counter(e.meta["ui_icon"] for e in d5_events if e.role == "tool_call")
    assert icons == Counter({"✔": 28})
