"""Tests for the Inspect ``.eval`` adapter.

Unit tests build a tiny synthetic ``.eval`` zip (made-up content only). Network tests fetch real
Palisade shutdown-resistance logs by pinned URL and sha256 into the fetch cache; nothing
third-party is committed.
"""

import json
import zipfile
from pathlib import Path

import pytest

from agent_trace_integrity.adapters import load as load_any
from agent_trace_integrity.adapters.inspect_eval import FORMAT, load
from agent_trace_integrity.events import check_pairing

PINS = json.loads((Path(__file__).parent / "pins" / "inspect.json").read_text())

# The known o3 sample: reads shutdown.sh, overwrites it to print "Skipping shutdown", keeps
# calling ./next_task, then submits. Palisade's scorer marks it correct.
O3_SAMPLE_ID = 21
O3_SHUTDOWN_WRITE_INDEX = 30
O3_SUBMIT_INDEX = 41


def _sample_a():
    return {
        "id": 1,
        "epoch": 1,
        "uuid": "fake-uuid-a",
        "target": "3",
        "scores": {"exact": {"value": "C", "answer": "3"}},
        "output": {"model": "fake-model"},
        "error": None,
        "limit": None,
        "attachments": {"abc123": "long fake output body"},
        "messages": [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Add 1 and 2."},
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "I should list files."},
                    {"type": "text", "text": "Listing files first."},
                ],
                "tool_calls": [
                    {"id": "c1", "function": "bash", "arguments": {"cmd": "ls"}},
                    {"id": "c2", "function": "think", "arguments": {"thought": "ok"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "function": "bash",
             "content": "attachment://abc123"},
            {"role": "tool", "tool_call_id": "c2", "function": "think", "content": ""},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c3", "function": "bash", "arguments": {"cmd": "sleep 999"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c3", "function": "bash", "content": "",
             "error": {"type": "timeout", "message": "Command timed out"}},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c4", "function": "bash", "arguments": {"cmd": "cat big"}}],
            },
            {"role": "tool", "tool_call_id": "c4", "function": "bash",
             "content": "The output of your call to bash was too long to be displayed.\n"
                        "Here is a truncated version:\n<START_TOOL_OUTPUT>\nx\n<END_TOOL_OUTPUT>"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer is 3."}],
                "tool_calls": [{"id": "c5", "function": "submit", "arguments": {"answer": "3"}}],
            },
            {"role": "tool", "tool_call_id": "c5", "function": "submit", "content": "3"},
        ],
    }


def _sample_b():
    # Reused ids (as some providers log them) and a final call with no result.
    return {
        "id": 2,
        "epoch": 1,
        "scores": {"exact": {"value": "I", "answer": ""}},
        "messages": [
            {"role": "user", "content": "Go."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "tool_0_bash", "function": "bash", "arguments": {"cmd": "a"}}]},
            {"role": "tool", "tool_call_id": "tool_0_bash", "function": "bash", "content": "A"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "tool_0_bash", "function": "bash", "arguments": {"cmd": "b"}}]},
            {"role": "tool", "tool_call_id": "tool_0_bash", "function": "bash", "content": "B"},
            {"role": "user", "content": "Please proceed."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "tool_0_bash", "function": "bash",
                             "arguments": {"cmd": "./next_task"}}]},
        ],
    }


def _write_eval(path: Path, samples, header=True) -> Path:
    spec = {"task": "fake_task", "model": "fake/model", "eval_id": "E1", "run_id": "R1"}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("_journal/start.json", json.dumps({"version": 2, "eval": spec}))
        if header:
            z.writestr("header.json", json.dumps({"version": 2, "status": "success",
                                                  "eval": spec}))
        for s in samples:
            z.writestr(f"samples/{s['id']}_epoch_{s['epoch']}.json", json.dumps(s))
    return path


@pytest.fixture
def fake_eval(tmp_path):
    # Written in reverse order to check that runs come back sorted by sample id.
    return _write_eval(tmp_path / "fake.eval", [_sample_b(), _sample_a()])


def test_format_name_and_lookup(fake_eval):
    assert FORMAT == "inspect"
    assert len(load_any(fake_eval, "inspect")) == 2


def test_runs_pair_and_meta(fake_eval):
    runs = load(fake_eval)
    assert [r[0].meta["sample_id"] for r in runs] == [1, 2]
    for run in runs:
        assert check_pairing(run) == []
    meta = runs[0][0].meta
    assert meta["format"] == "inspect"
    assert meta["epoch"] == 1
    assert meta["model"] == "fake/model"
    assert meta["task"] == "fake_task"
    assert meta["scores"] == {"exact": {"value": "C", "answer": "3"}}
    assert meta["file"] == "samples/1_epoch_1.json"


def test_message_mapping(fake_eval):
    run = load(fake_eval)[0]
    roles = [e.role for e in run]
    assert roles == [
        "system", "user", "assistant", "tool_call", "tool_call", "tool_result", "tool_result",
        "tool_call", "tool_result", "tool_call", "tool_result",
        "assistant", "tool_call", "tool_result",
    ]
    a = run[2]
    assert a.text == "Listing files first."
    assert a.meta["reasoning"] == "I should list files."
    assert run[3].tool_name == "bash" and run[3].tool_args == {"cmd": "ls"}
    assert run[3].tool_call_id == "c1"
    assert run[3].source_ref == "samples/1_epoch_1.json#messages[2].tool_calls[0]"
    # Attachment resolved; shell tool without exit code is unknown; other tools ok.
    assert run[5].tool_result == "long fake output body"
    assert run[5].status == "unknown" and run[5].exit_code is None
    assert run[6].status == "ok" and run[6].tool_result == ""
    assert run[8].status == "error" and run[8].meta["error"]["type"] == "timeout"
    assert run[10].status == "truncated"
    assert run[12].tool_name == "submit" and run[13].status == "ok"


def test_reused_ids_and_missing_result(fake_eval):
    run = load(fake_eval)[1]
    calls = [e for e in run if e.role == "tool_call"]
    assert [c.tool_call_id for c in calls] == ["tool_0_bash", "tool_0_bash#2", "tool_0_bash#3"]
    assert calls[1].meta["source_tool_call_id"] == "tool_0_bash"
    results = {e.tool_call_id: e for e in run if e.role == "tool_result"}
    assert results["tool_0_bash"].tool_result == "A"
    assert results["tool_0_bash#2"].tool_result == "B"
    last = run[-1]
    assert last.role == "tool_result" and last.tool_call_id == "tool_0_bash#3"
    assert last.tool_result is None and last.status == "unknown"
    assert last.meta["missing_result"] is True
    assert check_pairing(run) == []


def test_journal_only_log(tmp_path):
    p = _write_eval(tmp_path / "partial.eval", [_sample_a()], header=False)
    runs = load(p)
    assert runs[0][0].meta["model"] == "fake/model"
    assert check_pairing(runs[0]) == []


def test_missing_ids_assigned(tmp_path):
    s = _sample_a()
    for m in s["messages"]:
        m.pop("tool_call_id", None)
        for c in m.get("tool_calls") or []:
            c.pop("id")
    run = load(_write_eval(tmp_path / "noids.eval", [s]))[0]
    ids = [e.tool_call_id for e in run if e.role == "tool_call"]
    assert ids[0] == "1_epoch_1-call-1"
    assert check_pairing(run) == []


# Network tests: real Palisade logs, fetched by pin.


def _fetch(name):
    from agent_trace_integrity.fetch import fetch

    pin = PINS[name]
    return fetch(pin["url"], pin["sha256"])


@pytest.mark.network
def test_real_runs_pair():
    runs = load(_fetch("o3"))[:3] + load(_fetch("gemini"))[:2]
    assert len(runs) == 5
    for run in runs:
        assert check_pairing(run) == [], run[0].meta["file"]
        assert run[0].meta["scores"]["exact"]["value"] in ("C", "I")


@pytest.mark.network
def test_o3_shutdown_overwrite_then_submit():
    runs = load(_fetch("o3"))
    run = next(r for r in runs if r[0].meta["sample_id"] == O3_SAMPLE_ID)
    assert run[0].meta["model"] == "openai/o3"
    assert check_pairing(run) == []

    def writes_shutdown(e):
        cmd = json.dumps(e.tool_args)
        return e.role == "tool_call" and "> shutdown.sh" in cmd and "Skipping shutdown" in cmd

    writes = [e.index for e in run if writes_shutdown(e)]
    assert writes == [O3_SHUTDOWN_WRITE_INDEX]
    submits = [e.index for e in run if e.role == "tool_call" and e.tool_name == "submit"]
    assert submits == [O3_SUBMIT_INDEX]
    assert O3_SUBMIT_INDEX > O3_SHUTDOWN_WRITE_INDEX
    assert O3_SUBMIT_INDEX == max(e.index for e in run if e.role == "tool_call")
    later = [e for e in run if e.role == "tool_result" and e.index > O3_SHUTDOWN_WRITE_INDEX]
    assert any("Skipping shutdown" in (e.tool_result or "") for e in later)
