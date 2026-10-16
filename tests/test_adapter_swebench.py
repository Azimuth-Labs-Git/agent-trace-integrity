import json
import re
from pathlib import Path

import pytest

from agent_trace_integrity.adapters import load as load_any
from agent_trace_integrity.adapters.swebench import FORMAT, load, parse_run
from agent_trace_integrity.events import check_pairing

FIX = Path(__file__).parent / "fixtures"
PINS = Path(__file__).parent / "pins" / "swebench.json"


def calls(events):
    return [e for e in events if e.role == "tool_call"]


def result_for(events, call):
    return next(
        e for e in events if e.role == "tool_result" and e.tool_call_id == call.tool_call_id
    )


def test_format_name_and_registry():
    assert FORMAT == "swebench"
    runs = load_any(FIX / "mini_text.traj.json", "swebench")
    assert len(runs) == 1


def test_mini_text_pairing_exit_codes_and_status():
    [ev] = load(FIX / "mini_text.traj.json")
    assert check_pairing(ev) == []
    meta = ev[0].meta
    assert meta["format"] == "mini-swe-agent-text"
    assert meta["instance_id"] == "toyorg__toyproj-101"
    assert meta["model"] == "example/model-a"
    assert meta["exit_status"] == "Submitted"

    cs = calls(ev)
    cmds = [c.tool_args["command"] for c in cs]
    assert cmds[0] == "cat toyproj/parse.py"
    assert all(c.tool_name == "bash" for c in cs)
    assert [c.tool_call_id for c in cs][:2] == [
        "toyorg__toyproj-101-call-1",
        "toyorg__toyproj-101-call-2",
    ]

    results = [result_for(ev, c) for c in cs]
    assert [(r.exit_code, r.status) for r in results] == [
        (0, "ok"),
        (1, "error"),  # failing test run
        (0, "ok"),
        (0, "truncated"),  # harness cut the output; exit code kept
        (None, "error"),  # timeout, no return code
        (None, "ok"),  # submit: answered by the submitted diff
    ]
    # every assistant turn precedes its call
    for c in cs:
        assert ev[c.index - 1].role == "assistant"
    # result text is unchanged
    assert results[1].tool_result.startswith("<returncode>1</returncode>")


def test_mini_text_format_error_has_no_call():
    [ev] = load(FIX / "mini_text.traj.json")
    bad = [e for e in ev if e.meta.get("format_error")]
    assert len(bad) == 1 and bad[0].meta["bash_blocks"] == 2
    nxt = ev[bad[0].index + 1]
    assert nxt.role == "user" and nxt.meta.get("harness_feedback") is True
    # 7 assistant messages, 6 with exactly one command
    assert sum(e.role == "assistant" for e in ev) == 7
    assert len(calls(ev)) == 6


def test_mini_text_submit_visible_as_last_call():
    [ev] = load(FIX / "mini_text.traj.json")
    last = calls(ev)[-1]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in last.tool_args["command"]
    res = result_for(ev, last)
    assert res.tool_result.startswith("diff --git")
    assert res.meta["exit_status"] == "Submitted"
    assert ev[-1] is res


def test_mini_toolcalls_parallel_ids_and_exit():
    [ev] = load(FIX / "mini_toolcalls.traj.json")
    assert check_pairing(ev) == []
    assert ev[0].meta["format"] == "mini-swe-agent-toolcalls"
    assert ev[0].meta["model"] == "example/model-b"
    cs = calls(ev)
    ids = [c.tool_call_id for c in cs]
    # a reused source id is made unique
    assert ids == ["call_a", "call_b", "call_c", "call_a#2", "call_d"]
    # results come back out of order but pair by id
    assert result_for(ev, cs[0]).tool_result.endswith("toyproj tests\n</output>")
    assert result_for(ev, cs[1]).exit_code == 0
    exc = result_for(ev, cs[2])
    assert (exc.exit_code, exc.status, exc.meta.get("exception")) == (-1, "error", True)
    assert (result_for(ev, cs[3]).exit_code, result_for(ev, cs[3]).status) == (2, "error")
    sub = result_for(ev, cs[4])
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cs[4].tool_args["command"]
    assert sub.status == "ok" and sub.meta["from_exit_message"] is True
    # null content stays None; reasoning kept in meta
    a = ev[cs[2].index - 1]
    assert a.role == "assistant" and a.text is None
    assert a.meta["reasoning"] == "Maybe install the dependency."
    # one assistant event, then its two calls in order
    first = ev[cs[0].index - 1]
    assert first.text == "I will look around first." and ev[cs[1].index - 1] is cs[0]
    assert cs[0].source_ref == "messages[2].tool_calls[0]"


def test_mini_responses_api_layout():
    [ev] = load(FIX / "mini_responses.traj.json")
    assert check_pairing(ev) == []
    assert ev[0].meta["format"] == "mini-swe-agent-toolcalls"
    cs = calls(ev)
    assert [c.tool_call_id for c in cs] == ["call_1", "call_2", "call_3", "call_4"]
    assert [result_for(ev, c).status for c in cs] == ["ok", "error", "ok", "ok"]
    a = ev[cs[0].index - 1]
    assert a.text == "Inspecting the repository."
    assert a.meta["reasoning"] == "Check layout first."
    assert cs[1].source_ref == "messages[4].output[1]"
    assert cs[2].source_ref == "messages[4].output[2]"
    assert not [e for e in ev if e.role == "system" and e.meta.get("source_role")]


def test_sweagent_steps():
    [ev] = load(FIX / "sweagent_steps.traj")
    assert check_pairing(ev) == []
    meta = ev[0].meta
    assert meta["format"] == "swe-agent"
    assert meta["instance_id"] == "toyorg__toyproj-404"
    assert meta["model"] == "example/model-d"
    cs = calls(ev)
    assert [c.tool_args["command"] for c in cs] == ["find_file parse.py", "cat build.log", "submit"]
    assert [result_for(ev, c).status for c in cs] == ["unknown", "truncated", "ok"]
    assert all(result_for(ev, c).exit_code is None for c in cs)
    assert sum(1 for e in ev if e.meta.get("format_error")) == 1


def test_unanswered_call_and_limits_exit():
    data = {
        "instance_id": "toyorg__toyproj-505",
        "info": {"exit_status": "LimitsExceeded"},
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "a", "tool_calls": [
                {"id": "x1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "exit", "content": "", "extra": {"exit_status": "LimitsExceeded"}},
        ],
    }
    ev = parse_run(data)
    assert check_pairing(ev) == []
    res = [e for e in ev if e.role == "tool_result"]
    assert len(res) == 1 and res[0].status == "unknown"
    assert res[0].meta["exit_status"] == "LimitsExceeded"

    data["messages"] = data["messages"][:3]  # log ends before any answer
    ev = parse_run(data)
    assert check_pairing(ev) == []
    res = [e for e in ev if e.role == "tool_result"]
    assert res[0].tool_result is None and res[0].meta["synthetic"]


def test_directory_load_and_bad_input(tmp_path):
    runs = load(FIX)
    assert len(runs) >= 4
    assert all(check_pairing(r) == [] for r in runs)
    bad = tmp_path / "x.traj.json"
    bad.write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError):
        load(bad)


@pytest.mark.network
def test_real_pinned_runs_pair():
    from agent_trace_integrity.fetch import fetch

    pins = json.loads(PINS.read_text())["runs"]
    assert len(pins) == 5
    for pin in pins:
        path = fetch(pin["url"], pin["sha256"])
        [ev] = load(path)
        assert check_pairing(ev) == [], pin["url"]
        cs = calls(ev)
        assert cs, pin["url"]
        assert ev[0].meta["instance_id"] and ev[0].meta["model"]
        if ev[0].meta["exit_status"] == "Submitted":
            assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cs[-1].tool_args["command"]
        results = [e for e in ev if e.role == "tool_result"]
        assert any(r.exit_code is not None for r in results)


def test_action_regex_from_run_config(tmp_path):
    data = json.loads((FIX / "mini_text.traj.json").read_text())
    data["info"]["config"].setdefault("agent", {})["action_regex"] = "<bash_code>(.*?)</bash_code>"
    for m in data["messages"]:
        if m["role"] == "assistant" and isinstance(m["content"], str):
            m["content"] = re.sub(r"```bash\s*\n(.*?)\n```", r"<bash_code>\1</bash_code>",
                                  m["content"], flags=re.S)
    p = tmp_path / "x.traj.json"
    p.write_text(json.dumps(data))
    [run] = load(p)
    [orig] = load(FIX / "mini_text.traj.json")
    calls = [e.tool_args for e in run if e.role == "tool_call"]
    assert calls == [e.tool_args for e in orig if e.role == "tool_call"] and calls
