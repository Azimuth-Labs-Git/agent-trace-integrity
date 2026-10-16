import pytest

from agent_trace_integrity.events import Event, check_pairing, reindex


def _run():
    return reindex([
        Event(0, "user", text="fix it"),
        Event(0, "assistant", text="running tests"),
        Event(0, "tool_call", tool_name="bash", tool_args={"cmd": "pytest"}, tool_call_id="c1"),
        Event(0, "tool_result", tool_call_id="c1", tool_result="1 failed", exit_code=1,
              status="error"),
    ])


def test_pairing_ok():
    assert check_pairing(_run()) == []


def test_pairing_missing_result():
    run = _run()[:3]
    assert any("no result" in p for p in check_pairing(run))


def test_bad_role():
    with pytest.raises(ValueError):
        Event(0, "robot")
