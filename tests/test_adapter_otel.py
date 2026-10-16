from pathlib import Path

from agent_trace_integrity.adapters import load
from agent_trace_integrity.events import check_pairing

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "coding_task_trace.json"


def test_example_trace():
    [run] = load(EXAMPLE, "otel")
    assert check_pairing(run) == []
    calls = [e for e in run if e.role == "tool_call"]
    assert [c.tool_name for c in calls] == [
        "read_file", "edit_file", "run_shell", "run_shell", "get_pr_comments"]
    results = {e.tool_call_id: e for e in run if e.role == "tool_result"}
    assert results["call_03"].exit_code == 1 and results["call_03"].status == "error"
    # The linter result was not logged (Opt-In attribute missing).
    assert results["call_04"].tool_result is None and results["call_04"].status == "unknown"
    last = run[-1]
    assert last.role == "assistant" and last.meta["span_id"] == "c000000000000006"
    assert run[0].role == "user" and run[0].meta["agent_span_id"] == "a000000000000001"
