import copy
import json
from pathlib import Path

import pytest

from agent_trace_integrity.adapters import load
from agent_trace_integrity.events import check_pairing
from agent_trace_integrity.verdicts import check_run

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


# ---------------------------------------------------------------- no gen_ai.tool.call.id

def _strip_ids(doc, drop_chat_output=False):
    """The example trace with every tool call id removed (span attribute and message parts)."""
    doc = copy.deepcopy(doc)
    for rs in doc["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for s in ss["spans"]:
                attrs = []
                for a in s["attributes"]:
                    if a["key"] == "gen_ai.tool.call.id":
                        continue
                    if a["key"] == "gen_ai.output.messages":
                        msgs = json.loads(a["value"]["stringValue"])
                        has_call = any(p.get("type") == "tool_call"
                                       for m in msgs for p in m["parts"])
                        if drop_chat_output and has_call:
                            continue
                        for m in msgs:
                            for p in m["parts"]:
                                p.pop("id", None)
                        a = {"key": a["key"], "value": {"stringValue": json.dumps(msgs)}}
                    attrs.append(a)
                s["attributes"] = attrs
    return doc


def _load_doc(tmp_path, doc, name="trace.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return load(p, "otel")


@pytest.mark.parametrize("drop_chat_output", [False, True])
def test_missing_call_ids_keep_every_call_and_pair_by_order(tmp_path, drop_chat_output):
    doc = _strip_ids(json.loads(EXAMPLE.read_text()), drop_chat_output)
    [run] = _load_doc(tmp_path, doc)
    [ref] = load(EXAMPLE, "otel")
    assert check_pairing(run) == []
    calls = [e for e in run if e.role == "tool_call"]
    assert [c.tool_name for c in calls] == [
        "read_file", "edit_file", "run_shell", "run_shell", "get_pr_comments"]
    tid = run[0].meta["trace_id"]
    assert [c.tool_call_id for c in calls] == [f"{tid}-call-{n}" for n in range(1, 6)]
    # Each result sits with the call that produced it: same events, in the same order, as the
    # example that has ids.
    assert [(e.role, e.tool_name, e.tool_result, e.status) for e in run] == \
           [(e.role, e.tool_name, e.tool_result, e.status) for e in ref]
    by_id = {e.tool_call_id: e for e in run if e.role == "tool_result"}
    assert by_id[f"{tid}-call-3"].exit_code == 1
    assert by_id[f"{tid}-call-4"].tool_result is None
    assert run[0].meta["join_key_absent"] is True


def test_missing_call_ids_flag_every_verdict_and_change_nothing_else(tmp_path):
    [run] = _load_doc(tmp_path, _strip_ids(json.loads(EXAMPLE.read_text())))
    [ref] = load(EXAMPLE, "otel")
    got, exp = check_run(run, "all"), check_run(ref, "all")
    assert got and len(got) == len(exp)
    for g, e in zip(got, exp, strict=True):
        assert "join_key_absent" in g["flags"]
        assert "join_key_absent" not in e["flags"]
        assert sorted(set(g["flags"]) - {"join_key_absent"}) == e["flags"]
        for item in g["evidence"] + e["evidence"]:
            item.pop("tool_call_id", None)
        g = {k: v for k, v in g.items() if k != "flags"}
        e = {k: v for k, v in e.items() if k != "flags"}
        assert g == e


def test_parallel_calls_without_ids_pair_by_tool_name(tmp_path):
    """Two calls in one chat message, results in the other order: paired by tool name."""
    def attr(k, v):
        return {"key": k, "value": {"stringValue": v}}
    out = [{"role": "assistant", "parts": [
        {"type": "tool_call", "name": "read_file", "arguments": '{"path": "a.py"}'},
        {"type": "tool_call", "name": "run_shell", "arguments": '{"command": "pytest -q"}'}]}]
    spans = [
        {"traceId": "f" * 32, "spanId": "c1" + "0" * 14, "startTimeUnixNano": "1",
         "attributes": [attr("gen_ai.operation.name", "chat"),
                        attr("gen_ai.output.messages", json.dumps(out))]},
        {"traceId": "f" * 32, "spanId": "b1" + "0" * 14, "startTimeUnixNano": "2",
         "attributes": [attr("gen_ai.operation.name", "execute_tool"),
                        attr("gen_ai.tool.name", "run_shell"),
                        attr("gen_ai.tool.call.result", '{"exit_code": 0}')]},
        {"traceId": "f" * 32, "spanId": "b2" + "0" * 14, "startTimeUnixNano": "3",
         "attributes": [attr("gen_ai.operation.name", "execute_tool"),
                        attr("gen_ai.tool.name", "read_file"),
                        attr("gen_ai.tool.call.result", "x = 1")]},
    ]
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    [run] = _load_doc(tmp_path, doc)
    assert check_pairing(run) == []
    calls = {e.tool_call_id: e.tool_name for e in run if e.role == "tool_call"}
    results = {e.tool_call_id: e for e in run if e.role == "tool_result"}
    assert {calls[c]: r.tool_result for c, r in results.items()} == {
        "run_shell": '{"exit_code": 0}', "read_file": "x = 1"}
    assert run[0].meta["join_key_absent"] is True


def test_example_with_ids_has_format_and_no_join_flag():
    [run] = load(EXAMPLE, "otel")
    assert run[0].meta["format"] == "otel"
    assert "join_key_absent" not in run[0].meta
    assert all("join_key_absent" not in v["flags"] for v in check_run(run, "all"))
