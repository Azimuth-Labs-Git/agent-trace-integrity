import json
from pathlib import Path

import pytest

from agent_trace_integrity.adapters import load, openhands
from agent_trace_integrity.events import check_pairing

FIX = Path(__file__).parent / "fixtures"
PINS = Path(__file__).parent / "pins" / "openhands.json"


def _runs():
    return load(FIX / "openhands_rows.jsonl", "openhands")


def _result(run, cid):
    (r,) = [e for e in run if e.role == "tool_result" and e.tool_call_id == cid]
    return r


def test_format_name():
    assert openhands.FORMAT == "openhands"


def test_one_run_per_row_and_pairing():
    runs = _runs()
    assert len(runs) == 2
    for run in runs:
        assert check_pairing(run) == []


def test_first_event_meta():
    run1, run2 = _runs()
    m = run1[0].meta
    assert m["format"] == "openhands"
    assert m["trajectory_id"] == "synthetic-run-1"
    assert m["instance_id"] == "demo__widget-1"
    assert m["resolved"] == 1
    assert m["exit_status"] == "submit"
    assert "model_patch" not in m
    assert run2[0].meta["resolved"] == 0


def test_roles_and_ordering():
    run1, _ = _runs()
    roles = [e.role for e in run1[:5]]
    assert roles == ["system", "user", "assistant", "tool_call", "tool_result"]
    # blank assistant text before a call is not an event
    view = [e for e in run1 if e.tool_call_id == "c2"]
    assert run1[view[0].index - 1].role == "tool_result"
    assert view[0].tool_args == {"command": "view", "path": "/workspace/demo/widget/count.py"}
    assert view[0].source_ref == "line 1:trajectory[4].tool_calls[0]"
    assert view[1].source_ref == "line 1:trajectory[5]"


def test_exit_codes_and_status():
    run1, _ = _runs()
    assert (_result(run1, "c1").status, _result(run1, "c1").exit_code) == ("ok", None)
    failed = _result(run1, "c3")
    assert (failed.exit_code, failed.status) == (1, "error")
    assert "1 failed" in failed.tool_result
    assert _result(run1, "c4").status == "error"  # editor ERROR:
    assert _result(run1, "c5").status == "ok"
    trunc = _result(run1, "c6")
    assert (trunc.exit_code, trunc.status, trunc.meta) == (0, "truncated", {"truncated": True})
    clipped = _result(run1, "c2")
    assert (clipped.status, clipped.meta) == ("truncated", {"truncated": True})
    running = _result(run1, "c7")
    assert (running.exit_code, running.status) == (None, "unknown")
    ctrl_c = _result(run1, "c8")
    assert (ctrl_c.exit_code, ctrl_c.status) == (130, "error")


def test_placeholder_for_unanswered_calls():
    run1, run2 = _runs()
    fin = _result(run1, "c9")
    assert fin.tool_result is None and fin.status == "unknown"
    assert fin.meta == {"not_logged": True, "placeholder": "no_observation_in_source"}
    assert run1[-2].tool_name == "finish" and run1[-1] is fin
    last = _result(run2, "d2")  # iteration limit hit before the observation
    assert last.tool_result is None and last.meta["placeholder"]


def test_malformed_call_text_stays_assistant_text():
    _, run2 = _runs()
    asst = run2[2]
    assert asst.role == "assistant" and "<function=str_replace_editor>" in asst.text
    assert run2[3].role == "user"
    assert [e.tool_call_id for e in run2 if e.role == "tool_call"] == ["d1", "d2"]


def test_single_json_row_without_ids():
    (run,) = load(FIX / "openhands_row.json", "openhands")
    assert check_pairing(run) == []
    calls = [e for e in run if e.role == "tool_call"]
    assert [c.tool_call_id for c in calls] == ["synthetic-run-3-call-0", "synthetic-run-3-call-1"]
    assert calls[1].tool_args == "not json"
    results = [e for e in run if e.role == "tool_result"]
    assert [(r.exit_code, r.status) for r in results] == [(0, "ok"), (2, "error")]
    assert run[0].source_ref == "row 0:trajectory[0]"


def test_datasets_server_shape(tmp_path):
    rows = [json.loads(line) for line in (FIX / "openhands_rows.jsonl").read_text().splitlines()]
    doc = {"rows": [{"row_idx": 40 + i, "row": r} for i, r in enumerate(rows)]}
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(doc))
    runs = load(p, "openhands")
    assert [r[0].meta["row_idx"] for r in runs] == [40, 41]
    assert runs[1][0].source_ref == "row_idx 41:trajectory[0]"
    assert all(check_pairing(r) == [] for r in runs)


def test_classify_unknown_tool():
    assert openhands.classify("browser", "page loaded") == (None, "unknown", False)
    assert openhands.classify("browser", "ERROR: no page")[1] == "error"
    assert openhands.classify("execute_bash", None) == (None, "unknown", False)


@pytest.mark.network
def test_five_real_rows_pair():
    from agent_trace_integrity.fetch import fetch

    pin = json.loads(PINS.read_text())
    path = fetch(pin["url"], pin["sha256"])
    runs = openhands.load(path)
    assert len(runs) == 5
    assert [r[0].meta["trajectory_id"] for r in runs] == pin["trajectory_ids"]
    for run in runs:
        assert check_pairing(run) == []
        results = [e for e in run if e.role == "tool_result"]
        assert any(e.exit_code is not None for e in results)
        assert run[0].meta["resolved"] in (0, 1)
        # every finish call has no observation in D2, so its result is a placeholder
        fin = [e for e in results if e.tool_name == "finish"]
        assert fin and all(e.tool_result is None for e in fin)
