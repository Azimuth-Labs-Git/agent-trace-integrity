"""MALT adapter on made-up rows shaped like the dataset (no MALT content is committed)."""

from agent_trace_integrity.adapters.malt import parse_row
from agent_trace_integrity.events import check_pairing
from agent_trace_integrity.verdicts import check_run


def _node(i, parent, role, content="", branch=0, fc=None, name=None):
    return {"node_id": i, "parent_node_id": parent, "branch_id": branch,
            "node_data": {"message": {"role": role, "content": content, "name": name,
                                      "function_call": fc}, "metadata": {}}}


def _row(nodes, **kw):
    return {"run_id": 1, "task_id": "toy/task", "model": "example-model", "labels": ["normal"],
            "manually_reviewed": False, "run_source": "unprompted", "nodes": nodes, **kw}


def test_text_tags_path_and_exit_codes():
    nodes = [
        _node(0, None, "system", "You control a VM."),
        _node(1, 0, "user", "Fix the bug."),
        _node(2, 1, "user", "So far you have used 0 tokens.", branch=3),
        _node(3, 1, "assistant", "Let me look. <bash>cat a.py</bash>"),
        _node(4, 3, "user", "<bash-output>x = 1\n\nExit code: 0</bash-output>"),
        _node(5, 4, "assistant", "Now test. <bash>pytest -q</bash>"),
        _node(6, 5, "user", "<bash-output>1 failed, 3 passed\n\nExit code: 1</bash-output>"),
        _node(7, 6, "assistant", "All tests pass."),
    ]
    run = parse_row(_row(nodes))
    assert check_pairing(run) == []
    assert [e.role for e in run].count("tool_call") == 2
    assert not any("So far" in (e.text or "") for e in run)  # side branch dropped
    res = [e for e in run if e.role == "tool_result"]
    assert [r.exit_code for r in res] == [0, 1] and res[1].status == "error"
    assert run[0].meta["labels"] == ["normal"]
    [v] = [v for v in check_run(run) if v["claim"]["category"] == "test_success"]
    assert v["verdict"] == "contradicted"


def test_function_call_text_and_structured():
    nodes = [
        _node(0, None, "system", "go"),
        _node(1, 0, "assistant",
              ' function_call: {"name": "run_bash", "arguments": "{\\"command\\":\\"ls\\"}"}'),
        _node(2, 1, "user", "<run_bash-output> a.py b.py"),
        _node(3, 2, "assistant", "", fc={"name": "submit", "arguments": '{"answer": "42"}'}),
    ]
    run = parse_row(_row(nodes))
    calls = [e for e in run if e.role == "tool_call"]
    assert [(c.tool_name, c.tool_args) for c in calls] == [
        ("run_bash", {"command": "ls"}), ("submit", {"answer": "42"})]
    assert check_pairing(run) == []
    assert run[-1].meta.get("not_logged") is True


def test_path_prefers_branch_with_actions_over_branch_zero_side_question():
    nodes = [
        _node(0, None, "system", "Which file should I write to?", branch=0),
        _node(5, 0, "assistant", "solution/model.py", branch=0),
        _node(1, None, "system", "You control a VM.", branch=7),
        _node(2, 1, "assistant", "<bash>ls</bash>", branch=7),
        _node(3, 2, "user", "<bash-output>a.py\n\nExit code: 0", branch=7),
        _node(4, 3, "assistant", "Done.", branch=7),
    ]
    run = parse_row(_row(nodes))
    assert [e.role for e in run if e.role == "tool_call"] == ["tool_call"]
    assert run[0].text == "You control a VM."
