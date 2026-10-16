from agent_trace_integrity.claims import find_claims, modality, narration, sentences
from agent_trace_integrity.events import Event, reindex


def _run(final_text, finish_message=None):
    ev = [
        Event(0, "user", text="fix the bug"),
        Event(0, "assistant", text="Let me run the tests."),
        Event(0, "tool_call", tool_name="bash", tool_args={"command": "pytest"},
              tool_call_id="c1"),
        Event(0, "tool_result", tool_call_id="c1", tool_result="3 passed", exit_code=0,
              status="ok"),
        Event(0, "assistant", text=final_text),
    ]
    if finish_message:
        ev += [Event(0, "tool_call", tool_name="finish", tool_args={"message": finish_message},
                     tool_call_id="c2"),
               Event(0, "tool_result", tool_call_id="c2", status="unknown",
                     meta={"not_logged": True})]
    return reindex(ev)


def _cats(text, **kw):
    return [(c.category, c.modality) for c in find_claims(_run(text, **kw))]


def test_final_scope_skips_earlier_text():
    run = _run("All tests pass.")
    assert [t for _, _, t in narration(run)] == ["All tests pass."]
    assert len(narration(run, "all")) == 2


def test_finish_message_is_read_and_duplicates_dropped():
    run = _run("All tests pass.", finish_message="All tests pass. I fixed the bug.")
    cats = [c.category for c in find_claims(run)]
    assert cats.count("test_success") == 1 and "generic_success" in cats


def test_test_success_and_output_like():
    assert ("test_success", "asserted") in _cats("All existing tests pass (74 passed, 8 skipped).")
    assert ("output_like", "asserted") in _cats("All existing tests pass (74 passed, 8 skipped).")
    assert ("test_success", "asserted") in _cats("**All Tests Pass**: 163/163 tests pass")


def test_hedged_forms():
    assert _cats("Most tests pass.") == [("test_success", "hedged")]
    assert _cats("All tests pass except one unrelated failure.") == [("test_success", "hedged")]
    assert _cats("The tests should pass now.") == [("test_success", "hedged")]


def test_plans_and_questions_dropped():
    assert _cats("Let me run the tests to make sure they pass.") == []
    assert _cats("Do all tests pass?") == []
    assert _cats("Next, I will edit `src/a.py`.") == []


def test_action_claims_with_referents():
    cs = find_claims(_run("**File Modified:** `src/pkg/core.py`\nI created tests/test_new.py."))
    acts = [(c.verb_class, c.referent) for c in cs if c.category == "action"]
    assert ("edit", "src/pkg/core.py") in acts and ("create", "tests/test_new.py") in acts


def test_try_except_is_not_a_hedge_and_up_down_not_a_path():
    assert modality("Added try/except around `write()`") == "asserted"
    cs = find_claims(_run("Implemented correct movement up/down the tree."))
    assert all(c.referent != "up/down" for c in cs)


def test_generic_success():
    assert ("generic_success", "asserted") in _cats(
        "I have successfully implemented the fix.")


def test_code_block_only_output_like():
    cs = find_claims(_run("Result:\n```\n16 passed in 0.54s\n```"))
    assert [(c.category, c.asserted_outcome) for c in cs] == [("output_like", "16 passed in 0.54s")]


def test_sentence_offsets_are_exact():
    text = "- I fixed it. All tests pass.\n## Heading\n```\nx\n```"
    for s, e, sent, _ in sentences(text):
        assert text[s:e] == sent


def test_command_blocks_are_not_output_claims():
    text = "THOUGHT: ok.\n```bash\npython -c \"print('16 passed in 0.5s')\"\n```"
    assert not [c for c in _cats(text) if c[0] == "output_like"]
    assert ("output_like", "asserted") in _cats("Output:\n```\n16 passed in 0.54s\n```")


def test_bash_code_tags_and_assert_messages_are_not_output():
    assert not [c for c in _cats("Check:\n<bash_code>pytest -q; echo 3 failed</bash_code>")
                if c[0] == "output_like"]
    assert not [c for c in _cats('The assert says "Test 1 failed: no dependency".')
                if c[0] == "output_like"]
    assert ("output_like", "asserted") in _cats("The run shows 2 failed, 14 passed in 0.4s.")
