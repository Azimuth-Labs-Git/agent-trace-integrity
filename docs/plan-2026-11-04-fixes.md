# Plan, 2026-11-04: four small fixes found in a line-by-line read

Branch `fix/otel-call-id-readme-ci-runref`. Baseline before any change:
`uv run --extra dev pytest -q` gives 112 passed, 10 skipped (network); ruff clean.
`ati check examples/coding_task_trace.json --format otel` (both scopes) and `ati events` output
saved before the change, for a byte-for-byte comparison after.

No change may alter a verdict except where a fix requires it.

## 1. OTel adapter: missing `gen_ai.tool.call.id`

Bug (otel.py, `load`): with no call id, the first `execute_tool` adds `None` to `seen_calls`,
so every later id-less `execute_tool` skips building its call event and emits only an orphan
result. `check_pairing` then fails and the calls vanish from verdicts.

Fix:
- A `tool_call` part on a chat span with no `id` gets `"{trace_id}-call-{n}"` (n = position
  among the run's calls, 1-based), as events.py documents.
- An `execute_tool` span with no id is paired by order: it takes the oldest still-open call from
  a chat span with the same tool name (or either name missing). If there is none, the call event
  is built from the span itself, with a generated id.
- Whenever an id was generated or a result was paired by order, the run gets
  `events[0].meta["join_key_absent"] = True`.
- `verdicts.check_run` adds the schema flag `join_key_absent` to every verdict of such a run.
  It is added after the final-submit step, not inside `_verdict`, so it cannot by itself make
  the synthetic check-B submit verdict appear (that verdict is kept only when it has flags).
- Also add `meta["format"] = "otel"` to the first event, like the other adapters.

Tests: the example trace with every call id stripped (chat parts and `execute_tool` spans), and a
second variant whose chat spans also drop `gen_ai.output.messages`. Assert all five calls become
events, `check_pairing` is clean, each result sits with the right call, the flag is on every
verdict, and verdicts otherwise equal the example's. The example (which has ids) must not get the
flag. `ati check` on the example must be byte-identical before and after (`ati events` gains only
the `format` key on event 0).

## 2. README

- Line 57: "no matching call exists in the trace at all" becomes "no matching call before the
  claim". The rules only look at earlier calls (verdicts.py `c.index < claim.event_index`).
- Lines 78-79: the "small open-weight model" for one sub-type is not in the code
  (`evidence_basis` is always `deterministic`). Say it is planned, behind a flag, not in v0.
- Relayed claims (66-74) and the worked example (102-106) read as if v0 produces them. Add that
  relayed claims and the lint `insufficient-evidence` example are not in v0
  (`relayed_source` is always null; build-log.md records this). Nothing else is changed.

## 3. `run_ref` fallback

Bug (verdicts.py `run_ref`): MALT sets `run_id`, gemini-text sets no run meta, so both fall
back to the literal `"run"` and every such run shares one `trace_id` and synthetic span ids.

Fix:
- Add `run_id` to the end of the lookup (after `file`), so only runs that used to fall back to
  `"run"` change. Inspect sets `file`, so it is unaffected.
- gemini-text sets `format`, `file` (the file name, not the path, so output does not depend on
  the machine) and `lines` on event 0.

Tests: two MALT rows with different `run_id`, and two gemini files with different names, get
different `trace_id`s.

Published results: `results/*.json` hold no `trace_id`, `run_ref` or span ids (checked by grep),
so nothing published changes. Results are not regenerated (that needs network and pinned data).

## 4. CI

- Move `docs/github-actions-ci.yml` to `.github/workflows/ci.yml`. Run on push to `main` and on
  pull requests; matrix Python 3.11 and 3.12 through `astral-sh/setup-uv`; `uv sync --locked`,
  ruff, then pytest. Network tests stay skipped (conftest skips them unless
  `ATI_NETWORK_TESTS=1`).
- Note: `pytest` is in the `dev` extra, so the command is `uv run --extra dev pytest`.
- New offline tests: `ati check` on the example run twice through the CLI entry point gives
  identical bytes (both scopes), and so does `ati events`.
- After merge, branch protection on `main` requires both matrix checks, strict, no reviews,
  admins not enforced.

## 5. Build log

One dated entry in docs/build-log.md listing these fixes and the test counts.
