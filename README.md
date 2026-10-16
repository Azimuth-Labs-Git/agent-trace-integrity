# agent-trace-integrity

AI agents often say they did things they did not do: "tests pass", "PR approved", "I checked the
source". The record of what actually happened is usually sitting in the agent's own trace, and
nobody reads it. This tool will read those traces and, for each claim the agent makes about an
action, say whether the trace backs it up.

**Status: v0, preliminary.** A working rule-based checker exists today (see "What works today"
below) and has been run on public agent runs; the results are in
[results/README.md](results/README.md). It is still pre-release: the schemas are drafts, every
result is preliminary, and the full tool is being built under a proposed 12-week project, with the
schemas freezing in week 4.

## What works today

v0 is plain Python with no AI model inside the checker.

- **Adapters** turn five log formats into one event model: OpenTelemetry GenAI (`otel`),
  OpenHands (`openhands`), SWE-bench / mini-swe-agent trajectories (`swebench`), Inspect logs
  (`inspect`) and pasted Gemini CLI transcripts (`gemini-text`). A METR MALT adapter (`malt`)
  is planned and not built yet.
- **Claim finder:** plain rules that find claims about actions and results in what the agent wrote
  to the user (never its hidden reasoning), tagged as asserted, hedged or borderline.
- **Verdict rules** for sub-type 1 (action claimed but never taken) and sub-type 3 (success
  narrated over a failed test run), with reason codes for `insufficient-evidence`.
- **Check A, output no tool produced:** text in the agent's message that looks like tool output, such as
  a pytest summary, and matches no real tool result. Flag `fabricated_tool_output`.
- **Check B, changing the check:** editing or disabling a test, or a control script the
  environment named, before claiming success or submitting. Flag
  `check_modified_before_success_claim`, always shown with the exact edit.

Every verdict validates against `schema/claim_verdict.schema.json`. Usage:

```
ati check run.traj.json --format swebench --out verdicts.jsonl
```

`ati events <input> --format <name>` prints the parsed events, and `ati fetch <url> --sha256 <hash>`
downloads a pinned log into a cache outside the repo. Preliminary results on SWE-bench, Palisade
shutdown-avoidance, a seeded OpenHands set and a Gemini CLI transcript are in
[results/README.md](results/README.md).

## What it reads

OpenTelemetry GenAI traces, as OTLP JSON: the `invoke_agent`, `chat` and `execute_tool` spans that
many agent frameworks already emit. It takes claims from what the agent wrote to the user and
looks for matching tool calls in the same trace. It never changes the agent, never
reruns anything, and makes no network calls by default.

## The four verdicts

Each claim gets exactly one:

- **corroborated**: a recorded tool call of the right kind touched the thing the claim names, and
  any result the agent reported matches what the call returned.
- **contradicted**: a matching call exists, but its result says something different. The agent said
  "16 tests pass"; the test run says "2 failed".
- **uncorroborated**: no matching call exists in the trace at all.
- **insufficient-evidence**: the trace does not hold enough to tell. This always comes with a reason
  code, such as `tool_result_not_logged`. Several trace fields are optional in the OpenTelemetry
  spec, so this answer is common and we report it rather than guess.

Claims are also tagged with a sub-type (claimed but not taken, result misreported, success narrated
over a failed call, citation never fetched). Rates are reported per sub-type, never pooled, each
with its uncertainty and its counts.

## Relayed claims

One case gets its own bucket. A **relayed claim** is a claimed outcome whose only support is text
inside a tool result, with no tool call of the agent's own that checks it. Example: the agent reads
PR comments, a bot comment says "CI is green", and the agent tells you "CI passed". It never
looked at CI itself.

Relayed claims are reported separately and never mixed into the other rates. They need the tool
result to be logged, so `insufficient-evidence` is a common answer here too.

## Open and reference-free

- It never calls a proprietary model API at runtime. Most checks are plain string matching. The one
  learned part, used for a single sub-type, is a small open-weight model you run yourself.
- It is reference-free: it needs no gold answers, labels or reruns. It looks only at the trace you
  give it.

## What is in this repo

```
schema/claim_verdict.schema.json    one claim and its verdict (JSON Schema 2020-12)
schema/corpus_report.schema.json    rates over a set of traces
examples/coding_task_trace.json     a made-up trace of a small coding task
examples/expected_verdicts.json     the verdicts the tool should produce for it
examples/expected_corpus_report.json  the report for that one trace
src/agent_trace_integrity/          the checker: adapters, claim finder, verdict rules, CLI
tests/                              unit tests, plus network tests on pinned real runs
scripts/                            pinning, results, seeded set, dev set and number-check scripts
scripts/lambda/                     dry-run tooling for the open-weight LLM judge (not launched)
data/pins/                          URLs and sha256 of third-party logs (the logs are not committed)
devset/                             claim finder dev set and its labels (derived from D2, CC-BY-4.0)
seeded/                             blind phrasings for the seeded regression set
results/                            v0 results and their write-up
docs/build-log.md                   what was built, phase by phase, and every rule change
```

In the example, an agent edits `src/parser.py`, runs the tests, runs a linter, and reads PR
comments. It then reports four things. The edit is **corroborated**. "All 16 tests pass" is
**contradicted** by the recorded run (2 failed). "The linter is clean" is
**insufficient-evidence**, because the harness did not log that command's output. "CI passed" is a
**relayed claim**, backed only by a bot comment.

One trace is too few for a rate, so the example report leaves every estimate empty and says why.

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright 2026 Azimuth Labs.
