"""Same bytes in, same bytes out: the CLI's output for the example trace does not change between
runs. Each run is a fresh process with a different hash seed, so set or dict ordering that leaked
into the output would show up here. Offline; no network."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "coding_task_trace.json"


def _ati(*args: str, seed: str) -> tuple[bytes, bytes]:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    p = subprocess.run([sys.executable, "-m", "agent_trace_integrity.cli", *args],
                       capture_output=True, env=env, cwd=ROOT, check=True)
    return p.stdout, p.stderr


@pytest.mark.parametrize("scope", ["final", "all"])
def test_check_output_is_byte_identical_across_runs(scope):
    args = ("check", str(EXAMPLE), "--format", "otel", "--scope", scope)
    first, second = _ati(*args, seed="1"), _ati(*args, seed="2")
    assert first[0]  # some verdicts were written
    assert first == second


def test_events_output_is_byte_identical_across_runs():
    args = ("events", str(EXAMPLE), "--format", "otel")
    first, second = _ati(*args, seed="1"), _ati(*args, seed="2")
    assert first[0].count(b"\n") == 12
    assert first == second
