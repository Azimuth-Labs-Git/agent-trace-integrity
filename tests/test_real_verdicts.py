"""Phase 3 validation on real runs: every verdict the checker emits on the pinned D1, D2, D3
and D5 runs validates against the schema. Network only (third-party logs are not committed)."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from agent_trace_integrity.adapters import load
from agent_trace_integrity.adapters.gemini_text import load_lines
from agent_trace_integrity.fetch import fetch
from agent_trace_integrity.verdicts import check_run

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "tests" / "pins"
VALIDATOR = Draft202012Validator(json.loads((ROOT / "schema" / "claim_verdict.schema.json")
                                            .read_text()))
D5 = ("https://github.com/user-attachments/files/21372906/gemini.cli.screw.up.txt",
      "000ddab20df84d46ebbe70b36a73f21eadb6761b190f3d16722f404c31a34935")


def _runs():
    sw = json.loads((PINS / "swebench.json").read_text())["runs"]
    for r in sw:
        yield "swebench", load(fetch(r["url"], r["sha256"]), "swebench")
    oh = json.loads((PINS / "openhands.json").read_text())
    yield "openhands", load(fetch(oh["url"], oh["sha256"]), "openhands")
    for v in json.loads((PINS / "inspect.json").read_text()).values():
        yield "inspect", load(fetch(v["url"], v["sha256"]), "inspect")
    yield "gemini-text", [load_lines(fetch(*D5), 1, 511)[0]]


@pytest.mark.network
def test_every_real_verdict_validates():
    total = 0
    for _, runs in _runs():
        for run in runs:
            for v in check_run(run, "all"):
                VALIDATOR.validate(v)
                total += 1
    assert total > 100
