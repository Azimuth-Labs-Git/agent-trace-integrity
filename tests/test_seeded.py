"""Phase 4 gate: the seeded regression set reruns byte-identical, and no control false-alarms."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.network
def test_seeded_rerun_identical(tmp_path):
    out = tmp_path / "seeded.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "seeded_eval.py"), "--out", str(out)],
                   check=True, capture_output=True)
    committed = (ROOT / "results" / "seeded-d2.json").read_text()
    assert out.read_text() == committed
    for r in json.loads(committed)["checks"].values():
        assert r["false_alarms"] == 0
