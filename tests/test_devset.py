"""The Phase 2 gate, rerun offline: claim-finder precision on the dev set stays at or above
0.80 against the adjudicated reference (scored by scripts/dev_eval.py)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dev_precision_floor():
    d = ROOT / "devset"
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "dev_eval.py"), str(d / "dev_msgs.json"),
         str(d / "labels_A.json"), str(d / "labels_B.json"), "--adjudicated",
         str(d / "adjudicated.json")], capture_output=True, text=True, check=True).stdout
    report = json.loads(out[: out.index("\n}") + 2])
    assert report["precision"] >= 0.80
    assert report["precision_wilson95"][0] >= 0.80
