"""The Lambda scripts never touch the network in a dry run, and refuse real calls without a yes.

Every test blocks urllib, raw sockets and subprocess, so any attempted call fails the test.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "scripts" / "lambda"
EXAMPLE = LAMBDA_DIR / "example_borderline.jsonl"


class NetworkCalled(BaseException):
    """BaseException so no `except Exception` in a script can swallow it."""


def _blocked(*_a, **_k):
    raise NetworkCalled("a network or subprocess call was attempted")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(subprocess, "run", _blocked)
    monkeypatch.setattr(subprocess, "Popen", _blocked)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)


def _load(name):
    if str(LAMBDA_DIR) not in sys.path:
        sys.path.insert(0, str(LAMBDA_DIR))
    spec = importlib.util.spec_from_file_location(f"lambda_{name}", LAMBDA_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAUNCH = ["--instance-type", "gpu_1x_a10", "--region", "us-east-1", "--ssh-key", "test-key"]


def test_launch_dry_run_prints_body_and_cost(tmp_path, capsys):
    launch = _load("launch")
    rc = launch.main(LAUNCH + ["--hours", "3", "--price-per-hour", "0.75", "--dry-run",
                               "--local-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"instance_type_name": "gpu_1x_a10"' in out
    assert '"quantity": 1' in out
    assert "$2.25" in out
    assert not list(tmp_path.iterdir())


def test_launch_over_cap_exits_nonzero(tmp_path):
    launch = _load("launch")
    rc = launch.main(LAUNCH + ["--hours", "100", "--price-per-hour", "2.0", "--dry-run",
                               "--local-dir", str(tmp_path)])
    assert rc == 2


def test_launch_over_cap_refused_even_with_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "not-a-real-key")
    launch = _load("launch")
    rc = launch.main(LAUNCH + ["--hours", "100", "--price-per-hour", "2.0",
                               "--i-have-matts-yes", "--local-dir", str(tmp_path)])
    assert rc == 2


def test_launch_without_yes_exits_before_network(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "not-a-real-key")
    launch = _load("launch")
    rc = launch.main(LAUNCH + ["--hours", "3", "--price-per-hour", "0.75",
                               "--local-dir", str(tmp_path)])
    assert rc == 3
    assert not list(tmp_path.iterdir())


def test_launch_with_yes_but_no_key_exits_before_network(tmp_path):
    launch = _load("launch")
    rc = launch.main(LAUNCH + ["--hours", "3", "--price-per-hour", "0.75",
                               "--i-have-matts-yes", "--local-dir", str(tmp_path)])
    assert rc == 4


def test_launch_cli_dry_run_as_script(tmp_path, monkeypatch):
    """The script also works as a command (run in-process via runpy, sockets still blocked)."""
    import runpy

    monkeypatch.setattr(sys, "argv", ["launch.py", *LAUNCH, "--hours", "3", "--price-per-hour",
                                      "0.75", "--dry-run", "--local-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(LAMBDA_DIR / "launch.py"), run_name="__main__")
    assert exc.value.code == 0


def test_teardown_dry_run_without_record(tmp_path, capsys):
    teardown = _load("teardown")
    rc = teardown.main(["--dry-run", "--local-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "/instance-operations/terminate" in out
    assert "GET" in out and "/instances" in out


def test_teardown_dry_run_with_record(tmp_path, capsys):
    (tmp_path / "lambda-instance.json").write_text(json.dumps({"instance_id": "abc123"}))
    teardown = _load("teardown")
    assert teardown.main(["--dry-run", "--local-dir", str(tmp_path)]) == 0
    assert "abc123" in capsys.readouterr().out
    assert (tmp_path / "lambda-instance.json").exists()


def test_teardown_without_yes_exits_before_network(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "not-a-real-key")
    (tmp_path / "lambda-instance.json").write_text(json.dumps({"instance_id": "abc123"}))
    teardown = _load("teardown")
    assert teardown.main(["--local-dir", str(tmp_path)]) == 3
    assert (tmp_path / "lambda-instance.json").exists()


def test_run_judge_dry_run_prints_commands_and_three_prompts(tmp_path, capsys):
    judge = _load("run_judge")
    rc = judge.main(["--host", "ubuntu@203.0.113.7", "--input", str(EXAMPLE), "--dry-run",
                     "--local-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("--- prompt for id") == 3
    assert "scp" in out and "-L 18000:127.0.0.1:8000" in out and "rm -rf" in out
    assert '"temperature": 0' in out
    assert judge.QUESTION in out.replace("\\n", "\n")
    assert not list(tmp_path.iterdir())


def test_run_judge_answer_parsing():
    judge = _load("run_judge")
    assert judge.parse_answer("Yes.") == "yes"
    assert judge.parse_answer("<think>hmm</think>\nno") == "no"
    assert judge.parse_answer("maybe") == "unparsed"
    assert judge.model_slug("Qwen/Qwen3.5-4B") == "Qwen_Qwen3.5-4B"
