"""Shared helpers for the Lambda scripts: paths, the confirmation flag, and one HTTP call.

Stdlib only. Nothing here runs on import. The only function that touches the network is
`api_call`, and every script calls it only after its own dry-run and confirmation checks.

UNVERIFIED (check at launch against Lambda's API docs): the base URL below (Lambda has also
served the API from https://cloud.lambda.ai/api/v1) and Basic auth with the key as the user name.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

API_BASE = "https://cloud.lambdalabs.com/api/v1"
ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "local"
INSTANCE_FILE_NAME = "lambda-instance.json"
KEY_ENV = "LAMBDA_API_KEY"

# Exit codes shared by the scripts.
EXIT_OVER_CAP = 2
EXIT_NO_YES = 3
EXIT_NO_KEY = 4
EXIT_BAD_INPUT = 5

NO_YES_MESSAGE = (
    "refusing: a real call needs --i-have-matts-yes. Claude never launches or stops an "
    "instance without Matt's yes with the instance type and expected hours."
)


def api_key() -> str | None:
    key = os.environ.get(KEY_ENV, "").strip()
    return key or None


def api_call(method: str, path: str, key: str, body: dict | None = None,
             timeout: float = 30.0) -> dict:
    """One request to the Lambda Cloud API. Returns the decoded JSON body."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API_BASE + path, data=data, method=method)
    token = base64.b64encode(f"{key}:".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode() or "{}")
