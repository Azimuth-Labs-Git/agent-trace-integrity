"""Download third-party logs by URL, verify sha256, cache outside the repo.

Third-party logs are never committed. They live in a cache directory, by default
``~/.cache/agent-trace-integrity`` (override with ATI_CACHE_DIR), named by their sha256.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


class HashMismatch(Exception):
    pass


def cache_dir() -> Path:
    d = Path(os.environ.get("ATI_CACHE_DIR", Path.home() / ".cache" / "agent-trace-integrity"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(req: urllib.request.Request, out, retries: int) -> None:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                shutil.copyfileobj(resp, out)
            return
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        out.seek(0)
        out.truncate()
        time.sleep(2 ** attempt)


def fetch(url: str, sha256: str | None = None, headers: dict[str, str] | None = None,
          retries: int = 4) -> Path:
    """Return a local path for ``url``. With ``sha256``, a cached copy is reused and a download
    that does not match raises HashMismatch (and is deleted). Without it, the file is always
    downloaded and its hash is printed by the CLI so it can be pinned."""
    cache = cache_dir()
    if sha256:
        cached = cache / sha256
        if cached.exists() and sha256_file(cached) == sha256:
            return cached
    req = urllib.request.Request(url, headers=headers or {})
    fd, tmp = tempfile.mkstemp(dir=cache, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            _download(req, out, retries)
        got = sha256_file(Path(tmp))
        if sha256 and got != sha256:
            raise HashMismatch(f"{url}: expected {sha256}, got {got}")
        final = cache / got
        os.replace(tmp, final)
        return final
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
