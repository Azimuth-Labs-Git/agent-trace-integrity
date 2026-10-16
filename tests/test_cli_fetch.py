import hashlib

import pytest

from agent_trace_integrity import fetch as fetch_mod


def test_fetch_file_url_and_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("ATI_CACHE_DIR", str(tmp_path / "cache"))
    src = tmp_path / "log.txt"
    src.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    p = fetch_mod.fetch(src.as_uri(), digest)
    assert p.read_bytes() == b"hello"
    assert fetch_mod.fetch(src.as_uri(), digest) == p  # cached
    with pytest.raises(fetch_mod.HashMismatch):
        fetch_mod.fetch(src.as_uri(), "0" * 64)


def test_unknown_format_is_rejected(tmp_path):
    from agent_trace_integrity.adapters import load

    with pytest.raises(ValueError):
        load(tmp_path / "x", "nope")
