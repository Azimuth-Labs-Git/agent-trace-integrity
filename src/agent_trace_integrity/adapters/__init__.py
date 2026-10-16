"""Format adapters. Each module exposes ``load(path) -> list[list[Event]]`` (a file may hold
several runs) and ``FORMAT`` (the CLI name). Modules are looked up by name so adapters can be
added without editing a shared table."""

from __future__ import annotations

import importlib
from pathlib import Path

from ..events import Event

FORMATS = {
    "otel": "otel",
    "openhands": "openhands",
    "swebench": "swebench",
    "inspect": "inspect_eval",
    "malt": "malt",
    "gemini-text": "gemini_text",
}


def load(path: str | Path, fmt: str) -> list[list[Event]]:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(FORMATS)}")
    try:
        mod = importlib.import_module(f".{FORMATS[fmt]}", __name__)
    except ModuleNotFoundError as exc:
        raise NotImplementedError(f"adapter for {fmt!r} is not built yet") from exc
    return mod.load(Path(path))
