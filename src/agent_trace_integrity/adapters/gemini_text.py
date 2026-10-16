"""Adapter for Gemini CLI terminal transcripts copied as plain text (dataset D5).

A copied transcript is what the terminal showed, not a structured log, so this adapter reads the
screen layout:

- User prompts are boxes drawn at column 0 whose first inner line starts with ``>``.
- Model narration starts with ``✦`` at column 0 and continues on indented or blank lines.
- Tool calls are boxes indented by one space. Each header line inside a box (``│ <icon>  Name
  args``) starts one tool call; the lines under it, up to the next header or the box end, are
  whatever result text survived the copy. One box can hold several calls.
- Lines at column 0 starting with ``✕`` or ``ℹ`` are CLI notices and become ``system`` events.
- Anything else (the shell prompt before the CLI started, the banner, tips) is skipped.

Status rules (plan Phase 1): status comes from the result text and exit codes, never from the
``✔``/``x`` icons. The icon is kept in ``meta["ui_icon"]`` only so later checks can show where the
UI and the result disagree. In order:

1. ``Command exited with code: N`` (or ``Exit Code: N``) sets ``exit_code`` and the status.
2. Known error text (``Error:``, ``syntax is incorrect``, ``cannot find the file``, ...) gives
   ``error``.
3. Known harness truncation markers give ``truncated``.
4. Known tool success text (``Listed N item(s)``, ``No files found``, ``Successfully ...``) gives
   ``ok``.
5. Anything else, including plain command output with no exit code, is ``unknown``.

Lost results: a copied terminal often loses the result body (blank box, or only a stray border
character). Then ``tool_result`` is ``None`` (not logged) and status is ``unknown``. A
``tool_result`` event is still emitted right after every ``tool_call`` so that pairing holds; its
``meta["result_lost"]`` is True. Lines whose left box border was lost in the copy are kept from the
first surviving border onward and the result gets ``meta["copy_damaged"] = True``.

``tool_result`` text is the box body with the box border characters and the fixed four-space result
indent removed, otherwise unchanged. User text joins soft-wrapped lines with single spaces.

Every event's ``source_ref`` is ``L<n>`` or ``L<a>-L<b>`` with 1-based line numbers in the original
file, also when only a line range is loaded. Use ``line_span`` to read it back.

Tool calls have no ids in this format, so ids are ``"{run}-call-{n}"`` and each result follows its
call directly.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..events import Event, reindex, status_from_exit_code

FORMAT = "gemini-text"

ICONS = "✔✓xX✕✗×?⊷o"
HEADER = re.compile(r"^\s*│\s{1,2}([" + ICONS + r"])\s{1,3}(\S.*?)\s*│?\s*$")
MULTIWORD_TOOLS = ("Save Memory", "Google Search")
NOTICE_MARKS = ("✕", "ℹ")

EXIT_CODE = re.compile(r"(?:Command exited with code|Exit Code):\s*(-?\d+)", re.IGNORECASE)
ERROR_TEXT = [
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"^\s*Error:",
        r"syntax is incorrect",
        r"cannot find the (?:file|path) specified",
        r"is not recognized as an internal or external command",
        r"Access is denied",
        r"Failed to execute tool",
        r"No such file or directory",
        r"command not found",
        r"Permission denied",
        r"resolves outside the tool's root directory",
    )
]
TRUNCATED_TEXT = [
    re.compile(p, re.IGNORECASE)
    for p in (r"lines? hidden", r"\[truncated\]", r"output (?:was )?truncated")
]
OK_TEXT = [
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"^\s*Listed \d+ item\(s\)",
        r"^\s*No files found",
        r"^\s*Found \d+ ",
        r"^\s*Successfully ",
        r"^\s*Read \d+ ",
        r"^\s*Read lines \d+",
    )
]


def line_span(ref: str) -> tuple[int, int]:
    """Parse a source_ref like ``L12`` or ``L12-L15`` into (start, end), both inclusive."""
    m = re.fullmatch(r"L(\d+)(?:-L(\d+))?", ref)
    if not m:
        raise ValueError(f"not a line ref: {ref!r}")
    a = int(m.group(1))
    return a, int(m.group(2) or a)


def _ref(a: int, b: int) -> str:
    return f"L{a}" if a == b else f"L{a}-L{b}"


def result_status(text: str | None) -> tuple[str, int | None]:
    """Return (status, exit_code) from result text only."""
    if text is None:
        return "unknown", None
    m = EXIT_CODE.search(text)
    if m:
        code = int(m.group(1))
        return status_from_exit_code(code), code
    if any(p.search(text) for p in ERROR_TEXT):
        return "error", None
    if any(p.search(text) for p in TRUNCATED_TEXT):
        return "truncated", None
    if any(p.search(text) for p in OK_TEXT):
        return "ok", None
    return "unknown", None


def _strip_border(line: str, box_indent: int) -> tuple[str, bool]:
    """Return (content, damaged) for one inner box line."""
    s = line.rstrip()
    if not s.strip():
        return "", False
    indent = len(s) - len(s.lstrip())
    body = s.lstrip()
    damaged = indent != box_indent or not body.startswith("│")
    if body.startswith("│"):
        body = body[1:]
    if body.endswith("│"):
        body = body[:-1]
    body = body.rstrip()
    if not damaged:
        # Result text sits four spaces in from the border.
        body = body[4:] if body.startswith("    ") else body.lstrip()
    return body, damaged


def _split_tool(rest: str) -> tuple[str, str]:
    for name in MULTIWORD_TOOLS:
        if rest == name or rest.startswith(name + " "):
            return name, rest[len(name) :].strip()
    name, _, args = rest.partition(" ")
    return name, args.strip()


def _parse_lines(lines: list[tuple[int, str]], run: str) -> list[Event]:
    events: list[Event] = []
    ncall = 0
    i = 0
    n = len(lines)

    def is_box_start(s: str) -> bool:
        return s.lstrip().startswith("╭")

    while i < n:
        lno, line = lines[i]
        stripped = line.strip()

        if is_box_start(line):
            box_indent = len(line) - len(line.lstrip())
            start = lno
            inner: list[tuple[int, str]] = []
            i += 1
            end = start
            while i < n:
                lno2, l2 = lines[i]
                if l2.lstrip().startswith("╰"):
                    end = lno2
                    i += 1
                    break
                if is_box_start(l2):
                    break
                inner.append((lno2, l2))
                end = lno2
                i += 1
            events.extend(_parse_box(start, end, box_indent, inner, run, ncall))
            ncall = sum(1 for e in events if e.role == "tool_call")
            continue

        if line.startswith("✦"):
            start = lno
            text_lines = [line[1:].strip()]
            last = lno
            i += 1
            while i < n:
                lno2, l2 = lines[i]
                if is_box_start(l2) or (l2[:1] and not l2[:1].isspace()):
                    break
                if l2.strip():
                    text_lines.append(l2[2:] if l2.startswith("  ") else l2.strip())
                    last = lno2
                else:
                    text_lines.append("")
                i += 1
            text = "\n".join(text_lines).strip("\n")
            events.append(Event(0, "assistant", text=text, source_ref=_ref(start, last)))
            continue

        if stripped and line[:1] in NOTICE_MARKS:
            start = lno
            text_lines = [stripped]
            i += 1
            while i < n and lines[i][1].strip() and lines[i][1][:1].isspace():
                text_lines.append(lines[i][1].strip())
                i += 1
            last = lines[i - 1][0]
            events.append(
                Event(0, "system", text="\n".join(text_lines), source_ref=_ref(start, last))
            )
            continue

        i += 1

    return reindex(events)


def _parse_box(
    start: int,
    end: int,
    box_indent: int,
    inner: list[tuple[int, str]],
    run: str,
    ncall: int,
) -> list[Event]:
    first = next((ln for _, ln in inner if ln.strip()), "")
    first_content = first.strip().lstrip("│").strip()

    if first_content.startswith(">"):
        parts = []
        for _, ln in inner:
            c = ln.strip()
            if c.startswith("│"):
                c = c[1:]
            if c.endswith("│"):
                c = c[:-1]
            c = c.strip()
            if c:
                parts.append(c)
        if parts:
            parts[0] = parts[0][1:].strip()
        return [Event(0, "user", text=" ".join(parts), source_ref=_ref(start, end))]

    headers = [(k, HEADER.match(ln)) for k, (_, ln) in enumerate(inner)]
    headers = [(k, m) for k, m in headers if m]
    if not headers:
        texts = [_strip_border(ln, box_indent)[0] for _, ln in inner]
        text = "\n".join(texts).strip("\n") or None
        return [Event(0, "system", text=text, source_ref=_ref(start, end), meta={"box": True})]

    out: list[Event] = []
    for h, (k, m) in enumerate(headers):
        nxt = headers[h + 1][0] if h + 1 < len(headers) else len(inner)
        header_lno = inner[k][0]
        icon, rest = m.group(1), m.group(2)
        name, args = _split_tool(rest)
        ncall += 1
        cid = f"{run}-call-{ncall}"
        out.append(
            Event(
                0,
                "tool_call",
                tool_name=name,
                tool_args=args,
                tool_call_id=cid,
                source_ref=_ref(header_lno, header_lno),
                meta={"ui_icon": icon, "header_truncated": rest.endswith("…")},
            )
        )
        body = inner[k + 1 : nxt]
        contents: list[str] = []
        damaged = False
        for _, ln in body:
            c, d = _strip_border(ln, box_indent)
            damaged = damaged or d
            contents.append(c)
        while contents and not contents[0].strip():
            contents.pop(0)
        while contents and not contents[-1].strip():
            contents.pop()
        result = "\n".join(contents) if contents else None
        status, code = result_status(result)
        if body:
            ref = _ref(body[0][0], body[-1][0])
        else:
            ref = _ref(header_lno, header_lno)
        out.append(
            Event(
                0,
                "tool_result",
                tool_name=name,
                tool_call_id=cid,
                tool_result=result,
                exit_code=code,
                status=status,
                source_ref=ref,
                meta={"copy_damaged": damaged, "result_lost": result is None,
                      "not_logged": result is None},
            )
        )
    return out


def _read(path: Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def load_lines(
    path: str | Path, start: int = 1, end: int | None = None, run: str | None = None
) -> list[list[Event]]:
    """Parse lines ``start`` to ``end`` (1-based, inclusive) of a transcript. Line numbers in
    ``source_ref`` stay those of the whole file."""
    all_lines = _read(Path(path))
    end = len(all_lines) if end is None else min(end, len(all_lines))
    sel = [(k, all_lines[k - 1]) for k in range(max(start, 1), end + 1)]
    run = run or Path(path).stem[:12] or "run"
    return [_parse_lines(sel, run)]


def load(path: str | Path) -> list[list[Event]]:
    """Parse a whole transcript as one run."""
    return load_lines(path)
