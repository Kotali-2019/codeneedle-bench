"""UTF-8 text I/O helpers.

Python picks the *locale* encoding when `open()`, `Path.read_text()` and
`Path.write_text()` are called without one. On Windows that's typically
cp1252, which cannot represent most of what this project emits — `←` in the
chart navigation, `⚠`/`✓`/`✗`/`≥` in the console report, and the CJK
characters inside the bundled `plotly.min.js`. The result is a hard
UnicodeEncodeError rather than a degraded render:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2190'

Everything this project reads and writes is UTF-8 (TOML mandates it, JSON
defaults to it, and the HTML declares it), so the encoding is never actually
ambiguous — it just has to be stated. Prefer `read_text`/`write_text` from
this module over the pathlib methods so it always is.
"""
from __future__ import annotations

import sys
from pathlib import Path

ENCODING = "utf-8"


def read_text(path: Path | str) -> str:
    """Read a UTF-8 text file regardless of the platform's locale encoding."""
    return Path(path).read_text(encoding=ENCODING)


def write_text(path: Path | str, data: str) -> int:
    """Write a UTF-8 text file regardless of the platform's locale encoding."""
    return Path(path).write_text(data, encoding=ENCODING)


def write_text_atomic(path: Path | str, data: str) -> None:
    """Write a UTF-8 text file so readers never observe a partial one.

    Writes a sibling temp file and renames it over the target: `os.replace` is
    atomic within a filesystem. Needed because the results dump is rewritten
    after every query, so a crash mid-write would otherwise leave truncated
    JSON where a valid earlier dump used to be.
    """
    import os

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding=ENCODING)
    os.replace(tmp, path)


def use_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 so non-ASCII output survives redirection.

    On Windows an interactive console handles Unicode via the console API, but
    the moment output is piped or redirected (`python bench.py run > run.log`)
    the stream falls back to the locale encoding and any `⚠` or `≥` aborts the
    run. Reconfiguring to UTF-8 fixes the common case; `errors="replace"` means
    an exotic terminal degrades to `?` instead of raising.

    Safe to call more than once, and a no-op when the streams are missing
    (pythonw.exe) or already detached.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding=ENCODING, errors="replace")
        except (ValueError, OSError):
            # Already detached or not reconfigurable — nothing more to do.
            pass
