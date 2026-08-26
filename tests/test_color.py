"""Colour detection — raised in issue #5.

A user pasted results containing literal escape codes:

    ←[32m✓←[0m _url_collapse_path    matched=20/20

Colour was decided solely by `sys.stdout.isatty()`. Windows consoles report
True there but historically render ANSI literally, so the codes end up as text
in whatever the user copies out. There was also no NO_COLOR support.
"""
from __future__ import annotations

import io
import json
import subprocess

import pytest

from bench import report
from bench.report import color_enabled, render_function, render_summary, set_color_override
from bench.scorer import FunctionScore, score
from bench.textio import write_text

ESC = "\x1b"


@pytest.fixture(autouse=True)
def _reset_override():
    set_color_override(None)
    yield
    set_color_override(None)


def _score(passed=True):
    return FunctionScore(
        name="fn", primary_matched=20 if passed else 0, primary_total=20,
        hallucinated=0, bonus_matched=0, passed=passed,
        expected_tagged=[], predicted_tagged=[],
    )


class _Tty(io.StringIO):
    def isatty(self):
        return True


class _Pipe(io.StringIO):
    def isatty(self):
        return False


# --- detection order ------------------------------------------------------


def test_no_color_env_disables(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled(_Tty()) is False


def test_no_color_beats_force_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert color_enabled(_Tty()) is False, "NO_COLOR must win (no-color.org)"


def test_force_color_enables_when_piped(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert color_enabled(_Pipe()) is True


def test_empty_no_color_is_ignored(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert color_enabled(_Pipe()) is False   # still off: not a tty
    assert color_enabled(_Tty()) is True     # but NO_COLOR="" does not force off


def test_pipe_disables_by_default(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert color_enabled(_Pipe()) is False


def test_override_beats_everything(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    set_color_override(True)
    assert color_enabled(_Pipe()) is True
    set_color_override(False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color_enabled(_Tty()) is False


def test_stream_without_isatty_is_safe(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    class Odd:
        pass

    assert color_enabled(Odd()) is False, "must not raise on an exotic stream"


# --- Windows ---------------------------------------------------------------


def test_windows_console_without_vt_gets_no_color(monkeypatch):
    """The case from the issue: isatty() True but ANSI not interpreted."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(report.sys, "platform", "win32")
    monkeypatch.setattr(report, "_enable_windows_vt", lambda stream: False)
    assert color_enabled(_Tty()) is False


def test_windows_console_with_vt_gets_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(report.sys, "platform", "win32")
    monkeypatch.setattr(report, "_enable_windows_vt", lambda stream: True)
    assert color_enabled(_Tty()) is True


def test_enable_windows_vt_never_raises_off_windows():
    """It is only called on win32, but must be safe if reached elsewhere."""
    assert report._enable_windows_vt(_Tty()) is False


# --- rendering -------------------------------------------------------------


def test_render_emits_no_escapes_when_disabled():
    set_color_override(False)
    assert ESC not in render_function(_score())
    assert ESC not in render_summary([_score(), _score(False)])


def test_render_emits_escapes_when_forced():
    set_color_override(True)
    assert ESC in render_summary([_score()])


def test_explicit_color_arg_still_wins():
    set_color_override(True)
    assert ESC not in render_summary([_score()], color=False)


def test_markers_survive_without_color():
    """✓ / ✗ must still be distinguishable in plain text."""
    set_color_override(False)
    text = render_summary([_score(True), _score(False)])
    assert "✓" in text and "✗" in text


# --- end to end ------------------------------------------------------------


@pytest.fixture
def dump(tmp_path, repo_root):
    from bench.extract import extract

    t = next(x for x in extract(repo_root / "fixtures" / "http_server.py")
             if x.name == "is_cgi")
    p = tmp_path / "d.json"
    write_text(p, json.dumps({
        "files": [str(repo_root / "fixtures" / "http_server.py")],
        "model": "t",
        "results": [{"function": "is_cgi", "response": "\n".join(t.primary_lines)}],
    }))
    return p


def _rescore(python_bin, repo_root, dump, *extra, env=None):
    return subprocess.run(
        [python_bin, str(repo_root / "bench.py"), *extra, "rescore", str(dump),
         "--file", str(repo_root / "fixtures" / "http_server.py")],
        capture_output=True, text=True, cwd=repo_root, timeout=180, env=env,
    )


def test_cli_no_color_flag(python_bin, repo_root, dump):
    r = _rescore(python_bin, repo_root, dump, "--no-color")
    assert r.returncode == 0, r.stderr[-500:]
    assert ESC not in r.stdout


def test_cli_color_flag_forces_escapes_through_a_pipe(python_bin, repo_root, dump):
    r = _rescore(python_bin, repo_root, dump, "--color")
    assert r.returncode == 0, r.stderr[-500:]
    assert ESC in r.stdout, "--color must win over the pipe heuristic"


def test_cli_no_color_env(python_bin, repo_root, dump):
    import os

    env = dict(os.environ, NO_COLOR="1", FORCE_COLOR="1")
    r = _rescore(python_bin, repo_root, dump, env=env)
    assert r.returncode == 0, r.stderr[-500:]
    assert ESC not in r.stdout


def test_piped_output_is_clean_by_default(python_bin, repo_root, dump):
    """What users copy out of a redirect must not contain escape codes."""
    import os

    env = {k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR",)}
    r = _rescore(python_bin, repo_root, dump, env=env)
    assert r.returncode == 0, r.stderr[-500:]
    assert ESC not in r.stdout
    assert "✓" in r.stdout, "the result itself must still be readable"


def test_flags_are_mutually_exclusive(python_bin, repo_root, dump):
    r = _rescore(python_bin, repo_root, dump, "--no-color", "--color")
    assert r.returncode != 0
    assert "not allowed with" in (r.stderr + r.stdout)
