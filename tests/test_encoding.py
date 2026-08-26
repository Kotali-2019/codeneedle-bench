"""UTF-8 handling — regression cover for issue #10.

Windows defaults `open()`/`read_text()`/`write_text()` and redirected stdout to
the locale encoding (cp1252), which cannot represent `←`, `⚠`, `≥` or the CJK
characters inside the bundled plotly.min.js. Every check here fails on the
pre-fix code.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.textio import ENCODING, read_text, use_utf8_stdio, write_text

# Characters the project emits that cp1252 cannot encode.
UNMAPPABLE = "←→≥⚠✅✓✗❌"


def test_cp1252_really_cannot_encode_our_characters():
    """Pin the premise — if this ever passes, the rest is moot."""
    for ch in UNMAPPABLE:
        with pytest.raises(UnicodeEncodeError):
            ch.encode("cp1252")


# --- helpers --------------------------------------------------------------


def test_write_then_read_roundtrips_unicode(tmp_path):
    p = tmp_path / "x.txt"
    payload = "← all corpora · ⚠ INCOMPLETE ≥20 — 闰月"
    write_text(p, payload)
    assert read_text(p) == payload
    assert p.read_bytes().decode(ENCODING) == payload


def test_helpers_ignore_the_ambient_locale(tmp_path, monkeypatch):
    """Even with a hostile locale, the helpers must use UTF-8."""
    monkeypatch.setattr("locale.getpreferredencoding", lambda *a, **k: "cp1252")
    p = tmp_path / "x.txt"
    write_text(p, "←")
    assert read_text(p) == "←"


def test_helpers_accept_str_paths(tmp_path):
    p = str(tmp_path / "x.txt")
    write_text(p, "⚠")
    assert read_text(p) == "⚠"


def test_use_utf8_stdio_is_idempotent_and_safe():
    use_utf8_stdio()
    use_utf8_stdio()
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"


def test_use_utf8_stdio_survives_missing_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    use_utf8_stdio()  # must not raise


def test_use_utf8_stdio_survives_streams_without_reconfigure(monkeypatch):
    class Dummy:
        encoding = "cp1252"

    monkeypatch.setattr(sys, "stdout", Dummy())
    use_utf8_stdio()  # must not raise


# --- source-level guard ---------------------------------------------------


SOURCE_FILES = [
    "bench.py", "run-missing.py", "smoke_test.py", "analysis/visualize.py",
    "bench/extract.py", "bench/config.py", "bench/runner.py",
    "bench/report.py", "bench/scorer.py", "bench/client.py", "bench/textio.py",
]


@pytest.mark.parametrize("rel", SOURCE_FILES)
def test_no_encoding_blind_file_io(repo_root, rel):
    """No bare read_text()/write_text()/open() anywhere in shipped source.

    Walks the AST rather than grepping so a reformat can't sneak one past.
    """
    path = repo_root / rel
    tree = ast.parse(read_text(path))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", None) or getattr(fn, "id", None)
        if name not in ("read_text", "write_text", "open"):
            continue
        # Our own helpers take the path positionally and handle encoding.
        if isinstance(fn, ast.Name) and name in ("read_text", "write_text"):
            continue
        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        # Binary modes carry no encoding.
        if name == "open" and len(node.args) > 1:
            mode = node.args[1]
            if isinstance(mode, ast.Constant) and "b" in str(mode.value):
                continue
        offenders.append(f"{rel}:{node.lineno} {name}()")
    assert not offenders, (
        "locale-dependent file I/O (breaks on Windows cp1252): "
        + ", ".join(offenders)
    )


# --- end-to-end under a hostile encoding ----------------------------------


def _cp1252_env():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"   # what Windows does on redirect
    env.pop("PYTHONUTF8", None)
    return env


def test_extract_survives_redirected_cp1252_stdout(python_bin, repo_root, tmp_path):
    """`python bench.py extract --corpus X > out.log` on a cp1252 machine."""
    out = tmp_path / "out.log"
    with out.open("wb") as fh:
        r = subprocess.run(
            [python_bin, str(repo_root / "bench.py"), "extract",
             "--corpus", "http_server"],
            stdout=fh, stderr=subprocess.PIPE, cwd=repo_root,
            env=_cp1252_env(), timeout=180,
        )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-800:]
    text = out.read_bytes().decode("utf-8")
    assert "≥20 body lines" in text, "non-ASCII output must survive redirection"
    assert "⚠" in text, "the prose-target warning must survive too"


def test_rescore_summary_markers_survive_cp1252(python_bin, repo_root, tmp_path):
    """The per-function summary prints ✓/✗, which cp1252 cannot encode.

    Driven through `bench.py rescore` — a real entry point that needs no model,
    only a prior dump.
    """
    from bench.extract import extract

    target = next(t for t in extract(repo_root / "fixtures" / "http_server.py")
                  if t.name == "is_cgi")
    dump = tmp_path / "dump.json"
    write_text(dump, json.dumps({
        "files": [str(repo_root / "fixtures" / "http_server.py")],
        "model": "test",
        "results": [
            {"function": "is_cgi", "response": "\n".join(target.primary_lines)},
            {"function": "translate_path", "response": "nothing like the truth"},
        ],
    }))

    out = tmp_path / "summary.log"
    with out.open("wb") as fh:
        r = subprocess.run(
            [python_bin, str(repo_root / "bench.py"), "rescore", str(dump),
             "--file", str(repo_root / "fixtures" / "http_server.py")],
            stdout=fh, stderr=subprocess.PIPE, cwd=repo_root,
            env=_cp1252_env(), timeout=180,
        )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-800:]
    text = out.read_bytes().decode("utf-8")
    assert "✓" in text and "✗" in text, "pass/fail markers must survive redirection"


def test_visualize_survives_cp1252(python_bin, repo_root, tmp_path):
    """The exact command from issue #10.

    Supplies its own `--results-dir`: `results/*.json` is gitignored, so a
    fresh clone has none and visualize.py would correctly report nothing to
    do. Relying on whatever the developer happens to have run locally would
    make this pass or fail by accident.
    """
    results = tmp_path / "results"
    results.mkdir()
    write_text(results / "jquery__demo.json", json.dumps({
        "files": [str(repo_root / "fixtures" / "jquery.js")],
        "model": "demo",
        "results": [
            {"function": f"fn{i}", "passed": True, "error": None,
             "primary_matched": 18, "primary_total": 20,
             "hallucinated": 0, "bonus_matched": 0}
            for i in range(3)
        ],
    }))

    r = subprocess.run(
        [python_bin, str(repo_root / "analysis" / "visualize.py"),
         "--results-dir", str(results),
         "--output-dir", str(tmp_path / "charts")],
        capture_output=True, cwd=repo_root, env=_cp1252_env(), timeout=300,
    )
    err = r.stderr.decode("utf-8", "replace")
    assert "UnicodeEncodeError" not in err, err[-800:]
    assert r.returncode == 0, err[-800:]

    pages = list((tmp_path / "charts").rglob("*.html"))
    assert pages, "no charts written"
    for p in pages:
        p.read_bytes().decode("utf-8")   # must be valid UTF-8
    corpus_index = next(p for p in pages if p.parent.name != "charts")
    assert "← all corpora" in corpus_index.read_text(encoding="utf-8")


def test_bundled_plotly_js_written_intact(python_bin, repo_root, tmp_path):
    """plotly.min.js contains CJK; writing it under cp1252 used to abort.

    Specific to this branch — it is what bundles plotly for offline charts.
    """
    subprocess.run(
        [python_bin, str(repo_root / "analysis" / "visualize.py"),
         "--output-dir", str(tmp_path / "charts")],
        capture_output=True, cwd=repo_root, env=_cp1252_env(), timeout=300,
        check=True,
    )
    from plotly.offline import get_plotlyjs

    written = next((tmp_path / "charts").rglob("plotly.min.js"))
    assert read_text(written) == get_plotlyjs()


def test_run_missing_survives_cp1252(python_bin, repo_root, tmp_path):
    out = tmp_path / "out.log"
    with out.open("wb") as fh:
        r = subprocess.run(
            [python_bin, str(repo_root / "run-missing.py"), "--dry-run"],
            stdout=fh, stderr=subprocess.PIPE, cwd=repo_root,
            env=_cp1252_env(), timeout=180,
        )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-500:]
    out.read_bytes().decode("utf-8")


# --- non-ASCII corpora ----------------------------------------------------


def test_corpus_with_non_ascii_source_is_read_correctly(tmp_path):
    """A user corpus containing non-ASCII must not depend on the locale."""
    from bench.extract import load_source_glob

    body = "\n".join(f"    v_{i} = {i}  # café — naïve ✓" for i in range(25))
    src = f'def accented(a):\n    """Docstring with émoji ⚠ and 日本語."""\n{body}\n    return a\n'
    write_text(tmp_path / "mod.py", src)

    source = load_source_glob(tmp_path, "*.py", None)
    t = next(x for x in source.targets if x.name == "accented")
    assert "émoji ⚠ and 日本語" in t.primary_lines[0]
    assert "café" in source.text


def test_dump_with_non_ascii_response_roundtrips(tmp_path):
    """Model output is arbitrary text; the dump must survive it."""
    payload = {"results": [{"function": "f", "response": "héllo ⚠ 日本語 ←"}]}
    p = tmp_path / "d.json"
    write_text(p, json.dumps(payload, indent=2))
    assert json.loads(read_text(p))["results"][0]["response"] == "héllo ⚠ 日本語 ←"


def test_model_config_with_non_ascii_content(tmp_path):
    """TOML is UTF-8 by spec; reading it at the locale encoding corrupts it."""
    from bench.config import load_model_from_file

    write_text(tmp_path / "m.toml",
               '# Qwen3.6 · 27B — bf16 ✓\nname = "qwen3.6-27b"\n'
               'api_key = "clé-secrète"\n')
    cfg = load_model_from_file(tmp_path / "m.toml")
    assert cfg.client.model == "qwen3.6-27b"
    assert cfg.client.api_key == "clé-secrète"


def test_api_key_file_with_trailing_newline_and_unicode(tmp_path, monkeypatch):
    """Keys are read from disk; that read must not depend on the locale."""
    import bench.config as config

    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    write_text(tmp_path / "k.key", "sk-café-123\n")
    write_text(tmp_path / "m.toml",
               'name = "x"\napi_key_file = "k.key"\n')
    cfg = config.load_model_from_file(tmp_path / "m.toml")
    assert cfg.client.api_key == "sk-café-123"


def test_model_config_with_non_ascii_label(tmp_path):
    """The `label` field is displayed in charts, so it must round-trip."""
    from bench.config import load_model_from_file

    write_text(tmp_path / "m.toml",
               'name = "x"\nlabel = "Qwen3.6 · 27B — MLX ✓"\n')
    cfg = load_model_from_file(tmp_path / "m.toml")
    assert cfg.label == "Qwen3.6 · 27B — MLX ✓"
