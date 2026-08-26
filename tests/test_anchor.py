"""The prompt must anchor on text the corpus actually contains — issue #4.

The old wording told the model to look for `function <name>(`. Five of the 16
sampled jQuery targets are property- or assignment-style, so that string does
not occur in the file at all. Those five averaged 59% against 79% for the rest
across every stored run — a prompt defect being measured as model failure.
"""
from __future__ import annotations

import pytest

from bench.extract import load_source_glob, stratified_sample
from bench.runner import _build_prompt, _signature_block
from bench.textio import write_text


def _sample(source, k=16, seed=42):
    pool = [t for t in source.targets if not t.ambiguous]
    return stratified_sample(pool, source.text.count("\n") + 1, k=k, seed=seed)


# --- the anchor is real text ---------------------------------------------


def test_every_sampled_js_anchor_occurs_exactly_once(js_source):
    for t in _sample(js_source):
        assert t.signature_text, f"{t.name}: no signature captured"
        n = js_source.text.count(t.signature_text)
        assert n == 1, f"{t.name}: signature occurs {n}x, must be unique"


def test_every_sampled_py_anchor_occurs_exactly_once(py_source):
    for t in _sample(py_source, k=16):
        assert t.signature_text
        assert py_source.text.count(t.signature_text) == 1, t.name


@pytest.mark.parametrize("name", ["PSEUDO", "val", "then", "init", "parseHTML"])
def test_property_and_assignment_styles_get_a_real_anchor(js_source, name):
    """The five targets the old `function <name>(` guess never matched."""
    t = next(x for x in js_source.targets if x.name == name)
    assert f"function {name}(" not in js_source.text, "premise: old anchor was fictional"
    assert t.signature_text in js_source.text, "new anchor must be real text"


def test_prompt_quotes_the_signature(js_source):
    t = next(x for x in js_source.targets if x.name == "val")
    prompt = _build_prompt(t, js_source.text, False, False)
    assert "val: function( value ) {" in prompt
    assert "function val(" not in prompt, "must not invent a signature"


def test_prompt_body_starts_immediately_after_the_quoted_signature(js_source):
    """The contract the prompt states must hold in the source."""
    for t in _sample(js_source):
        lines = js_source.text.splitlines()
        sig = t.signature_text.splitlines()
        end = t.start_line - 1                 # 0-indexed line after signature
        assert lines[end - len(sig):end] == sig, t.name
        assert lines[end] == t.body_lines[0], t.name


def test_multiline_signature_is_quoted_whole(js_source):
    """A signature split across lines must not leave a fragment as line one."""
    multi = [t for t in js_source.targets
             if t.signature_text and "\n" in t.signature_text]
    if not multi:
        pytest.skip("corpus has no multi-line signatures")
    for t in multi[:5]:
        lines = js_source.text.splitlines()
        sig = t.signature_text.splitlines()
        assert lines[t.start_line - 1 - len(sig):t.start_line - 1] == sig
        assert not t.body_lines[0].strip().endswith("{"), \
            f"{t.name}: body should not start with signature remnant"


def test_python_signature_excludes_decorators(tmp_path):
    from bench.extract import extract

    body = "\n".join(f"    v{i} = {i}" for i in range(25))
    write_text(tmp_path / "m.py",
               f"import functools\n\n@functools.cache\ndef decorated(a):\n{body}\n    return a\n")
    t = next(x for x in extract(tmp_path / "m.py") if x.name == "decorated")
    assert t.signature_text == "def decorated(a):"
    assert "@functools.cache" not in t.signature_text


def test_python_multiline_signature(tmp_path):
    from bench.extract import extract

    body = "\n".join(f"    v{i} = {i}" for i in range(25))
    write_text(tmp_path / "m.py", f"def wide(a,\n         b):\n{body}\n    return a\n")
    t = next(x for x in extract(tmp_path / "m.py") if x.name == "wide")
    assert t.signature_text == "def wide(a,\n         b):"
    assert t.body_lines[0] == "    v0 = 0"


# --- ambiguity ------------------------------------------------------------


def test_duplicate_names_with_distinct_signatures_are_kept(js_source):
    """Quoting the signature disambiguates 10 of jQuery's 11 duplicated names."""
    for name in ("PSEUDO", "find", "CHILD", "stop"):
        t = next(x for x in js_source.targets if x.name == name)
        assert not t.ambiguous, f"{name} is distinguishable by its signature"


def test_identical_signatures_are_flagged_ambiguous(js_source, py_source):
    """`get` appears twice in jQuery as `get: function( elem ) {`."""
    g = next((x for x in js_source.targets if x.name == "get"), None)
    if g is not None:
        assert g.ambiguous

    # http_server declares send_head twice with the same signature; only one
    # body is long enough to extract, so eligibility alone would miss it.
    sh = next((x for x in py_source.targets if x.name == "send_head"), None)
    if sh is not None:
        assert sh.ambiguous, "whole-file ambiguity, not just among targets"


def test_ambiguous_targets_are_excluded_from_sampling(js_source, py_source):
    for src in (js_source, py_source):
        assert not any(t.ambiguous for t in _sample(src)), \
            "an unanswerable question must never be asked"


def test_block_occurrences_counts_whole_lines():
    from bench.extract import _block_occurrences

    lines = ["def f():", "    pass", "def f():", "    pass", "xdef f():"]
    assert _block_occurrences(lines, "def f():") == 2, "substring match must not count"
    assert _block_occurrences(lines, "def f():\n    pass") == 2
    assert _block_occurrences(lines, "nope") == 0
    assert _block_occurrences(lines, "") == 0


# --- fallback -------------------------------------------------------------


def test_missing_signature_falls_back_to_prose():
    """A hand-built target without a captured signature must still prompt."""
    from bench.extract import FunctionTarget

    t = FunctionTarget(name="thing", start_line=1, body_lines=["x"] * 20,
                       language="js")
    block = _signature_block(t)
    assert "function thing(" in block, "falls back to the old description"

    t.language = "py"
    assert "def thing(" in _signature_block(t)


def test_prompt_keeps_corpus_first_for_kv_cache_reuse(js_source):
    """The file must stay at the front or prefix caching breaks."""
    t = next(x for x in js_source.targets if x.name == "val")
    prompt = _build_prompt(t, js_source.text, False, False)
    assert prompt.startswith(js_source.text)
    assert prompt.index("Task:") > len(js_source.text)
