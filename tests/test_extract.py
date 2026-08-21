"""Extractor: ground truth, line classification, multi-file offsets."""
from __future__ import annotations

import textwrap

from bench.extract import (
    KIND_BLANK, KIND_CODE, KIND_COMMENT, KIND_DOCSTRING,
    extract, line_kinds, load_source_glob, stratified_sample,
)


# --- ground truth ---------------------------------------------------------


def test_python_targets_match_real_source(py_source):
    """Every extracted body line must equal the actual file line at that number."""
    text_lines = py_source.text.splitlines()
    assert py_source.targets, "no targets extracted"
    for t in py_source.targets:
        for offset, line in enumerate(t.body_lines):
            assert text_lines[t.start_line - 1 + offset] == line, (
                f"{t.name}: body line {offset} does not match combined text at "
                f"line {t.start_line + offset}"
            )


def test_js_targets_match_real_source(js_source):
    text_lines = js_source.text.splitlines()
    assert js_source.targets
    for t in js_source.targets:
        for offset, line in enumerate(t.body_lines):
            assert text_lines[t.start_line - 1 + offset] == line, (
                f"{t.name}: body line {offset} mismatch at line {t.start_line + offset}"
            )


def test_body_excludes_signature_and_closing_brace(js_source):
    """The first body line must not be the signature; that's the whole probe."""
    for t in js_source.targets:
        first = t.body_lines[0].strip()
        assert not first.startswith("function "), f"{t.name} body starts at signature"


def test_kinds_are_parallel_to_lines(py_source, js_source):
    for src in (py_source, js_source):
        for t in src.targets:
            assert len(t.primary_kinds) == len(t.primary_lines)
            assert len(t.bonus_kinds) == len(t.bonus_lines)


# --- line classification --------------------------------------------------


def test_python_line_kinds():
    src = (
        'def f():\n'
        '    """Doc one.\n'
        '\n'
        '    Doc two.\n'
        '    """\n'
        '    # comment\n'
        '    x = 1  # trailing comment\n'
        '\n'
        '    return x\n'
    )
    assert line_kinds(src, "py") == [
        KIND_CODE,        # def f():
        KIND_DOCSTRING,   # """Doc one.
        KIND_BLANK,       # blank inside docstring
        KIND_DOCSTRING,   # Doc two.
        KIND_DOCSTRING,   # """
        KIND_COMMENT,     # # comment
        KIND_CODE,        # x = 1  # trailing -> still code
        KIND_BLANK,
        KIND_CODE,        # return x
    ]


def test_python_multiline_string_is_not_a_docstring():
    """Only the FIRST statement is a docstring; other strings are code."""
    src = (
        'def f():\n'
        '    x = """not\n'
        '    a docstring"""\n'
        '    return x\n'
    )
    assert line_kinds(src, "py") == [KIND_CODE] * 4


def test_javascript_line_kinds():
    src = (
        "function g() {\n"
        "\t// line comment\n"
        "\t/* block start\n"
        "\t   block middle\n"
        "\t   block end */\n"
        "\tvar y = 2; // trailing\n"
        "\n"
        "\treturn y;\n"
        "}\n"
    )
    k = line_kinds(src, "js")
    assert k[1] == KIND_COMMENT, "line comment"
    assert k[2] == KIND_COMMENT, "block opening line"
    assert k[3] == KIND_COMMENT, "block interior"
    assert k[4] == KIND_COMMENT, "block closing line (no leading asterisk)"
    assert k[5] == KIND_CODE, "trailing comment must not blank out the code"
    assert k[6] == KIND_BLANK
    assert k[7] == KIND_CODE


def test_javascript_code_before_block_comment_stays_code():
    src = "function g() {\n\tvar a = 1; /* tail\n\t comment */\n\treturn a;\n}\n"
    k = line_kinds(src, "js")
    assert k[1] == KIND_CODE, "code precedes the comment on this line"
    assert k[2] == KIND_COMMENT, "closing line is all comment"


def test_jsdoc_asterisk_block_classified():
    src = "function g() {\n\t/**\n\t * docs here\n\t */\n\tvar a = 1;\n\treturn a;\n}\n"
    k = line_kinds(src, "js")
    assert k[1] == k[2] == k[3] == KIND_COMMENT
    assert k[4] == KIND_CODE


def test_unparseable_source_degrades_gracefully():
    """A syntax error must not crash classification."""
    assert line_kinds("def broken(:\n  x=1\n", "py")[0] == KIND_CODE
    assert line_kinds("function ( { [ bad", "js")


# --- multi-file corpora ---------------------------------------------------


def _fn(name: str, n: int = 25) -> str:
    body = "\n".join(f"    v_{i} = {i}" for i in range(n))
    return f"def {name}(a):\n{body}\n    return a\n"


def test_multifile_offsets_land_on_real_lines(tmp_path):
    """Both shipped corpora are single-file, so this path is otherwise untested."""
    (tmp_path / "a_mod.py").write_text(_fn("alpha") + "\n" + _fn("beta"))
    (tmp_path / "b_mod.py").write_text(_fn("gamma"))

    src = load_source_glob(tmp_path, "*.py", None)
    names = {t.name for t in src.targets}
    assert {"alpha", "beta", "gamma"} <= names

    combined = src.text.splitlines()
    for t in src.targets:
        for offset, line in enumerate(t.body_lines):
            assert combined[t.start_line - 1 + offset] == line, (
                f"{t.name}: offset wrong at combined line {t.start_line + offset}"
            )


def test_multifile_headers_present(tmp_path):
    (tmp_path / "a_mod.py").write_text(_fn("alpha"))
    (tmp_path / "b_mod.py").write_text(_fn("gamma"))
    src = load_source_glob(tmp_path, "*.py", None)
    assert src.text.count("# ====== ") == 2, "one header per file"


def test_multifile_name_collision_deduped(tmp_path):
    (tmp_path / "a_mod.py").write_text(_fn("same"))
    (tmp_path / "b_mod.py").write_text(_fn("same"))
    src = load_source_glob(tmp_path, "*.py", None)
    assert [t.name for t in src.targets] == ["same"], "first occurrence wins"


def test_mixed_languages_rejected(tmp_path):
    import pytest

    (tmp_path / "a.py").write_text(_fn("alpha"))
    (tmp_path / "b.js").write_text("function g(){\n" + "\n".join(
        f"  var v{i} = {i};" for i in range(25)) + "\n}\n")
    with pytest.raises(ValueError, match="mixed languages"):
        load_source_glob(tmp_path, "*.*", None)


# --- sampling -------------------------------------------------------------


def test_stratified_sample_is_deterministic(js_source):
    total = js_source.text.count("\n") + 1
    a = stratified_sample(js_source.targets, total, k=16, seed=42)
    b = stratified_sample(js_source.targets, total, k=16, seed=42)
    assert [t.name for t in a] == [t.name for t in b]


def test_stratified_sample_spans_the_file(js_source):
    """The point of the benchmark is depth coverage, not just the head."""
    total = js_source.text.count("\n") + 1
    picked = stratified_sample(js_source.targets, total, k=16, seed=42)
    lines = [t.start_line for t in picked]
    assert min(lines) < total * 0.2, "no target near the top"
    assert max(lines) > total * 0.8, "no target near the bottom"


def test_code_line_count_flags_prose_targets(py_source):
    by_name = {t.name: t for t in py_source.targets}
    # Verified by hand against the fixture: this window is almost all docstring.
    assert by_name["log_message"].code_line_count == 1
