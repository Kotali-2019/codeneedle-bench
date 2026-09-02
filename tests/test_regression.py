"""Findings pinned against the real fixtures and stored results.

These assert the specific facts that motivated the scoring policy, so a future
change that quietly reintroduces the inflation fails loudly.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from bench.extract import KIND_BLANK, KIND_CODE, PROSE_KINDS
from bench.scorer import score
from bench.textio import read_text, write_text


# --- corpus composition ---------------------------------------------------


def test_http_server_is_mostly_non_code(py_source):
    kinds = [k for t in py_source.targets for k in t.primary_kinds]
    n = len(kinds)
    code = sum(1 for k in kinds if k == KIND_CODE)
    assert n == 220, "11 functions x 20 expected lines"
    assert code == 88
    assert code / n == pytest.approx(0.40, abs=0.01), \
        "http_server is 40% code — the finding that motivated this policy"


def test_jquery_composition(js_source):
    kinds = [k for t in js_source.targets for k in t.primary_kinds]
    n = len(kinds)
    blank = sum(1 for k in kinds if k == KIND_BLANK)
    assert blank / n == pytest.approx(0.19, abs=0.01), "~19% blank"


def test_blank_share_is_material_in_both_corpora(py_source, js_source):
    for src in (py_source, js_source):
        kinds = [k for t in src.targets for k in t.primary_kinds]
        blank = sum(1 for k in kinds if k == KIND_BLANK)
        assert blank / len(kinds) > 0.15, "blank lines were a large free-credit pool"


def test_log_message_has_a_single_code_line(py_source):
    t = next(x for x in py_source.targets if x.name == "log_message")
    assert t.code_line_count == 1
    assert sum(1 for k in t.primary_kinds if k == KIND_BLANK) == 6
    assert sum(1 for k in t.primary_kinds if k in PROSE_KINDS) == 13


def test_blank_only_credit_cannot_reach_the_threshold_anywhere(py_source, js_source):
    """Even at its worst, whitespace alone must never pass a function."""
    for src in (py_source, js_source):
        for t in src.targets:
            sc = score(t.name, t.primary_lines, t.bonus_lines,
                       "\n".join("" for _ in t.primary_lines),
                       primary_kinds=t.primary_kinds, bonus_kinds=t.bonus_kinds)
            assert sc.primary_matched == 0, f"{t.name} scored on blanks alone"
            assert not sc.passed


# --- stored results -------------------------------------------------------


def _stored(repo_root):
    files = sorted((repo_root / "results").glob("*.json"))
    if not files:
        pytest.skip("no stored results to check against")
    return files


def test_stored_results_contained_real_blank_inflation(repo_root, py_source,
                                                       js_source):
    """~22% of the old matched-line total across every shipped run was blanks."""
    by_corpus = {
        "http_server": {t.name: t for t in py_source.targets},
        "jquery": {t.name: t for t in js_source.targets},
    }
    old = new = 0
    for p in _stored(repo_root):
        d = json.loads(read_text(p))
        corpus = pathlib.Path(p).stem.split("__")[0]
        if corpus not in by_corpus:
            continue
        relax = bool((d.get("scoring") or {}).get(
            "relax_indent", d.get("relax_indent", False)))
        for r in d["results"]:
            t = by_corpus[corpus].get(r["function"])
            if t is None or r.get("error"):
                continue
            sc = score(t.name, t.primary_lines, t.bonus_lines,
                       r.get("response", ""), relax_indent=relax,
                       primary_kinds=t.primary_kinds, bonus_kinds=t.bonus_kinds)
            old += r["primary_matched"]
            new += sc.primary_matched
    assert old > 0
    removed = (old - new) / old
    assert removed > 0.15, (
        f"expected substantial blank-line credit in the stored runs, got {removed:.1%}"
    )


def test_rescoring_is_deterministic(repo_root, py_source):
    """Same dump, same policy, same numbers — twice."""
    p = repo_root / "results" / "http_server__gpt-5.5.json"
    if not p.is_file():
        pytest.skip("reference result not present")
    by = {t.name: t for t in py_source.targets}
    runs = []
    for _ in range(2):
        total = 0
        for r in json.loads(read_text(p))["results"]:
            t = by.get(r["function"])
            if t is None:
                continue
            sc = score(t.name, t.primary_lines, t.bonus_lines,
                       r.get("response", ""), primary_kinds=t.primary_kinds,
                       bonus_kinds=t.bonus_kinds)
            total += sc.primary_matched
        runs.append(total)
    assert runs[0] == runs[1]


# --- config integrity -----------------------------------------------------


def test_every_model_config_loads(repo_root):
    from bench.config import load_model_from_file

    configs = sorted((repo_root / "configs" / "models").glob("*.toml"))
    assert configs
    for c in configs:
        if "api_key_file" in read_text(c) and not (
            repo_root / ".secrets"
        ).exists():
            continue  # hosted config needs a key file we may not have
        cfg = load_model_from_file(c)
        assert cfg.client.model, f"{c.name}: empty model id"


def test_quant_pair_configs_are_distinctly_labelled(repo_root):
    """The MLX 4bit/8bit pair must not be ambiguous in charts."""
    import tomllib

    d = repo_root / "configs" / "models"
    four = tomllib.loads(read_text(d / "qwen36-27b-mlx-4bit.toml"))
    eight = tomllib.loads(read_text(d / "qwen36-27b-mlx-8bit.toml"))
    assert four["label"] != eight["label"]
    assert "4bit" in four["label"] and "8bit" in eight["label"]
    # Server-side ids stay whatever LM Studio registered.
    assert four["name"] == "qwen3.6-27b"
    assert eight["name"] == "qwen3.6-27b-mlx"


def test_every_corpus_config_loads(repo_root):
    from bench.config import load_corpus

    for c in sorted((repo_root / "configs" / "corpora").glob("*.toml")):
        cfg = load_corpus(c.stem)
        assert cfg.directory.is_dir(), f"{c.name}: directory missing"
        assert cfg.sample_k > 0
        assert cfg.primary_lines >= 20
        assert cfg.relax_indent is False, (
            f"{c.name}: shipped comparison corpora must score every model strictly"
        )


def test_model_config_cannot_set_scoring_policy(tmp_path):
    from bench.config import load_model_from_file

    p = tmp_path / "model.toml"
    write_text(p, 'name = "mock"\nrelax_indent = true\n')
    with pytest.raises(ValueError, match="scoring policy.*corpus config"):
        load_model_from_file(p)


def test_corpus_primary_window_cannot_be_smaller_than_extractor_floor(tmp_path):
    from bench.config import load_corpus

    p = tmp_path / "bad-window.toml"
    write_text(p, '''
[files]
directory = "fixtures"
glob = "*.py"
[sample]
primary_lines = 19
''')
    with pytest.raises(ValueError, match="primary_lines must be at least 20"):
        load_corpus(p)
