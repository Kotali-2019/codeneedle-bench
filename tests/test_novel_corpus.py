"""Regression coverage for the deterministic, model-unseen benchmark corpora."""
from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest

from bench.config import load_corpus
from bench.extract import configure_primary_window, load_source_glob
from bench.textio import read_text


@pytest.fixture(scope="module")
def generator(repo_root):
    spec = importlib.util.spec_from_file_location(
        "generate_novel_corpora", repo_root / "tools" / "generate_novel_corpora.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_novel_fixtures_match_generator(generator, repo_root):
    assert generator.check_generated(repo_root / "fixtures" / "novel")


def test_generator_is_deterministic(generator):
    assert generator.generated_files() == generator.generated_files()


@pytest.mark.parametrize("name", ["novel_16k", "novel_64k", "novel_128k"])
def test_novel_corpus_is_valid_stratified_and_auditable(name, repo_root):
    manifest = json.loads(read_text(repo_root / "fixtures" / "novel" / "manifest.json"))
    expected = manifest["corpora"][name]
    corpus = load_corpus(name)
    source = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
    configure_primary_window(source, corpus.primary_lines)
    source_file = source.files[0]
    target_by_name = {target.name: target for target in source.targets}

    assert len(source.targets) == expected["functions"]
    assert hashlib.sha256(read_text(source_file).encode("utf-8")).hexdigest() == expected["sha256"]
    assert all(target.code_line_count >= corpus.min_code_lines for target in source.targets)
    assert len({target.name for target in source.targets}) == len(source.targets)
    assert len({tuple(target.primary_lines) for target in source.targets}) == len(source.targets)

    assert corpus.sample_functions == manifest["paired_targets"]
    assert corpus.primary_lines == manifest["primary_lines"] == 48
    total_lines = source.text.count("\n") + 1
    sampled = [target_by_name[name] for name in corpus.sample_functions]
    fractions = [target.start_line / total_lines for target in sampled]
    assert len(sampled) == 18
    assert min(fractions) < 0.05
    assert max(fractions) > 0.95


def test_novel_sizes_use_identical_paired_target_bodies():
    bodies_by_size = []
    for name in ("novel_16k", "novel_64k", "novel_128k"):
        corpus = load_corpus(name)
        source = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
        configure_primary_window(source, corpus.primary_lines)
        target_by_name = {target.name: target for target in source.targets}
        bodies_by_size.append({
            target_name: tuple(target_by_name[target_name].primary_lines)
            for target_name in corpus.sample_functions
        })

    assert bodies_by_size[0] == bodies_by_size[1] == bodies_by_size[2]


def test_novel_targets_span_three_seeds_and_have_neighboring_decoys(repo_root):
    manifest = json.loads(read_text(repo_root / "fixtures" / "novel" / "manifest.json"))
    seed_counts = {
        seed: list(manifest["target_seed_by_name"].values()).count(seed)
        for seed in manifest["target_seeds"]
    }
    assert seed_counts == {20260901: 6, 20260917: 6, 20261003: 6}

    source = read_text(repo_root / "fixtures" / "novel" / "novel_16k.py")
    for target_name in manifest["paired_targets"]:
        shadow = f"def {target_name}_shadow(arg_a, arg_b, arg_c):"
        target = f"def {target_name}(arg_a, arg_b, arg_c):"
        assert shadow in source
        assert 0 < source.index(target) - source.index(shadow) < 1500
