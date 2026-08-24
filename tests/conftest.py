"""Shared fixtures. Puts the repo root on sys.path so `import bench` works."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def js_source():
    """The real jquery.js corpus, parsed once."""
    from bench.config import load_corpus
    from bench.extract import load_source_glob

    c = load_corpus("jquery")
    return load_source_glob(c.directory, c.glob, c.limit)
