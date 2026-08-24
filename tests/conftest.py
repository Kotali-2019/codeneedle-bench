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


@pytest.fixture
def python_bin(repo_root: Path) -> str:
    venv = repo_root / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable
