"""CLI-level behavior, exercised via subprocess so argparse and exit codes count."""
from __future__ import annotations

import json
import subprocess

import pytest


def run_cli(python_bin, repo_root, *args, cwd=None):
    return subprocess.run(
        [python_bin, str(repo_root / "bench.py"), *args],
        capture_output=True, text=True, cwd=cwd or repo_root, timeout=180,
    )


# --- argument handling ----------------------------------------------------


def test_missing_source_is_a_clean_error(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract")
    assert r.returncode != 0
    assert "pass either --corpus" in (r.stderr + r.stdout)


def test_unknown_function_exits_nonzero_not_traceback(python_bin, repo_root):
    """Regression: this used to IndexError, and exit 0 with --skip-preflight."""
    r = run_cli(python_bin, repo_root, "run", "--corpus", "http_server",
                "--model", "nonexistent", "--function", "no_such_fn")
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "Traceback" not in out, "must fail cleanly, not crash"
    assert "none of the requested" in out


def test_unknown_function_exits_nonzero_with_skip_preflight(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "run", "--corpus", "http_server",
                "--model", "nonexistent", "--function", "no_such_fn",
                "--skip-preflight")
    assert r.returncode != 0, "an empty selection must never look like success"


def test_corpus_resolves_from_any_cwd(python_bin, repo_root, tmp_path):
    """Corpus dirs resolve from the repo root, not the process cwd."""
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server",
                cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "11 function(s)" in r.stdout


def test_bad_corpus_name_is_a_clean_error(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "does_not_exist")
    assert r.returncode != 0
    assert "config not found" in (r.stderr + r.stdout)


# --- extract --------------------------------------------------------------


def test_extract_reports_composition(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server")
    assert r.returncode == 0
    assert "scoreable composition" in r.stdout
    assert "code 88 (40%)" in r.stdout, "the 40%-code finding, pinned"
    assert "blank 38 (17%)" in r.stdout


def test_extract_warns_about_prose_targets(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server")
    assert "prose-dominated" in r.stdout
    assert "log_message(1)" in r.stdout


def test_extract_min_code_lines_filters_and_feeds_sampling(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server",
                "--min-code-lines", "5")
    assert "filtered to 8 target(s)" in r.stdout
    assert "stratified sample of 8" in r.stdout
    assert "log_message" not in r.stdout.split("stratified sample")[1]


def test_extract_show_labels_line_kinds(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server",
                "--show", "log_message")
    assert r.returncode == 0
    assert "docs|" in r.stdout and "blan|" in r.stdout


def test_extract_show_unknown_function(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server",
                "--show", "nope")
    assert r.returncode == 1
    assert "not found" in r.stdout


# --- rescore --------------------------------------------------------------


@pytest.fixture
def sample_dump(repo_root):
    p = repo_root / "results" / "http_server__gpt-5.5.json"
    if not p.is_file():
        pytest.skip("reference result not present")
    return p


def test_rescore_matches_stored_scores(python_bin, repo_root, sample_dump):
    """Re-scoring must reproduce the stored pass count for a complete run."""
    r = run_cli(python_bin, repo_root, "rescore", str(sample_dump),
                "--corpus", "http_server")
    assert r.returncode == 0, r.stderr
    stored = json.loads(sample_dump.read_text())["results"]
    want = sum(1 for x in stored if x.get("passed"))
    assert f"Pass:                  {want}/11" in r.stdout


def test_rescore_no_comments_is_stricter(python_bin, repo_root, sample_dump):
    base = run_cli(python_bin, repo_root, "rescore", str(sample_dump),
                   "--corpus", "http_server")
    strict = run_cli(python_bin, repo_root, "rescore", str(sample_dump),
                     "--corpus", "http_server", "--no-comments")
    assert strict.returncode == 0
    assert "only code lines earn credit" in strict.stdout
    assert "Scored lines matched" in base.stdout


def test_rescore_reports_blank_lines_skipped(python_bin, repo_root, sample_dump):
    r = run_cli(python_bin, repo_root, "rescore", str(sample_dump),
                "--corpus", "http_server")
    assert "blank lines skipped: 38" in r.stdout
    assert "never earn credit" in r.stdout


def test_rescore_denominator_excludes_blanks(python_bin, repo_root, sample_dump):
    """220 raw lines minus 38 blanks = 182 scoreable."""
    r = run_cli(python_bin, repo_root, "rescore", str(sample_dump),
                "--corpus", "http_server")
    assert "/182" in r.stdout


def test_rescore_warns_on_incomplete_dump(python_bin, repo_root, tmp_path,
                                          sample_dump):
    d = json.loads(sample_dump.read_text())
    d["complete"] = False
    d["queries_run"], d["queries_planned"] = 3, 11
    d["aborted_reason"] = "fail-fast: synthetic"
    p = tmp_path / "partial.json"
    p.write_text(json.dumps(d))
    r = run_cli(python_bin, repo_root, "rescore", str(p), "--corpus", "http_server")
    assert "INCOMPLETE" in r.stderr


# --- help / flag surface --------------------------------------------------


@pytest.mark.parametrize("flag", [
    "--no-comments", "--count-comments", "--min-code-lines", "--notes",
    "--relax-indent", "--strict-indent", "--skip-preflight", "--no-fail-fast",
])
def test_run_help_exposes_flag(python_bin, repo_root, flag):
    r = run_cli(python_bin, repo_root, "run", "--help")
    assert flag in r.stdout
