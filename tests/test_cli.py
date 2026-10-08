"""CLI-level behavior, exercised via subprocess so argparse and exit codes count."""
from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from bench.textio import read_text, write_text


def _load_cli():
    # bench.py is a script shadowed by the `bench` package,
    # so load it directly to reach its cmd_run_all and
    # pre-flight helpers.
    spec = importlib.util.spec_from_file_location(
        "bench_cli_under_test",
        Path(__file__).resolve().parent.parent / "bench.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_check_endpoint_model_variants():
    """The --corpus all pre-flight must distinguish a
    model the server serves (ok), one it does not (clean
    error naming the served ids), and a server that cannot
    verify (proceed)."""
    cli = _load_cli()

    class Handler(BaseHTTPRequestHandler):
        mode = "listed"

        def do_GET(self):
            if self.path != "/v1/models" or self.mode == "404":
                self.send_response(404)
                self.end_headers()
                return
            if self.mode == "listed":
                body = json.dumps({"data": [
                    {"id": "rtxA4000"}, {"id": "vllm"},
                ]}).encode()
            else:  # "empty"
                body = json.dumps({"data": []}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        # Model the server serves -> ok.
        assert cli._check_endpoint_model(url, "rtxA4000") is None
        # Model it does not serve -> error naming the ids.
        err = cli._check_endpoint_model(url, "gfx906")
        assert err is not None
        assert "gfx906" in err and "rtxA4000" in err
        # Empty listing -> cannot verify, proceed.
        Handler.mode = "empty"
        assert cli._check_endpoint_model(url, "anything") is None
        # No /v1/models at all -> cannot verify, proceed.
        Handler.mode = "404"
        assert cli._check_endpoint_model(url, "anything") is None
    finally:
        server.shutdown()


def test_check_endpoint_model_dead_endpoint():
    """A dead endpoint fails the pre-flight with a clean
    'not reachable' message instead of a traceback."""
    cli = _load_cli()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    err = cli._check_endpoint_model(
        f"http://127.0.0.1:{port}", "some-model")
    assert err is not None
    assert "not reachable" in err


def test_cmd_run_all_validates_endpoint_before_checklist(
        monkeypatch):
    """Regression: the --corpus all checklist used to show
    before the model was even resolved, so a dead endpoint
    or a model the server refuses failed only at the first
    query of the first selected corpus."""
    from bench import config as bench_config

    cli = _load_cli()
    order = []

    def fake_select(corpora, args):
        order.append("checklist")
        return []  # 'none'

    def fake_check(base_url, model):
        order.append("check")
        return None

    class FakeClient:
        base_url = "http://x"
        model = "m"
        api_key = "k"
        temperature = 0.0
        max_tokens = 2000
        timeout = 120.0

    class FakeModel:
        client = FakeClient()
        name = "testmodel"
        suppress_thinking = False

    monkeypatch.setattr(bench_config, "load_model",
                        lambda name: (FakeModel(), False))
    monkeypatch.setattr(cli, "_select_corpora", fake_select)
    monkeypatch.setattr(cli, "_check_endpoint_model", fake_check)

    args = argparse.Namespace(
        model="testmodel", base_url=None, api_key=None,
        temperature=None, max_tokens=None, timeout=None,
        think=False, skip_preflight=False, select=None,
    )
    assert cli.cmd_run_all(args) == 0
    assert order == ["check", "checklist"]


def test_cmd_run_all_dead_endpoint_fails_before_running(
        python_bin, repo_root):
    """End to end: with a dead endpoint the CLI exits 2
    with a clean pre-flight error — no checklist, no
    corpora run."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    r = run_cli(
        python_bin, repo_root, "run", "--corpus", "all",
        "--model", "some-model",
        "--base-url", f"http://127.0.0.1:{port}",
        "--select", "none",
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 2, out
    assert "not reachable" in out
    assert "Running" not in out


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


# --- hardened scoring policy ---------------------------------------------


def test_corpus_configs_load_tuned_scoring_policy(repo_root):
    """The hardening pass: stricter pass ratio + code-dense windows."""
    from bench.config import load_corpus

    for name in ("http_server", "jquery", "rustproj", "cppproj"):
        cfg = load_corpus(name)
        assert cfg.pass_ratio == pytest.approx(0.7), name
        assert cfg.min_code_lines == 5, name


def test_http_server_extract_drops_prose_dominated(python_bin, repo_root):
    """Docstring-heavy windows must not pad the recall average."""
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server")
    assert r.returncode == 0, r.stderr
    assert "filtered to 7 target(s)" in r.stdout


def test_bad_pass_ratio_is_a_config_error(tmp_path):
    from bench.config import load_corpus

    p = tmp_path / "bad-ratio.toml"
    write_text(p, """
[files]
directory = "fixtures"
glob = "http_server.py"
limit = 1
[sample]
k = 16
seed = 42
[scoring]
pass_ratio = 1.5
""")
    with pytest.raises(ValueError, match="pass_ratio"):
        load_corpus(p)


@pytest.mark.parametrize("flags", [
    ("--relax-indent", "--strict-indent"),
    ("--no-comments", "--count-comments"),
])
def test_conflicting_scoring_overrides_are_rejected(python_bin, repo_root, flags):
    r = run_cli(
        python_bin, repo_root, "run", "--corpus", "http_server",
        "--model", "mock", *flags,
    )
    assert r.returncode != 0
    assert "not allowed with argument" in r.stderr


# --- extract --------------------------------------------------------------


def test_extract_reports_composition(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server")
    assert r.returncode == 0
    assert "scoreable composition" in r.stdout
    assert "code 88 (40%)" in r.stdout, "the 40%-code finding, pinned"
    assert "blank 38 (17%)" in r.stdout


def test_extract_warns_about_prose_targets(python_bin, repo_root):
    # The corpus config now filters thin targets (min_code_lines=5);
    # opt out to exercise the warning path itself.
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server",
                "--min-code-lines", "0")
    assert "prose-dominated" in r.stdout
    assert "log_message(1)" in r.stdout


def test_extract_min_code_lines_filters_and_feeds_sampling(python_bin, repo_root):
    r = run_cli(python_bin, repo_root, "extract", "--corpus", "http_server",
                "--min-code-lines", "5")
    # 11 eligible, minus send_head (unanswerable), minus 3 prose-dominated.
    assert "filtered to 7 target(s)" in r.stdout
    assert "stratified sample of 7" in r.stdout
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


def test_fixed_target_corpus_rejects_sampling_overrides(python_bin, repo_root):
    r = run_cli(
        python_bin, repo_root, "extract", "--corpus", "novel_16k", "-k", "1"
    )
    assert r.returncode != 0
    assert "fixed [sample].functions" in (r.stdout + r.stderr)


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
    stored = json.loads(read_text(sample_dump))["results"]
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
    d = json.loads(read_text(sample_dump))
    d["complete"] = False
    d["queries_run"], d["queries_planned"] = 3, 11
    d["aborted_reason"] = "fail-fast: synthetic"
    p = tmp_path / "partial.json"
    write_text(p, json.dumps(d))
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
