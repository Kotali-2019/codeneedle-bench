"""End-to-end runs against a mock OpenAI-compatible server."""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bench.client import ClientConfig
from bench.generation import current_generation
from bench.runner import DUMP_SCHEMA_VERSION, run_benchmark
from bench.textio import read_text, write_text


class _Handler(BaseHTTPRequestHandler):
    mode = "perfect"
    targets: dict = {}

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        prompt = body["messages"][0]["content"]
        m = re.search(r"function named `([A-Za-z_0-9]+)`", prompt)
        t = self.targets.get(m.group(1)) if m else None

        if self.mode == "http500":
            self.send_response(500)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"boom!")
            return
        if t is None:
            out = ""
        elif self.mode == "prose_only":
            out = "\n".join(
                l for l, k in zip(t.primary_lines, t.primary_kinds) if k != "code"
            )
        elif self.mode == "empty":
            out = ""
        else:
            out = "\n".join(t.primary_lines)

        raw = json.dumps({"choices": [{"message": {"content": out}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def mock_server(py_source):
    _Handler.targets = {t.name: t for t in py_source.targets}
    _Handler.mode = "perfect"
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, _Handler
    srv.shutdown()


def _answerable(source) -> int:
    """Targets a full run plans: eligible minus the unanswerable ones.

    Derived rather than hardcoded — excluding ambiguous targets (a name whose
    signature is not unique in the file) legitimately changes the count.
    """
    return len([t for t in source.targets if not t.ambiguous])


def _cfg(srv) -> ClientConfig:
    host, port = srv.server_address
    return ClientConfig(base_url=f"http://{host}:{port}", model="mock", timeout=20)


# --- happy path -----------------------------------------------------------


def test_perfect_model_passes_everything(mock_server, py_source, tmp_path, capsys):
    srv, h = mock_server
    dump = tmp_path / "out.json"
    scores = run_benchmark(
        source=py_source, cfg=_cfg(srv), k=16, seed=42, dump_path=dump,
        skip_preflight=True, corpus_name="http_server",
    )
    assert len(scores) == _answerable(py_source)
    assert all(s.passed for s in scores)
    assert all(s.error is None for s in scores)
    assert all(s.hallucinated == 0 for s in scores)


def test_dump_records_full_provenance(mock_server, py_source, tmp_path):
    srv, _ = mock_server
    dump = tmp_path / "out.json"
    run_benchmark(
        source=py_source, cfg=_cfg(srv), k=16, seed=42, dump_path=dump,
        skip_preflight=True, corpus_name="http_server",
        notes="unit test, no KV quant", model_label="Mock Model 4bit",
    )
    d = json.loads(read_text(dump))

    assert d["schema_version"] == DUMP_SCHEMA_VERSION
    assert d["benchmark_generation"] == current_generation()
    assert d["complete"] is True
    assert d["valid"] is True
    assert d["query_errors"] == 0
    assert d["queries_run"] == d["queries_planned"] == _answerable(py_source)
    assert d["aborted_reason"] is None
    assert d["corpus"] == "http_server"
    assert len(d["corpus_sha256"]) == 16
    assert d["runtime_notes"] == "unit test, no KV quant"
    assert d["model_label"] == "Mock Model 4bit"
    assert d["sample_k"] == 16 and d["sample_seed"] == 42
    assert d["primary_lines"] == 20
    assert d["scoring"] == {
        "relax_indent": False, "count_comments": True,
        "count_blank_lines": False, "pass_ratio": 0.4,
    }
    for key in ("temperature", "max_tokens", "timeout", "reasoning_effort",
                "prefill_no_think", "use_max_completion_tokens", "stop",
                "suppress_thinking", "min_code_lines", "primary_lines"):
        assert key in d, f"missing provenance field: {key}"


def test_corpus_hash_is_stable_and_content_derived(mock_server, py_source, tmp_path):
    srv, _ = mock_server
    hashes = []
    for i in range(2):
        p = tmp_path / f"d{i}.json"
        run_benchmark(source=py_source, cfg=_cfg(srv), dump_path=p,
                      skip_preflight=True, function_filter=["is_cgi"])
        hashes.append(json.loads(read_text(p))["corpus_sha256"])
    assert hashes[0] == hashes[1]


def test_result_rows_carry_composition_breakdown(mock_server, py_source, tmp_path):
    srv, _ = mock_server
    dump = tmp_path / "out.json"
    run_benchmark(source=py_source, cfg=_cfg(srv), dump_path=dump,
                  skip_preflight=True, corpus_name="http_server")
    rows = json.loads(read_text(dump))["results"]
    for r in rows:
        for key in ("code_matched", "code_total", "prose_matched",
                    "prose_total", "blank_skipped", "raw_total"):
            assert key in r
        assert r["raw_total"] == 20, "every window is 20 expected lines"
        assert r["primary_total"] == r["code_total"] + r["prose_total"]
        assert r["primary_total"] + r["blank_skipped"] == 20


# --- the prose-only false pass -------------------------------------------


def test_prose_only_model_passes_by_default_but_shows_zero_code(
    mock_server, py_source, tmp_path
):
    """The headline finding: a model can pass log_message with no code at all."""
    srv, h = mock_server
    h.mode = "prose_only"
    dump = tmp_path / "out.json"
    run_benchmark(source=py_source, cfg=_cfg(srv), dump_path=dump,
                  skip_preflight=True, corpus_name="http_server")
    rows = {r["function"]: r for r in json.loads(read_text(dump))["results"]}
    lm = rows["log_message"]
    assert lm["passed"] is True, "prose alone still passes under the default policy"
    assert lm["code_matched"] == 0, "and it reproduced zero code lines"
    assert lm["code_total"] == 1, "because the window has only one"


def test_prose_only_model_fails_under_no_comments(mock_server, py_source, tmp_path):
    srv, h = mock_server
    h.mode = "prose_only"
    dump = tmp_path / "out.json"
    run_benchmark(source=py_source, cfg=_cfg(srv), dump_path=dump,
                  skip_preflight=True, corpus_name="http_server",
                  count_comments=False)
    rows = {r["function"]: r for r in json.loads(read_text(dump))["results"]}
    assert rows["log_message"]["passed"] is False
    assert json.loads(read_text(dump))["scoring"]["count_comments"] is False


def test_min_code_lines_drops_prose_targets(mock_server, py_source, tmp_path):
    srv, _ = mock_server
    dump = tmp_path / "out.json"
    scores = run_benchmark(source=py_source, cfg=_cfg(srv), dump_path=dump,
                           skip_preflight=True, corpus_name="http_server",
                           min_code_lines=5)
    names = {s.name for s in scores}
    assert "log_message" not in names, "1-code-line target must be filtered out"
    assert "list_directory" in names
    assert json.loads(read_text(dump))["min_code_lines"] == 5


# --- failure paths --------------------------------------------------------


def test_fail_fast_marks_dump_incomplete(mock_server, py_source, tmp_path):
    srv, h = mock_server
    h.mode = "http500"
    dump = tmp_path / "out.json"
    scores = run_benchmark(source=py_source, cfg=_cfg(srv), dump_path=dump,
                           skip_preflight=True, fail_fast_after=2,
                           corpus_name="http_server")
    d = json.loads(read_text(dump))
    assert len(scores) == 2, "aborted after 2 consecutive errors"
    assert d["complete"] is False
    assert d["valid"] is False
    assert d["query_errors"] == 2
    assert d["queries_run"] == 2
    assert d["queries_planned"] == _answerable(py_source)
    assert "fail-fast" in d["aborted_reason"]


def test_no_fail_fast_runs_every_query(mock_server, py_source, tmp_path):
    srv, h = mock_server
    h.mode = "http500"
    dump = tmp_path / "out.json"
    scores = run_benchmark(source=py_source, cfg=_cfg(srv), dump_path=dump,
                           skip_preflight=True, fail_fast_after=None,
                           corpus_name="http_server")
    assert len(scores) == _answerable(py_source)
    d = json.loads(read_text(dump))
    assert d["complete"] is True, \
        "running every query is a complete run, even if all errored"
    assert d["valid"] is False, "completed execution with errors is not a valid score"
    assert d["query_errors"] == _answerable(py_source)


def test_empty_response_is_error_not_fail(mock_server, py_source, tmp_path):
    """HTTP 200 with no content must read as ERROR, not a recall miss."""
    srv, h = mock_server
    h.mode = "empty"
    scores = run_benchmark(source=py_source, cfg=_cfg(srv), skip_preflight=True,
                           fail_fast_after=None, function_filter=["is_cgi"])
    assert scores[0].error is not None
    assert "empty response" in scores[0].error
    assert not scores[0].passed


def test_empty_selection_raises(mock_server, py_source):
    srv, _ = mock_server
    with pytest.raises(SystemExit, match="none of the requested"):
        run_benchmark(source=py_source, cfg=_cfg(srv), skip_preflight=True,
                      function_filter=["no_such_function"])
