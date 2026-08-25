"""Results are checkpointed after every query, not only at the end.

The dump used to be written once, after the loop, so a crash / OOM / Ctrl-C
threw away the entire run. On an 80K-token corpus against a large model that
is potentially half an hour of inference.

Idea from tool-eval-bench (MIT), raised by @mazar in issue #5.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bench.client import ClientConfig
from bench.runner import run_benchmark
from bench.textio import read_text, write_text, write_text_atomic


class _Handler(BaseHTTPRequestHandler):
    targets: dict = {}
    delay_after: int = 0          # queries served before stalling
    served: int = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        m = re.search(r"function named `([A-Za-z_0-9]+)`", body["messages"][0]["content"])
        t = self.targets.get(m.group(1)) if m else None
        type(self).served += 1
        out = "\n".join(t.primary_lines) if t else ""
        raw = json.dumps({"choices": [{"message": {"content": out}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server(py_source):
    _Handler.targets = {t.name: t for t in py_source.targets}
    _Handler.served = 0
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _cfg(srv):
    host, port = srv.server_address
    return ClientConfig(base_url=f"http://{host}:{port}", model="mock", timeout=20)


# --- atomic write ---------------------------------------------------------


def test_atomic_write_roundtrips(tmp_path):
    p = tmp_path / "d.json"
    write_text_atomic(p, '{"a": 1}')
    assert json.loads(read_text(p)) == {"a": 1}


def test_atomic_write_leaves_no_temp_file(tmp_path):
    p = tmp_path / "d.json"
    write_text_atomic(p, "x")
    assert [q.name for q in tmp_path.iterdir()] == ["d.json"]


def test_atomic_write_replaces_existing(tmp_path):
    p = tmp_path / "d.json"
    write_text(p, "old")
    write_text_atomic(p, "new")
    assert read_text(p) == "new"


def test_atomic_write_preserves_prior_file_on_failure(tmp_path, monkeypatch):
    """A crash mid-write must not truncate the dump that was already there."""
    p = tmp_path / "d.json"
    write_text_atomic(p, '{"good": true}')

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_text_atomic(p, '{"partial"')
    assert json.loads(read_text(p)) == {"good": True}, "previous dump survived"


# --- checkpointing --------------------------------------------------------


def test_dump_exists_and_grows_during_the_run(server, py_source, tmp_path):
    """A finished run ends up complete and not flagged in progress."""
    dump = tmp_path / "out.json"
    run_benchmark(source=py_source, cfg=_cfg(server), dump_path=dump,
                  skip_preflight=True,
                  function_filter=["is_cgi", "guess_type", "translate_path"],
                  corpus_name="http_server")
    d = json.loads(read_text(dump))
    assert len(d["results"]) == 3
    assert d["in_progress"] is False
    assert d["complete"] is True


def test_partial_dump_is_written_before_the_run_finishes(server, py_source, tmp_path):
    """Kill the process mid-run; the completed queries must still be on disk."""
    dump = tmp_path / "out.json"
    host, port = server.server_address
    script = f'''
import sys, json, os
from pathlib import Path
sys.path.insert(0, {os.getcwd()!r})
from bench.client import ClientConfig
from bench.config import load_corpus
from bench.extract import load_source_glob
from bench.runner import run_benchmark
c = load_corpus("http_server")
s = load_source_glob(c.directory, c.glob, c.limit)
cfg = ClientConfig(base_url="http://{host}:{port}", model="mock", timeout=20)
# Die hard partway through, the way an OOM kill or Ctrl-C would.
n = [0]
import bench.runner as R
orig = R.chat_complete
def counting(cfg, system, user):
    n[0] += 1
    if n[0] > 3:
        os._exit(137)          # SIGKILL-like: no cleanup, no finally blocks
    return orig(cfg, system=system, user=user)
R.chat_complete = counting
run_benchmark(source=s, cfg=cfg, dump_path=Path({str(dump)!r}), skip_preflight=True,
              corpus_name="http_server")
'''
    r = subprocess.run([os.sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=180)
    assert r.returncode != 0, "the child was supposed to die"
    assert dump.exists(), f"a killed run left NOTHING on disk\n{r.stderr[-600:]}"

    d = json.loads(read_text(dump))
    assert d["results"], "dump is empty"
    assert len(d["results"]) == 3, f"expected 3 checkpointed queries, got {len(d['results'])}"
    assert d["in_progress"] is True, "an interrupted dump must say so"
    assert d["complete"] is False


def test_interrupted_dump_is_valid_json(server, py_source, tmp_path):
    dump = tmp_path / "out.json"
    run_benchmark(source=py_source, cfg=_cfg(server), dump_path=dump,
                  skip_preflight=True, function_filter=["is_cgi"],
                  corpus_name="http_server")
    json.loads(read_text(dump))          # must parse


# --- consumers ------------------------------------------------------------


def test_run_missing_treats_in_progress_as_not_done(tmp_path, repo_root):
    import importlib.util

    spec = importlib.util.spec_from_file_location("rm", repo_root / "run-missing.py")
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)

    p = tmp_path / "d.json"
    write_text(p, json.dumps({
        "results": [{"function": "f", "error": None, "passed": True}],
        "in_progress": True, "complete": False,
        "queries_run": 1, "queries_planned": 11,
    }))
    done, why = rm.result_state(p, "http_server")
    assert not done and "still running" in why


def test_charts_flag_an_in_progress_run(tmp_path, repo_root):
    import sys

    sys.path.insert(0, str(repo_root / "analysis"))
    import visualize

    write_text(tmp_path / "jquery__wip.json", json.dumps({
        "files": ["fixtures/jquery.js"], "model": "wip", "in_progress": True,
        "queries_run": 2, "queries_planned": 16,
        "results": [{"function": f"f{i}", "passed": True, "error": None,
                     "primary_matched": 5, "primary_total": 10,
                     "hallucinated": 0, "bonus_matched": 0} for i in range(2)],
    }))
    run = visualize.load_runs(tmp_path)["jquery"][0]
    assert run.complete is False
    assert "INCOMPLETE" in run.label
