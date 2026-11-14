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
    """~22% of the old matched-line total across every shipped run was blanks.

    Only runs whose stored scores actually contained blank credit
    count: strict-policy runs (old == new) would otherwise dilute
    the ratio as the results directory grows.
    """
    by_corpus = {
        "http_server": {t.name: t for t in py_source.targets},
        "jquery": {t.name: t for t in js_source.targets},
    }
    inflated = []
    for p in _stored(repo_root):
        d = json.loads(read_text(p))
        corpus = pathlib.Path(p).stem.split("__")[0]
        if corpus not in by_corpus:
            continue
        relax = bool((d.get("scoring") or {}).get(
            "relax_indent", d.get("relax_indent", False)))
        old = new = 0
        for r in d["results"]:
            t = by_corpus[corpus].get(r["function"])
            if t is None or r.get("error"):
                continue
            sc = score(t.name, t.primary_lines, t.bonus_lines,
                       r.get("response", ""), relax_indent=relax,
                       primary_kinds=t.primary_kinds, bonus_kinds=t.bonus_kinds)
            old += r["primary_matched"]
            new += sc.primary_matched
        if old > new:
            inflated.append((old, new))
    assert inflated, "no stored run showed blank-line credit"
    old = sum(o for o, _ in inflated)
    new = sum(n for _, n in inflated)
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


# --- endpoint URL handling --------------------------------------------------


def _openai_stub_server():
    """Serve OpenAI endpoints under exactly one /v1 and 404 everything else.

    Mirrors the vLLM router behind the bug: GET /v1/models lists one real id,
    POST /v1/chat/completions answers any model EXCEPT 'strict-rejected' (which
    it refuses with 400), and any path that is not exactly one of those two
    returns 404 — so a client that double-prefixes /v1 hits 404 on every call.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    completion = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "served",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send(200, {"data": [{"id": "qwen-ud-iq2_xs", "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            try:
                model = json.loads(self.rfile.read(n)).get("model")
            except Exception:
                model = None
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
            elif model == "strict-rejected":
                self._send(400, {"error": {"message": "model not served"}})
            else:
                self._send(200, completion)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_api_base_normalizes_both_url_forms():
    """The OpenAI-SDK form (…/v1) and the LM Studio form (…) collapse to one /v1.

    Joining /v1 onto an already-/v1 base double-prefixes it (…/v1/v1/models ->
    404); api_base must return exactly one trailing /v1 for every accepted form.
    """
    from bench.client import api_base

    for url in ("http://h:8888", "http://h:8888/", "http://h:8888/v1",
                "http://h:8888/v1/"):
        assert api_base(url) == "http://h:8888/v1", url
    # A path prefix on the host is kept, not truncated.
    assert api_base("http://host/gateway") == "http://host/gateway/v1"


def test_chat_complete_accepts_url_with_and_without_v1():
    """Every request must reach the server for BOTH base_url forms.

    Regression: with base_url ending in /v1 the client built /v1/v1/... and the
    router 404'd it — the pre-flight passed (listing 'not verifiable') so the
    user only hit 404 after choosing a corpus.
    """
    from bench.client import ClientConfig, chat_complete, list_served_models

    server = _openai_stub_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for form in (base, base + "/v1"):
            cfg = ClientConfig(base_url=form, model="gfx906",
                               max_tokens=8, timeout=10.0)
            assert chat_complete(cfg, system=None, user="ok") == "ok", form
            assert list_served_models(form) is not None, form
    finally:
        server.shutdown()


def test_preflight_accepts_alias_and_rejects_unserved(repo_root):
    """A served alias absent from /v1/models is accepted; a refused id is fatal.

    The /v1/models listing is a hint, not proof: an alias (gfx906) served by the
    router is not listed, so the pre-flight must probe before trusting it. A real
    refusal (strict-rejected -> 400) and a dead endpoint must still fail loudly.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench_cli_preflight", repo_root / "bench.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    server = _openai_stub_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert cli._check_endpoint_model(base, "gfx906") is None
        assert cli._check_endpoint_model(base + "/v1", "gfx906") is None
        err = cli._check_endpoint_model(base, "strict-rejected")
        assert err and "not served" in err
        assert "not reachable" in cli._check_endpoint_model("http://127.0.0.1:1", "m")
    finally:
        server.shutdown()
