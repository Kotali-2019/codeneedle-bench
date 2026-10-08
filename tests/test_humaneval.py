"""Execution-based code benchmarks: dataset handling, extraction, exec harness.

All tests are offline — the HumanEval download is never triggered here.
"""
from __future__ import annotations

import json

import pytest

from bench.humaneval import (
    FIXEVAL_PROBLEMS,
    HumanEvalProblem,
    _aggregate_samples,
    _compose_program,
    _extract_code,
    _parse_humaneval,
    _served_model_identity,
    run_humaneval_benchmark,
    run_python,
    score_fixeval,
    score_humaneval,
)


# --- response extraction -------------------------------------------------


def test_extract_code_strips_python_fence():
    response = "```python\n    def f():\n        return 1\n```"
    # Common indent is dropped (markdown semantics), so the
    # indented def still parses at top level.
    assert _extract_code(response) == "def f():\n    return 1"


def test_extract_code_strips_bare_fence():
    response = "Here you go:\n```\ndef f():\n    return 1\n```"
    assert _extract_code(response) == "def f():\n    return 1"


def test_extract_code_strips_leading_think_block():
    response = "</think>thinking hard</think>\n\ndef f():\n    return 1"
    assert _extract_code(response) == "def f():\n    return 1"


def test_extract_code_no_fence_keeps_whole_response():
    response = "def f():\n    return 1"
    assert _extract_code(response) == response


# --- execution harness ---------------------------------------------------


def test_run_python_passes():
    ok, category, out = run_python("print('hi')\n", 5.0)
    assert ok and category == "pass" and out == "hi"


def test_run_python_test_failure():
    ok, category, first = run_python("assert 1 == 2\n", 5.0)
    assert not ok and category == "test_failure" and "assert" in first.lower()


def test_run_python_compile_error():
    ok, category, first = run_python("def f(:\n    pass\n", 5.0)
    assert not ok and category == "compile_error" and "SyntaxError" in first


def test_run_python_timeout():
    ok, category, out = run_python("while True:\n    pass\n", 0.5)
    assert not ok and category == "timeout"


# --- program composition -------------------------------------------------


def test_compose_program_appends_missing_check_call():
    test = "def check(candidate):\n    assert candidate(2) == 4\n"
    program = _compose_program("def f(x):\n    return x * 2\n", test, "f")
    assert "check(f)" in program


def test_compose_program_keeps_explicit_check_call():
    test = "def check(candidate):\n    assert candidate(2) == 4\ncheck(f)\n"
    program = _compose_program("def f(x):\n    return x * 2\n", test, "f")
    assert program.count("check(f)") == 1


def test_compose_program_bare_asserts_untouched():
    program = _compose_program("def f(x):\n    return x * 2\n",
                               "assert f(2) == 4\n", "f")
    assert "check(" not in program


# --- humaneval scoring ---------------------------------------------------


def _problem() -> HumanEvalProblem:
    return HumanEvalProblem(
        task_id="HumanEval/999",
        prompt="def double(x):\n    \"\"\"Return x doubled.\"\"\"\n",
        canonical_solution="    return x * 2\n",
        test="assert double(2) == 4\nassert double(0) == 0\n",
        entry_point="double",
    )


def test_humaneval_body_completion_passes():
    ok, category, code, out = score_humaneval(_problem(), "    return x * 2\n", 5.0)
    assert ok and category == "pass" and code == "    return x * 2"


def test_humaneval_whole_function_completion_passes():
    """Tolerate models that echo the full definition instead of the body."""
    ok, category, _, _ = score_humaneval(
        _problem(), "def double(x):\n    return x * 2\n", 5.0
    )
    assert ok and category == "pass"


def test_humaneval_whole_function_keeps_prompt_imports():
    """A whole-function answer that uses a prompt-side import must
    still see the import (the classic HumanEval NameError trap)."""
    problem = HumanEvalProblem(
        task_id="HumanEval/998",
        prompt="from typing import List\n\n"
               "def pair_up(items: List[int]) -> List[tuple]:\n"
               "    \"\"\"Pair each item with its index.\"\"\"\n",
        canonical_solution="    return list(enumerate(items))\n",
        test="assert pair_up([7, 8]) == [(0, 7), (1, 8)]\n",
        entry_point="pair_up",
    )
    answer = (
        "from typing import List\n\n"
        "def pair_up(items: List[int]) -> List[tuple]:\n"
        "    return list(enumerate(items))\n"
    )
    ok, category, _, _ = score_humaneval(problem, answer, 5.0)
    assert ok and category == "pass"


def test_humaneval_whole_function_without_import_still_works():
    """The model omits the import — the prompt header supplies it."""
    problem = HumanEvalProblem(
        task_id="HumanEval/997",
        prompt="from typing import List\n\n"
               "def pair_up(items: List[int]) -> List[tuple]:\n"
               "    \"\"\"Pair each item with its index.\"\"\"\n",
        canonical_solution="    return list(enumerate(items))\n",
        test="assert pair_up([7, 8]) == [(0, 7), (1, 8)]\n",
        entry_point="pair_up",
    )
    answer = (
        "def pair_up(items: List[int]) -> List[tuple]:\n"
        "    return list(enumerate(items))\n"
    )
    ok, category, _, _ = score_humaneval(problem, answer, 5.0)
    assert ok and category == "pass"


def test_humaneval_whole_function_keeps_helper_functions():
    """Regression: prompts with a helper def above the target
    (HumanEval/10, /32, /38, /50 pattern). Splitting the
    header at the first `def` dropped the helper, so the
    model's correct answer failed with NameError."""
    problem = HumanEvalProblem(
        task_id="HumanEval/996",
        prompt="def is_palindrome(s):\n"
               "    return s == s[::-1]\n\n\n"
               "def make_palindrome(s):\n"
               "    \"\"\"Return the shortest palindrome starting with s.\"\"\"\n",
        canonical_solution="    if is_palindrome(s):\n        return s\n"
                           "    for i in range(len(s)):\n"
                           "        if is_palindrome(s[i:]):\n"
                           "            return s + s[:i][::-1]\n"
                           "    return s + s[:-1][::-1]\n",
        test="assert make_palindrome('cat') == 'catac'\n"
             "assert make_palindrome('cata') == 'catac'\n"
             "assert make_palindrome('') == ''\n",
        entry_point="make_palindrome",
    )
    answer = (
        "def make_palindrome(s):\n"
        "    if is_palindrome(s):\n"
        "        return s\n"
        "    for i in range(len(s)):\n"
        "        if is_palindrome(s[i:]):\n"
        "            return s + s[:i][::-1]\n"
        "    return s + s[:-1][::-1]\n"
    )
    ok, category, _, _ = score_humaneval(problem, answer, 5.0)
    assert ok and category == "pass"


def test_humaneval_wrong_body_fails():
    ok, category, _, _ = score_humaneval(_problem(), "    return x + 2\n", 5.0)
    assert not ok and category == "test_failure"


def test_humaneval_empty_response():
    ok, category, _, _ = score_humaneval(_problem(), "   \n", 5.0)
    assert not ok and category == "empty_response"


# --- fixeval scoring -----------------------------------------------------


def test_fixeval_correct_patch_passes():
    problem = FIXEVAL_PROBLEMS[0]  # binary_search off-by-one
    ok, category, _, _ = score_fixeval(problem, problem.fixed_code, 5.0)
    assert ok and category == "pass"


def test_fixeval_buggy_patch_fails():
    problem = FIXEVAL_PROBLEMS[0]
    ok, category, _, _ = score_fixeval(problem, problem.buggy_code, 5.0)
    assert not ok


# --- fixeval machine-verification (taste: verify gold, never trust) ------


def test_every_fixeval_problem_is_machine_verified():
    """Buggy code must fail its test; fixed code must pass.

    This is the invariant that makes fixeval trustworthy: if a
    problem stops satisfying it, the benchmark silently becomes
    unscoreable.
    """
    assert len(FIXEVAL_PROBLEMS) >= 10
    for problem in FIXEVAL_PROBLEMS:
        buggy_ok, _, _, _ = score_fixeval(problem, problem.buggy_code, 5.0)
        fixed_ok, _, _, _ = score_fixeval(problem, problem.fixed_code, 5.0)
        assert not buggy_ok, f"{problem.id}: buggy code unexpectedly passes"
        assert fixed_ok, f"{problem.id}: fixed code unexpectedly fails"


# --- dataset parsing (offline) -------------------------------------------


def test_parse_humaneval_valid_jsonl():
    text = "\n".join([
        '{"task_id": "HumanEval/0", "prompt": "def f():\\n", '
        '"canonical_solution": "    pass\\n", '
        '"test": "assert f() is None\\n", "entry_point": "f"}',
        '{"task_id": "HumanEval/1", "prompt": "def g():\\n", '
        '"canonical_solution": "    pass\\n", '
        '"test": "assert g() is None\\n", "entry_point": "g"}',
        "",
    ])
    problems = _parse_humaneval(text)
    assert len(problems) == 2
    assert problems[0].task_id == "HumanEval/0"
    assert problems[1].entry_point == "g"


def test_parse_humaneval_rejects_missing_fields():
    row = {"task_id": "HumanEval/0", "prompt": "def f():\n"}
    with pytest.raises(ValueError, match="missing fields"):
        _parse_humaneval(json.dumps(row))


# --- multi-sample aggregation (--samples N) --------------------------


def test_aggregate_samples_all_pass():
    assert _aggregate_samples(
        [{"passed": True, "category": "pass"},
         {"passed": True, "category": "pass"}]
    ) == (True, "pass")


def test_aggregate_samples_mixed_flags_nondeterministic():
    # Samples disagree: the problem was solved at least once
    # so it counts as solved, but the category flags the
    # variance instead of silently picking one verdict.
    assert _aggregate_samples(
        [{"passed": True, "category": "pass"},
         {"passed": False, "category": "test_failure"}]
    ) == (True, "nondeterministic")


def test_aggregate_samples_all_fail_uses_common_category():
    assert _aggregate_samples(
        [{"passed": False, "category": "test_failure"},
         {"passed": False, "category": "test_failure"},
         {"passed": False, "category": "compile_error"}]
    ) == (False, "test_failure")


def test_run_humaneval_multi_sample(monkeypatch, tmp_path):
    """--samples N runs every problem N times: the score
    carries per-sample detail and the dump reports the
    pass@1 per-sample mean inputs."""
    responses = ["    return x * 2\n", "    return x * 3\n"]
    calls = {"n": 0}

    def fake_chat(cfg, system, user):
        text = responses[calls["n"]]
        calls["n"] += 1
        return text

    problem = HumanEvalProblem(
        task_id="HumanEval/990",
        prompt="def double(x):\n    \"\"\"Double x.\"\"\"\n",
        canonical_solution="    return x * 2\n",
        test="assert double(3) == 6\nassert double(0) == 0\n",
        entry_point="double",
    )
    monkeypatch.setattr("bench.humaneval.chat_complete", fake_chat)
    monkeypatch.setattr("bench.humaneval.load_humaneval_problems",
                        lambda: [problem])
    monkeypatch.setattr("bench.humaneval._served_model_identity",
                        lambda *a, **k: None)
    monkeypatch.setattr("bench.humaneval._install_interrupt_handler",
                        lambda: None)

    scores = run_humaneval_benchmark(
        "http://localhost:8000/v1", "test-model",
        dump_path=tmp_path / "he.json", samples=2,
    )
    sc = scores[0]
    assert sc.samples_total == 2
    assert sc.passed_count == 1
    assert sc.passed is True               # solved at least once
    assert sc.category == "nondeterministic"

    d = json.loads((tmp_path / "he.json").read_text(encoding="utf-8"))
    assert d["samples_per_problem"] == 2
    assert d["complete"] is True
    r0 = d["results"][0]
    assert r0["passed_count"] == 1
    assert r0["samples_total"] == 2
    assert r0["category"] == "nondeterministic"
    assert [s["passed"] for s in r0["samples"]] == [True, False]


def test_run_humaneval_empty_response_is_flagged(
        monkeypatch, tmp_path, capsys):
    """A 200 with no content (reasoning models burn the
    token budget on CoT) is recorded as empty_response with
    an actionable hint, not a cryptic AttributeError."""
    problem = HumanEvalProblem(
        task_id="HumanEval/991",
        prompt="def double(x):\n    \"\"\"Double x.\"\"\"\n",
        canonical_solution="    return x * 2\n",
        test="assert double(3) == 6\n",
        entry_point="double",
    )
    monkeypatch.setattr("bench.humaneval.chat_complete",
                        lambda *a: None)
    monkeypatch.setattr("bench.humaneval.load_humaneval_problems",
                        lambda: [problem])
    monkeypatch.setattr("bench.humaneval._served_model_identity",
                        lambda *a, **k: None)
    monkeypatch.setattr("bench.humaneval._install_interrupt_handler",
                        lambda: None)

    scores = run_humaneval_benchmark(
        "http://localhost:8000/v1", "test-model",
        dump_path=tmp_path / "he.json",
    )
    sc = scores[0]
    assert sc.passed is False
    assert sc.category == "empty_response"
    out = capsys.readouterr().out
    assert "--max-tokens" in out


def test_served_model_identity_detects_alias():
    """A vLLM server can serve one set of weights under
    several aliases; the probe records the advertised root
    so two 'different' model ids are not silently compared
    as if they were different models."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"data": [
                {"id": "rtxA4000", "root": "ukisai/Swift",
                 "max_model_len": 262144},
            ]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ident = _served_model_identity(
            f"http://127.0.0.1:{server.server_port}", "rtxA4000")
        assert ident == {"id": "rtxA4000", "root": "ukisai/Swift",
                         "max_model_len": 262144}
        # Unknown id -> None (never fatal to the benchmark).
        assert _served_model_identity(
            f"http://127.0.0.1:{server.server_port}", "nope") is None
    finally:
        server.shutdown()


def test_cmd_run_humaneval_warning_and_flat_suffix(
        monkeypatch, tmp_path, capsys):
    """--relax-indent is a no-op for execution-based corpora
    (warn, don't silently ignore) and --function task ids
    containing '/' flatten to a legal dump filename."""
    import argparse
    import importlib.util
    from pathlib import Path

    from bench import config as bench_config

    # bench.py is a script shadowed by the `bench` package,
    # so load it directly to reach its cmd_run_humaneval.
    spec = importlib.util.spec_from_file_location(
        "bench_cli_under_test",
        Path(__file__).resolve().parent.parent / "bench.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

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

    monkeypatch.setattr(bench_config, "load_model",
                        lambda name: (FakeModel(), False))

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("bench.humaneval.run_humaneval_benchmark",
                        fake_run)
    monkeypatch.setattr(cli, "DEFAULT_RESULTS_DIR", tmp_path)

    args = argparse.Namespace(
        model="testmodel", base_url=None, api_key=None,
        temperature=None, max_tokens=None, timeout=None,
        function=["HumanEval/10"], dump=None, samples=2,
        exec_timeout=10.0, relax_indent=True,
    )
    assert cli.cmd_run_humaneval(args) == 0

    err = capsys.readouterr().err
    assert "no-op" in err
    assert captured["function_filter"] == ["HumanEval/10"]
    assert captured["samples"] == 2
    assert "/" not in captured["dump_path"].name
