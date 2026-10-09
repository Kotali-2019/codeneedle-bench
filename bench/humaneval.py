"""Execution-based code benchmarks: HumanEval-style generation + bug-fix.

Mirrors the `tools`/`gsm8k` pattern: special, non-TOML corpora with
their own module, `run --corpus <name>` handlers, and `--corpus all`
checklist entries.

humaneval — code generation. All 164 problems are downloaded at
runtime from the MIT-licensed openai/human-eval dataset and cached
locally (`.cache/humaneval/HumanEval.jsonl`), so every run is
traceable to the upstream source. A problem passes when the model's
completed function passes its canonical unit tests, executed in an
isolated subprocess.

fixeval — SWE-style-lite. Ten hand-written problems: a function with
a subtle bug plus the failing test. The model must localize and fix
the bug; the fix passes when the test suite goes green in the same
execution harness. Every problem is machine-verified (buggy code
fails its tests, fixed code passes) — see tests/test_humaneval.py.

Security: both corpora execute model-generated code in a subprocess
with a timeout. That is the standard HumanEval execution model, not a
sandbox — point these benchmarks at trusted endpoints only.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .client import (
    ClientConfig,
    chat_complete,
    served_model_identity,
)
from .runner import _check_aborted, _install_interrupt_handler

# Kept under the historical private name so existing
# monkeypatches (tests) and call sites keep working.
_served_model_identity = served_model_identity

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── humaneval dataset ───────────────────────────────────────────

HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)
HUMANEVAL_TOTAL = 164
CACHE_DIR = REPO_ROOT / ".cache" / "humaneval"
CACHE_PATH = CACHE_DIR / "HumanEval.jsonl"


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str            # signature + docstring, no body
    canonical_solution: str
    test: str              # test code (asserts, possibly a check() def)
    entry_point: str


def _parse_humaneval(text: str) -> list[HumanEvalProblem]:
    problems: list[HumanEvalProblem] = []
    required = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"HumanEval.jsonl line {lineno} missing fields: {sorted(missing)}"
            )
        problems.append(
            HumanEvalProblem(
                task_id=row["task_id"],
                prompt=row["prompt"],
                canonical_solution=row["canonical_solution"],
                test=row["test"],
                entry_point=row["entry_point"],
            )
        )
    return problems


def _cache_valid() -> bool:
    if not CACHE_PATH.is_file():
        return False
    try:
        problems = _parse_humaneval(CACHE_PATH.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return False
    return len(problems) == HUMANEVAL_TOTAL


def load_humaneval_problems(force_refresh: bool = False) -> list[HumanEvalProblem]:
    """Return all 164 HumanEval problems, downloading them once.

    Uses the local cache when valid; otherwise downloads the
    gzipped JSONL from the upstream repo and caches the
    decompressed text. A failed download falls back to a valid
    cache; with neither available it exits with actionable
    instructions.
    """
    if not force_refresh and _cache_valid():
        return _parse_humaneval(CACHE_PATH.read_text(encoding="utf-8"))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text: str | None = None
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.get(HUMANEVAL_URL)
            r.raise_for_status()
        text = gzip.decompress(r.content).decode("utf-8")
    except Exception as e:
        if _cache_valid():
            print(
                f"⚠ download failed ({e}); using the cached dataset",
                flush=True,
            )
            return _parse_humaneval(CACHE_PATH.read_text(encoding="utf-8"))
        raise SystemExit(
            f"error: cannot download the HumanEval dataset ({e})\n"
            f"  expected {HUMANEVAL_TOTAL} problems from\n"
            f"  {HUMANEVAL_URL}\n"
            f"  fix: check network access, or place the decompressed\n"
            f"  dataset at\n"
            f"  {CACHE_PATH}"
        )

    problems = _parse_humaneval(text)
    if len(problems) != HUMANEVAL_TOTAL:
        raise SystemExit(
            f"error: downloaded {len(problems)} problems, "
            f"expected {HUMANEVAL_TOTAL} — refusing to cache a partial set"
        )
    CACHE_PATH.write_text(text, encoding="utf-8")
    return problems


# ── fixeval problems (hand-written, machine-verified) ────────────


@dataclass
class FixEvalProblem:
    id: str
    description: str       # what the function should do
    buggy_code: str        # complete function with a subtle bug
    test_code: str         # asserts that the buggy version fails
    fixed_code: str        # reference fix; must pass test_code


FIXEVAL_PROBLEMS: list[FixEvalProblem] = [
    FixEvalProblem(
        "fixeval-01",
        "Return the index of target in a sorted list, or -1 if absent.",
        "def binary_search(items, target):\n"
        "    lo, hi = 0, len(items) - 1\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n",
        "assert binary_search([1, 2, 3, 4, 5], 5) == 4\n"
        "assert binary_search([5], 5) == 0\n"
        "assert binary_search([1, 2, 3], 1) == 0\n"
        "assert binary_search([1, 2, 3], 9) == -1\n",
        "def binary_search(items, target):\n"
        "    lo, hi = 0, len(items) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n",
    ),
    FixEvalProblem(
        "fixeval-02",
        "Return the mean of each consecutive window-sized slice.",
        "def moving_average(values, window):\n"
        "    if window <= 0 or window > len(values):\n"
        "        return []\n"
        "    return [\n"
        "        sum(values[i:i + window]) / window\n"
        "        for i in range(len(values) - window)\n"
        "    ]\n",
        "assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n"
        "assert moving_average([1, 2, 3], 3) == [2.0]\n"
        "assert moving_average([1, 2], 3) == []\n",
        "def moving_average(values, window):\n"
        "    if window <= 0 or window > len(values):\n"
        "        return []\n"
        "    return [\n"
        "        sum(values[i:i + window]) / window\n"
        "        for i in range(len(values) - window + 1)\n"
        "    ]\n",
    ),
    FixEvalProblem(
        "fixeval-03",
        "Group words by length, appending into `buckets`. Calls without "
        "an explicit `buckets` must not share state.",
        "def group_by_length(words, buckets={}):\n"
        "    for w in words:\n"
        "        buckets.setdefault(len(w), []).append(w)\n"
        "    return buckets\n",
        "first = group_by_length(['a', 'bb'])\n"
        "second = group_by_length(['ccc'])\n"
        "assert second == {3: ['ccc']}\n"
        "assert first == {1: ['a'], 2: ['bb']}\n",
        "def group_by_length(words, buckets=None):\n"
        "    if buckets is None:\n"
        "        buckets = {}\n"
        "    for w in words:\n"
        "        buckets.setdefault(len(w), []).append(w)\n"
        "    return buckets\n",
    ),
    FixEvalProblem(
        "fixeval-04",
        "Return the fraction of values strictly below threshold, in [0, 1].",
        "def percentile_rank(values, threshold):\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    below = sum(1 for v in values if v < threshold)\n"
        "    return below // len(values)\n",
        "assert percentile_rank([1, 2, 3, 4], 3) == 0.5\n"
        "assert percentile_rank([1, 1, 1], 2) == 1.0\n"
        "assert percentile_rank([], 5) == 0.0\n",
        "def percentile_rank(values, threshold):\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    below = sum(1 for v in values if v < threshold)\n"
        "    return below / len(values)\n",
    ),
    FixEvalProblem(
        "fixeval-05",
        "Pair each item with the one that follows it; the last item "
        "wraps around and pairs with the first.",
        "def wrap_pairs(items):\n"
        "    return [\n"
        "        (items[i], items[(i + 1) % len(items)])\n"
        "        for i in range(len(items) - 1)\n"
        "    ]\n",
        "assert wrap_pairs(['a', 'b', 'c']) == [('a', 'b'), ('b', 'c'), ('c', 'a')]\n"
        "assert wrap_pairs(['x']) == [('x', 'x')]\n",
        "def wrap_pairs(items):\n"
        "    return [\n"
        "        (items[i], items[(i + 1) % len(items)])\n"
        "        for i in range(len(items))\n"
        "    ]\n",
    ),
    FixEvalProblem(
        "fixeval-06",
        "Return the first value that appears more than once, or None.",
        "def first_duplicate(items):\n"
        "    seen = set()\n"
        "    for it in items:\n"
        "        if it in seen:\n"
        "            return it\n"
        "        seen.add(items[0])\n"
        "    return None\n",
        "assert first_duplicate([1, 2, 3, 2]) == 2\n"
        "assert first_duplicate([5, 5]) == 5\n"
        "assert first_duplicate([1, 2, 3]) is None\n",
        "def first_duplicate(items):\n"
        "    seen = set()\n"
        "    for it in items:\n"
        "        if it in seen:\n"
        "            return it\n"
        "        seen.add(it)\n"
        "    return None\n",
    ),
    FixEvalProblem(
        "fixeval-07",
        "Return the second-largest distinct value, or None when there "
        "is none (including empty input).",
        "def second_largest(values):\n"
        "    largest = max(values)\n"
        "    second = None\n"
        "    for v in values:\n"
        "        if v != largest and (second is None or v > second):\n"
        "            second = v\n"
        "    return second\n",
        "assert second_largest([]) is None\n"
        "assert second_largest([4, 1, 4, 3]) == 3\n"
        "assert second_largest([7, 7]) is None\n",
        "def second_largest(values):\n"
        "    if not values:\n"
        "        return None\n"
        "    largest = max(values)\n"
        "    second = None\n"
        "    for v in values:\n"
        "        if v != largest and (second is None or v > second):\n"
        "            second = v\n"
        "    return second\n",
    ),
    FixEvalProblem(
        "fixeval-08",
        "Return the sum of integers in the inclusive range [lo, hi].",
        "def sum_range(lo, hi):\n"
        "    total = 0\n"
        "    for n in range(lo, hi):\n"
        "        total += n\n"
        "    return total\n",
        "assert sum_range(1, 3) == 6\n"
        "assert sum_range(5, 5) == 5\n"
        "assert sum_range(3, 1) == 0\n",
        "def sum_range(lo, hi):\n"
        "    total = 0\n"
        "    for n in range(lo, hi + 1):\n"
        "        total += n\n"
        "    return total\n",
    ),
    FixEvalProblem(
        "fixeval-09",
        "Return tags lowercased, deduplicated, in first-seen order.",
        "def normalize_tags(tags):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for t in tags:\n"
        "        key = t.lower()\n"
        "        if key not in seen:\n"
        "            seen.add(t)\n"
        "            out.append(key)\n"
        "    return out\n",
        "assert normalize_tags(['A', 'a', 'B']) == ['a', 'b']\n"
        "assert normalize_tags(['x', 'X', 'x']) == ['x']\n",
        "def normalize_tags(tags):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for t in tags:\n"
        "        key = t.lower()\n"
        "        if key not in seen:\n"
        "            seen.add(key)\n"
        "            out.append(key)\n"
        "    return out\n",
    ),
    FixEvalProblem(
        "fixeval-10",
        "Count leaf values in a nested dict-of-dicts. An empty dict "
        "contains no leaves.",
        "def count_leaves(node):\n"
        "    if not isinstance(node, dict):\n"
        "        return 1\n"
        "    total = 0\n"
        "    for child in node.values():\n"
        "        total += count_leaves(child)\n"
        "    return total if total else 1\n",
        "assert count_leaves({'a': {'b': 1}}) == 1\n"
        "assert count_leaves({}) == 0\n"
        "assert count_leaves({'a': {}, 'b': 1}) == 1\n"
        "assert count_leaves({'a': {'b': 1, 'c': 2}}) == 2\n",
        "def count_leaves(node):\n"
        "    if not isinstance(node, dict):\n"
        "        return 1\n"
        "    total = 0\n"
        "    for child in node.values():\n"
        "        total += count_leaves(child)\n"
        "    return total\n",
    ),
]


# ── response extraction + execution harness ─────────────────────


def _extract_code(response: str) -> str:
    """Pull the Python code out of a model response.

    Strips a leading think block (some servers echo the prefill
    back) and markdown fences; tolerates surrounding prose by keeping
    the fenced block when one exists, else the whole response.
    Only blank-line edges are trimmed — the first line's leading
    whitespace is load-bearing (a bare completion must stay indented
    so it nests inside the prompt's signature).
    """
    text = response.strip("\n")
    for open_tag, close in (
        ("</think>", "</think>"),
        ("<thinking>", "</thinking>"),
    ):
        if text.startswith(open_tag):
            idx = text.find(close, len(open_tag))
            if idx != -1:
                text = text[idx + len(close):].strip("\n")
            break
    lines = text.splitlines()
    fence_idxs = [i for i, l in enumerate(lines) if l.lstrip().startswith("```")]
    if len(fence_idxs) >= 2:
        kept: list[str] = []
        for open_i, close_i in zip(fence_idxs[0::2], fence_idxs[1::2]):
            kept.extend(lines[open_i + 1 : close_i])
        # A fenced block is a self-contained unit: drop the common
        # indent (markdown semantics) so an indented `def` still
        # parses at top level.
        return textwrap.dedent("\n".join(kept)).strip("\n")
    return text


def _compose_program(function_code: str, test_code: str, entry_point: str | None) -> str:
    """Join function code and tests into one executable script.

    HumanEval's `test` field may be a `def check(candidate):` block
    that the canonical harness invokes explicitly — append the call
    when it is missing so both dataset shapes run unchanged.
    """
    program = function_code + "\n\n" + test_code
    if entry_point and "def check(" in test_code and f"check({entry_point})" not in test_code:
        program += f"\n\ncheck({entry_point})\n"
    return program


def run_python(program: str, timeout: float) -> tuple[bool, str, str]:
    """Execute a program in an isolated subprocess.

    Returns (ok, category, output). category is one of
    'pass', 'compile_error', 'test_failure', 'timeout', 'crash'.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout", f"exceeded {timeout}s"
    if proc.returncode == 0:
        return True, "pass", proc.stdout.strip()
    err_lines = [l for l in (proc.stderr or "").splitlines() if l.strip()]
    # The exception line is the LAST line of a traceback; the
    # first is always "Traceback (most recent call last):".
    last = err_lines[-1] if err_lines else ""
    if "SyntaxError" in last or "IndentationError" in last:
        return False, "compile_error", last
    return False, "test_failure", last


# ── Scoring ─────────────────────────────────────────────────────


@dataclass
class HumanEvalScore:
    problem: HumanEvalProblem
    raw_response: str = ""
    code: str = ""
    passed: bool = False
    category: str = ""       # pass | compile_error | test_failure | timeout | ...
    exec_output: str = ""
    error: str | None = None
    latency_s: float = 0.0
    # Multi-sample (--samples N): per-sample results and
    # the aggregate over them. passed_count == samples_total
    # means every sample agreed; 0 < passed_count < total
    # is flagged category="nondeterministic".
    samples_total: int = 1
    passed_count: int = 0
    samples: list[dict] = field(default_factory=list)


@dataclass
class FixEvalScore:
    problem: FixEvalProblem
    raw_response: str = ""
    code: str = ""
    passed: bool = False
    category: str = ""
    exec_output: str = ""
    error: str | None = None
    latency_s: float = 0.0
    samples_total: int = 1
    passed_count: int = 0
    samples: list[dict] = field(default_factory=list)


def _prompt_header(prompt: str, entry_point: str) -> str:
    """Everything in a HumanEval prompt before the target function.

    HumanEval prompts can carry imports AND helper functions
    above the target signature (e.g. /10's `is_palindrome`,
    /32's `poly`, /38's `encode_cyclic`). A model that
    returns the whole target function omits the preamble,
    so the composed program must re-attach it — splitting at
    the target's own `def` line, not the first one, or the
    helpers vanish and the tests fail with NameError even
    though the implementation is correct.
    """
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"def {entry_point}("):
            return "\n".join(lines[:i])
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("async def "):
            return "\n".join(lines[:i])
    return ""


def score_humaneval(problem: HumanEvalProblem, raw_response: str,
                    exec_timeout: float) -> tuple[bool, str, str, str]:
    """Extract, compose and execute a HumanEval completion.

    Returns (passed, category, code, exec_output).
    """
    code = _extract_code(raw_response)
    if not code.strip():
        return False, "empty_response", code, ""
    # Tolerate models that return the whole function instead of just
    # the body: a full definition supersedes the prompt's signature,
    # but the prompt's imports still apply.
    if f"def {problem.entry_point}" in code:
        function_code = (
            _prompt_header(problem.prompt, problem.entry_point)
            + "\n" + code
        )
    else:
        function_code = problem.prompt + code
    program = _compose_program(function_code, problem.test, problem.entry_point)
    ok, category, output = run_python(program, exec_timeout)
    return ok, category, code, output


def score_fixeval(problem: FixEvalProblem, raw_response: str,
                  exec_timeout: float) -> tuple[bool, str, str, str]:
    """Extract, compose and execute a fixeval patch."""
    code = _extract_code(raw_response)
    if not code.strip():
        return False, "empty_response", code, ""
    program = _compose_program(code, problem.test_code, None)
    ok, category, output = run_python(program, exec_timeout)
    return ok, category, code, output


# ── Runners ─────────────────────────────────────────────────────


HUMANEVAL_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Implement the function "
    "exactly as its docstring specifies. Output only the function "
    "implementation — no commentary, no tests, no markdown."
)

FIXEVAL_SYSTEM_PROMPT = (
    "You are an expert Python debugger. The function below has a bug "
    "that makes its tests fail. Output the complete corrected function "
    "— no commentary, no test code, no markdown."
)


def _model_config(base_url, model, api_key, temperature, max_tokens, timeout):
    return ClientConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _aggregate_samples(sample_results: list[dict]) -> tuple[bool, str]:
    """Fold per-sample results into (passed, category).

    A problem counts as solved when at least one sample
    passes; when samples disagree the category is
    "nondeterministic" so the variance stays visible
    instead of silently picking one sample's verdict.
    """
    passed_count = sum(1 for s in sample_results if s["passed"])
    if passed_count == len(sample_results):
        return True, "pass"
    if passed_count > 0:
        return True, "nondeterministic"
    cats = Counter(s["category"] for s in sample_results)
    return False, cats.most_common(1)[0][0] if cats else "test_failure"


def run_humaneval_benchmark(
    base_url: str,
    model: str,
    dump_path: Path | None = None,
    api_key: str = "not-needed",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: float = 120.0,
    function_filter: list[str] | None = None,
    exec_timeout: float = 10.0,
    samples: int = 1,
) -> list[HumanEvalScore]:
    # --function filter: run only the named task ids (mirrors the
    # tools corpus' per-tool filtering).
    problems = load_humaneval_problems()
    if function_filter:
        known = {p.task_id for p in problems}
        unknown = [n for n in function_filter if n not in known]
        if unknown:
            raise SystemExit(
                f"error: unknown task id(s): {', '.join(unknown)}; "
                f"available: {', '.join(sorted(known))}"
            )
        wanted = set(function_filter)
        problems = [p for p in problems if p.task_id in wanted]

    if samples < 1:
        raise SystemExit("error: --samples must be >= 1")

    cfg = _model_config(base_url, model, api_key, temperature, max_tokens, timeout)
    scores: list[HumanEvalScore] = []
    total = len(problems)
    served = _served_model_identity(base_url, model)
    if served and served.get("root") and served["root"] != model:
        print(f"note: '{model}' is an alias the server maps to "
              f"'{served['root']}'", flush=True)
    print(f"HumanEval benchmark: {total} problems (execution-based, "
          f"exec_timeout={exec_timeout}s, samples={samples})\n", flush=True)

    _install_interrupt_handler()
    aborted_reason: str | None = None

    for i, problem in enumerate(problems, 1):
        print(f"[{i}/{total}] {problem.task_id:<14} {problem.entry_point:<28}",
              end="", flush=True)
        start = time.monotonic()
        try:
            problem_samples: list[dict] = []
            all_empty = True
            for _ in range(samples):
                raw = chat_complete(cfg, HUMANEVAL_SYSTEM_PROMPT, problem.prompt)
                if raw is None or raw.strip() == "":
                    # Reasoning models can spend the whole token
                    # budget on CoT and return no content.
                    problem_samples.append({
                        "code": "", "passed": False,
                        "category": "empty_response", "exec_output": "",
                        "latency_s": time.monotonic() - start,
                        "raw_response": raw or "",
                    })
                    continue
                all_empty = False
                passed, category, code, exec_output = score_humaneval(
                    problem, raw, exec_timeout
                )
                problem_samples.append({
                    "code": code, "passed": passed, "category": category,
                    "exec_output": exec_output,
                    "latency_s": time.monotonic() - start,
                    "raw_response": raw,
                })
            latency = time.monotonic() - start
            passed, category = _aggregate_samples(problem_samples)
            passed_count = sum(1 for s in problem_samples if s["passed"])
            if all_empty:
                print(f"  ✗ {latency:.1f}s  empty response — model spent "
                      f"its token budget on reasoning; "
                      f"try --max-tokens 16000", flush=True)
            else:
                mark = "✓" if passed else "✗"
                frag = f"  [{passed_count}/{samples} samples]" if samples > 1 else ""
                print(f"  {mark} {latency:.1f}s  {category}{frag}", flush=True)
            last = problem_samples[-1]
            sc = HumanEvalScore(
                problem=problem, raw_response=last["raw_response"],
                code=last["code"], passed=passed, category=category,
                exec_output=last["exec_output"], latency_s=latency,
                samples_total=len(problem_samples),
                passed_count=passed_count, samples=problem_samples,
            )
        except Exception as e:
            latency = time.monotonic() - start
            sc = HumanEvalScore(problem=problem, error=str(e), latency_s=latency)
            print(f"  ERROR {latency:.1f}s  {e}", flush=True)
        scores.append(sc)

        # Checkpoint after every problem: a crash or Ctrl+C
        # mid-run keeps everything scored so far.
        if dump_path:
            _dump_results(scores, "humaneval", model, base_url, dump_path,
                          _humaneval_render, complete=False,
                          served_model=served, samples=samples)
        if _check_aborted(i, total):
            aborted_reason = "interrupted by user (Ctrl+C)"
            break

    _print_code_summary(scores, "HUMANEVAL", key=lambda p: p.problem.task_id)

    if dump_path:
        _dump_results(scores, "humaneval", model, base_url, dump_path,
                      _humaneval_render, complete=True,
                      aborted_reason=aborted_reason,
                      served_model=served, samples=samples)
    return scores


def run_fixeval_benchmark(
    base_url: str,
    model: str,
    dump_path: Path | None = None,
    api_key: str = "not-needed",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: float = 120.0,
    function_filter: list[str] | None = None,
    exec_timeout: float = 10.0,
    samples: int = 1,
) -> list[FixEvalScore]:
    if function_filter:
        known = {p.id for p in FIXEVAL_PROBLEMS}
        unknown = [n for n in function_filter if n not in known]
        if unknown:
            raise SystemExit(
                f"error: unknown problem id(s): {', '.join(unknown)}; "
                f"available: {', '.join(sorted(known))}"
            )
        wanted = set(function_filter)
        problems = [p for p in FIXEVAL_PROBLEMS if p.id in wanted]
    else:
        problems = FIXEVAL_PROBLEMS

    if samples < 1:
        raise SystemExit("error: --samples must be >= 1")

    cfg = _model_config(base_url, model, api_key, temperature, max_tokens, timeout)
    scores: list[FixEvalScore] = []
    total = len(problems)
    served = _served_model_identity(base_url, model)
    if served and served.get("root") and served["root"] != model:
        print(f"note: '{model}' is an alias the server maps to "
              f"'{served['root']}'", flush=True)
    print(f"fixeval benchmark: {total} problems (bug-fix, execution-based, "
          f"exec_timeout={exec_timeout}s, samples={samples})\n", flush=True)

    _install_interrupt_handler()
    aborted_reason: str | None = None

    for i, problem in enumerate(problems, 1):
        print(f"[{i}/{total}] {problem.id:<14} {problem.description[:52]}...",
              end="", flush=True)
        user_prompt = (
            f"Task: {problem.description}\n\n"
            f"The following function has a bug — its tests fail:\n\n"
            f"{problem.buggy_code}\n"
            f"Failing test:\n\n{problem.test_code}\n\n"
            f"Output the complete corrected function."
        )
        start = time.monotonic()
        try:
            problem_samples: list[dict] = []
            all_empty = True
            for _ in range(samples):
                raw = chat_complete(cfg, FIXEVAL_SYSTEM_PROMPT, user_prompt)
                if raw is None or raw.strip() == "":
                    problem_samples.append({
                        "code": "", "passed": False,
                        "category": "empty_response", "exec_output": "",
                        "latency_s": time.monotonic() - start,
                        "raw_response": raw or "",
                    })
                    continue
                all_empty = False
                passed, category, code, exec_output = score_fixeval(
                    problem, raw, exec_timeout
                )
                problem_samples.append({
                    "code": code, "passed": passed, "category": category,
                    "exec_output": exec_output,
                    "latency_s": time.monotonic() - start,
                    "raw_response": raw,
                })
            latency = time.monotonic() - start
            passed, category = _aggregate_samples(problem_samples)
            passed_count = sum(1 for s in problem_samples if s["passed"])
            if all_empty:
                print(f"  ✗ {latency:.1f}s  empty response — model spent "
                      f"its token budget on reasoning; "
                      f"try --max-tokens 16000", flush=True)
            else:
                mark = "✓" if passed else "✗"
                frag = f"  [{passed_count}/{samples} samples]" if samples > 1 else ""
                print(f"  {mark} {latency:.1f}s  {category}{frag}", flush=True)
            last = problem_samples[-1]
            sc = FixEvalScore(
                problem=problem, raw_response=last["raw_response"],
                code=last["code"], passed=passed, category=category,
                exec_output=last["exec_output"], latency_s=latency,
                samples_total=len(problem_samples),
                passed_count=passed_count, samples=problem_samples,
            )
        except Exception as e:
            latency = time.monotonic() - start
            sc = FixEvalScore(problem=problem, error=str(e), latency_s=latency)
            print(f"  ERROR {latency:.1f}s  {e}", flush=True)
        scores.append(sc)

        # Checkpoint after every problem: a crash or Ctrl+C
        # mid-run keeps everything scored so far.
        if dump_path:
            _dump_results(scores, "fixeval", model, base_url, dump_path,
                          _fixeval_render, complete=False,
                          served_model=served, samples=samples)
        if _check_aborted(i, total):
            aborted_reason = "interrupted by user (Ctrl+C)"
            break

    _print_code_summary(scores, "FIXEVAL", key=lambda p: p.problem.id)

    if dump_path:
        _dump_results(scores, "fixeval", model, base_url, dump_path,
                      _fixeval_render, complete=True,
                      aborted_reason=aborted_reason,
                      served_model=served, samples=samples)
    return scores


def _print_code_summary(scores, title: str, key) -> None:
    total = len(scores)
    passed = sum(1 for s in scores if s.passed)
    errored = [s for s in scores if s.error]
    print(f"\n{'='*60}", flush=True)
    print(f"  {title} SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Passed:  {passed}/{total}"
          + (f"  ({len(errored)} errored)" if errored else ""), flush=True)
    total_samples = sum(s.samples_total for s in scores)
    if total_samples > total:
        # With --samples N the headline count is optimistic
        # (solved at least once); the per-sample mean is the
        # honest pass@1 estimate under the run's temperature.
        passed_samples = sum(s.passed_count for s in scores)
        print(f"  pass@1 (per-sample mean): {passed_samples}/{total_samples}"
              f"  = {passed_samples / total_samples:.1%}", flush=True)
    by_cat: Counter[str] = Counter(
        s.category if not s.error else "request_error" for s in scores
    )
    print("  failure breakdown:", flush=True)
    for cat, n in by_cat.most_common():
        if cat == "pass":
            continue
        print(f"    {cat:<18} {n}", flush=True)


def _dump_results(scores, benchmark: str, model: str, base_url: str,
                  dump_path: Path, render, complete: bool = True,
                  aborted_reason: str | None = None,
                  served_model: dict | None = None,
                  samples: int = 1) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": benchmark,
        "model": model,
        # The real model name: the server's advertised
        # root HF id when it reports one (the id sent in
        # API calls is often just a CLI alias).
        "model_label": (served_model or {}).get("root") or model,
        "base_url": base_url,
        "total_tests": len(scores),
        "total_passed": sum(1 for s in scores if s.passed),
        "samples_per_problem": samples,
        "served_model": served_model,
        "complete": complete,
        "aborted_reason": aborted_reason,
        "results": [render(sc) for sc in scores],
    }
    dump_path.write_text(json.dumps(payload, indent=2))
    if complete:
        print(f"\nResults dumped to {dump_path}", flush=True)


def _humaneval_render(sc: HumanEvalScore) -> dict:
    return {
        "task_id": sc.problem.task_id,
        "entry_point": sc.problem.entry_point,
        "prompt": sc.problem.prompt,
        "code": sc.code,
        "passed": sc.passed,
        "passed_count": sc.passed_count,
        "samples_total": sc.samples_total,
        "category": sc.category,
        "exec_output": sc.exec_output,
        "error": sc.error,
        "latency_s": sc.latency_s,
        "samples": sc.samples,
        "raw_response": sc.raw_response,
    }


def _fixeval_render(sc: FixEvalScore) -> dict:
    return {
        "id": sc.problem.id,
        "description": sc.problem.description,
        "buggy_code": sc.problem.buggy_code,
        "test_code": sc.problem.test_code,
        "code": sc.code,
        "passed": sc.passed,
        "passed_count": sc.passed_count,
        "samples_total": sc.samples_total,
        "category": sc.category,
        "exec_output": sc.exec_output,
        "error": sc.error,
        "latency_s": sc.latency_s,
        "samples": sc.samples,
        "raw_response": sc.raw_response,
    }
