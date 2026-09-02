"""Orchestrate a full benchmark run: extract targets, query the model, score, report."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .client import ClientConfig, chat_complete
from .extract import (
    MIN_BODY_LINES, Source, extract, load_source_glob, stratified_sample,
)
from .report import render_function, render_summary
from .textio import read_text, write_text, write_text_atomic
from .scorer import FunctionScore, score
from .generation import GENERATION_FIELD, current_generation
from .scoring_policy import ScoringPolicy


# Bumped when the dump layout changes in a way consumers must notice.
# 2 = added completeness + provenance + code/prose breakdown.
# 3 = added explicit prompt/scorer generation metadata.
# 4 = added explicit run validity and query-error counts.
# 5 = records the configured primary-line window.
DUMP_SCHEMA_VERSION = 5


# Keeping the file FIRST and the tiny task suffix LAST is deliberate:
# llama.cpp / LM Studio / Ollama all reuse the KV cache for common prefix tokens,
# so across the 16 queries only the tail re-processes. Move the file and the
# cache is invalidated every request.
# The anchor is the target's OWN signature text, quoted verbatim from the
# corpus. The previous wording told the model to look for `function <name>(`,
# which does not exist for property- or assignment-style definitions — 5 of the
# 16 sampled jQuery targets (`PSEUDO`, `init`, `then`, `val`, `parseHTML`) are
# defined as `val: function( value ) {` or `jQuery.parseHTML = function(...)`.
# Across the stored runs those five averaged 59% against 79% for the rest, a
# ~20-point gap present in every model and not explained by depth in the file.
# That was the prompt measuring its own defect. Quoting the real signature also
# disambiguates 10 of the 11 duplicated names in jQuery, since their signatures
# differ even where their names do not.
PROMPT_TEMPLATE = (
    "{file_contents}\n"
    "\n"
    "---\n"
    "\n"
    "Task: reproduce verbatim the first {n} lines of the body of the function named "
    "`{name}`{file_qualifier} from the source above.\n"
    "\n"
    "It is the function introduced by exactly this text:\n"
    "\n"
    "{signature_block}\n"
    "\n"
    "Output the {n} lines that come immediately after it.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY those lines, one per line, in original order.\n"
    "- Preserve original indentation and characters exactly.\n"
    "- Do NOT output the signature text shown above.\n"
    "- Do NOT add commentary, line numbers, or markdown code fences.\n"
    "- If there are blank lines in the body, include them as blank lines.\n"
    "{thinking_suffix}"
)
# Fallback for targets extracted before signature capture existed (or by a
# caller that builds FunctionTarget by hand).
LEGACY_ANCHOR = {
    "js": "the line containing `function {name}(` or the assignment that "
          "introduces it (the line with the opening brace `{{`)",
    "py": "the `def {name}(...):` signature line",
}
# Qwen3 (and other reasoning-enabled models) treat `/no_think` as a directive
# to skip chain-of-thought. Ignored by non-reasoning models. For a pure recall
# benchmark, reasoning wastes tokens and risks drift — so suppress by default.
NO_THINK_SUFFIX = "\n/no_think\n"


@dataclass
class _Run:
    function: str
    source_path: str | None
    prompt_chars: int
    response: str
    latency_s: float
    error: str | None = None


def _signature_block(target) -> str:
    """The quoted definition text, indented so it reads as a block."""
    sig = getattr(target, "signature_text", None)
    if not sig:
        # Nothing captured — fall back to describing the anchor in prose.
        return "    " + LEGACY_ANCHOR[target.language].format(name=target.name)
    return "\n".join("    " + l for l in sig.splitlines())


def _build_prompt(target, text: str, multi_file: bool, suppress_thinking: bool) -> str:
    file_qualifier = (
        f" in file `{target.source_path}`" if multi_file and target.source_path else ""
    )
    return PROMPT_TEMPLATE.format(
        file_contents=text,
        name=target.name,
        file_qualifier=file_qualifier,
        n=len(target.primary_lines),
        signature_block=_signature_block(target),
        thinking_suffix=NO_THINK_SUFFIX if suppress_thinking else "",
    )


def _preflight_context_check(prompt: str, cfg: ClientConfig) -> str | None:
    """Send the actual prompt with a tiny max_tokens to detect context-too-small.

    Returns None on success, an error message string otherwise. Cheap because
    no real generation happens — the model only ingests the prompt and emits
    a single token. As a side benefit it warms the server's prefix KV cache
    for the rest of the run.

    Inherits the full request shape from `cfg` (so flags like
    `use_max_completion_tokens`, `reasoning_effort`, `prefill_no_think`,
    and `stop` apply) — otherwise the probe and the real queries would hit
    different server-side validation paths.

    `max_tokens=16` (not 1): some hosted APIs reject very small budgets
    with "Could not finish the message" before even processing the prompt.
    16 is still negligible cost-wise and finishes in a fraction of a second.
    """
    from dataclasses import replace

    probe_cfg = replace(cfg, max_tokens=16)
    try:
        chat_complete(probe_cfg, system=None, user=prompt)
        return None
    except Exception as e:
        return str(e)


def _is_context_error(msg: str) -> bool:
    m = msg.lower()
    return any(s in m for s in ("context length", "n_ctx", "n_keep", "too long", "exceeds"))


def run_benchmark(
    source: Source,
    cfg: ClientConfig,
    k: int = 16,
    seed: int = 42,
    dump_path: Path | None = None,
    function_filter: list[str] | None = None,
    suppress_thinking: bool = True,
    skip_preflight: bool = False,
    fail_fast_after: int | None = 2,
    relax_indent: bool = False,
    count_comments: bool = True,
    corpus_name: str | None = None,
    notes: str | None = None,
    min_code_lines: int = 0,
    model_label: str | None = None,
) -> list[FunctionScore]:
    text = source.text
    total_lines = text.count("\n") + 1
    print(
        f"Source: {source.display_name}  ({len(text):,} chars, {total_lines:,} lines, "
        f"{len(source.files)} file(s))",
        flush=True,
    )
    print(
        f"Extracted {len(source.targets)} named functions with "
        f"≥{len(source.targets[0].primary_lines) if source.targets else MIN_BODY_LINES} body lines",
        flush=True,
    )

    pool = source.targets

    # A target whose name AND signature are shared with another definition
    # cannot be identified by any prompt, so asking about it is an unanswerable
    # question that scores near zero however good the model is. Drop those.
    unanswerable = [t for t in pool if t.ambiguous]
    if unanswerable:
        pool = [t for t in pool if not t.ambiguous]
        print(
            f"Excluded {len(unanswerable)} target(s) with a duplicate name AND "
            f"identical signature (no prompt could disambiguate): "
            f"{', '.join(sorted(t.name for t in unanswerable))}",
            flush=True,
        )

    if min_code_lines > 0:
        before = len(pool)
        pool = [t for t in pool if t.code_line_count >= min_code_lines]
        dropped = before - len(pool)
        print(
            f"Filtered to {len(pool)} target(s) with ≥{min_code_lines} code line(s) "
            f"in the primary window ({dropped} dropped)",
            flush=True,
        )

    if function_filter:
        wanted = {n for n in function_filter}
        chosen = [t for t in pool if t.name in wanted]
        missing = wanted - {t.name for t in chosen}
        if missing:
            print(f"WARNING: requested but not found: {sorted(missing)}", flush=True)
    else:
        chosen = stratified_sample(pool, total_lines, k=k, seed=seed)

    if not chosen:
        if function_filter:
            raise SystemExit(
                "error: none of the requested --function names exist in this corpus "
                "(see the WARNING above) — nothing to run"
            )
        raise SystemExit(
            "error: the corpus has no named functions long enough for the configured "
            "primary-line window — nothing to run"
        )

    primary_window_lines = len(chosen[0].primary_lines)
    if any(len(target.primary_lines) != primary_window_lines for target in chosen):
        raise ValueError("all selected targets must use the same primary-line window")

    print(f"Selected {len(chosen)} target function(s):", flush=True)
    thin = []
    for t in chosen:
        loc = f"  ({t.source_path.name})" if t.source_path else ""
        n_code = t.code_line_count
        print(
            f"  - {t.name}  line {t.start_line}  body_lines={len(t.body_lines)}  "
            f"code_lines={n_code}/{len(t.primary_lines)}{loc}",
            flush=True,
        )
        if n_code < 5:
            thin.append((t.name, n_code))
    if thin:
        print(
            f"\n  ⚠ {len(thin)} target(s) are mostly comments/blank lines — their score "
            f"reflects prose recall more than code recall:",
            flush=True,
        )
        for nm, n in thin:
            print(f"      {nm} ({n} code line(s) of {primary_window_lines})", flush=True)
        print(
            "    Use --no-comments to score code only, or set "
            "[sample] min_code_lines in the corpus config to exclude them.",
            flush=True,
        )

    multi_file = len(source.files) > 1

    # Pre-flight: send the first real prompt with max_tokens=16 to check that
    # the loaded context is big enough. Misleading FAILs from context-too-small
    # are the easiest mistake to make with LM Studio (TTL-driven JIT reload at
    # default 4K context). Better to abort up front.
    if not skip_preflight:
        probe_prompt = _build_prompt(chosen[0], text, multi_file, suppress_thinking)
        print(
            f"\nPre-flight: probing context fit with a {len(probe_prompt):,}-char prompt "
            f"(max_tokens=16)...",
            flush=True,
        )
        err = _preflight_context_check(probe_prompt, cfg)
        if err is None:
            print("Pre-flight OK.", flush=True)
        elif _is_context_error(err):
            print(f"\n❌ pre-flight failed (context too small):\n   {err}\n", flush=True)
            print("The loaded model context is smaller than the prompt. The most common", flush=True)
            print("cause is LM Studio JIT-reloading at default 4K context after its TTL", flush=True)
            print("expired. Force-reload at the size you need:", flush=True)
            print(f"\n   lms unload {cfg.model}", flush=True)
            print(f"   lms load {cfg.model} --context-length 131072 --gpu max -y\n", flush=True)
            print("Re-run after the model is loaded. (Pass --skip-preflight to override.)", flush=True)
            raise SystemExit(2)
        else:
            print(f"\n❌ pre-flight failed: {err}\n", flush=True)
            print("The server is reachable but rejected the request for a non-context reason.", flush=True)
            print("Fix the server-side error or pass --skip-preflight to push past this check.", flush=True)
            raise SystemExit(2)

    scores: list[FunctionScore] = []
    runs: list[_Run] = []
    consecutive_errors = 0
    aborted_reason: str | None = None

    # Checkpoint after every query. The dump used to be written once, after
    # the loop, so a crash, an OOM or a Ctrl-C discarded the entire run — on an
    # 80K-token corpus that is potentially half an hour of inference. Written
    # atomically, and flagged `in_progress` until the loop finishes, so a dump
    # found mid-run is never mistaken for a finished one.
    def _write_dump(in_progress: bool) -> None:
        if not dump_path:
            return
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        query_errors = sum(1 for score in scores if score.error)
        payload = {
        "schema_version": DUMP_SCHEMA_VERSION,
        GENERATION_FIELD: current_generation(),
        "files": [str(p) for p in source.files],
        "corpus": corpus_name,
        "corpus_sha256": _sha256(source.text),
        "corpus_chars": len(source.text),
        # Completeness — a run that fail-fasted is NOT comparable to a full
        # one. Consumers (charts, run-missing.py) must check this before
        # treating the dump as a finished result.
        "complete": aborted_reason is None,
        "queries_planned": len(chosen),
        "queries_run": len(scores),
        "query_errors": query_errors,
        # A run may finish every planned request yet still contain transport,
        # context, or empty-response errors.  That is complete execution but
        # not a valid model-quality measurement.
        "valid": (
            not in_progress
            and aborted_reason is None
            and len(scores) == len(chosen)
            and query_errors == 0
        ),
        "aborted_reason": aborted_reason,
        # Request shape — everything that changes what the model saw.
        "model": cfg.model,
        "model_label": model_label or cfg.model,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "timeout": cfg.timeout,
        "reasoning_effort": cfg.reasoning_effort,
        "prefill_no_think": cfg.prefill_no_think,
        "use_max_completion_tokens": cfg.use_max_completion_tokens,
        "stop": cfg.stop,
        "suppress_thinking": suppress_thinking,
        # Sampling + scoring policy, so a dump can be reproduced exactly.
        "sample_k": k,
        "sample_seed": seed,
        "function_filter": function_filter,
        "primary_lines": primary_window_lines,
        "min_code_lines": min_code_lines,
        "scoring": ScoringPolicy(
            relax_indent=relax_indent,
            count_comments=count_comments,
        ).as_dict(),
        # Server-side settings the API can't report (KV-cache quantization,
        # loaded context length, runtime/quant build). Record them via
        # `--notes` so published comparisons are auditable.
        "runtime_notes": notes,
        "relax_indent": relax_indent,   # legacy mirror for older readers
        "results": [
            {
                "function": sc.name,
                "source_file": r.source_path,
                "passed": sc.passed,
                "error": sc.error,
                "primary_matched": sc.primary_matched,
                "primary_total": sc.primary_total,
                "code_matched": sc.code_matched,
                "code_total": sc.code_total,
                "prose_matched": sc.prose_matched,
                "prose_total": sc.prose_total,
                "blank_skipped": sc.blank_skipped,
                "raw_total": sc.raw_total,
                "hallucinated": sc.hallucinated,
                "bonus_matched": sc.bonus_matched,
                "latency_s": r.latency_s,
                "prompt_chars": r.prompt_chars,
                "response": r.response,
            }
            for sc, r in zip(scores, runs)
        ],
    }
        payload["in_progress"] = in_progress
        if in_progress:
            payload["complete"] = False
        write_text_atomic(dump_path, json.dumps(payload, indent=2))

    for i, t in enumerate(chosen, 1):
        prompt = _build_prompt(t, text, multi_file, suppress_thinking)
        print(
            f"\n[{i}/{len(chosen)}] `{t.name}` — prompt {len(prompt):,} chars, waiting on model...",
            flush=True,
        )
        start = time.monotonic()
        request_error: str | None = None
        try:
            resp = chat_complete(cfg, system=None, user=prompt)
        except Exception as e:
            request_error = str(e)
            print(f"  ERROR: {request_error}", flush=True)
            resp = ""
        latency = time.monotonic() - start
        if resp is None:
            print(f"  response: None in {latency:.1f}s", flush=True)
            resp = ""
        else:
            print(f"  response: {len(resp)} chars in {latency:.1f}s", flush=True)

        # Empty content with no exception = HTTP 200 but the model produced
        # nothing. On reasoning models that's typically the CoT eating the
        # entire max_tokens budget. Treat as a non-recall error so it shows
        # as ERROR, not FAIL.
        score_error = request_error
        if resp.strip() == "" and score_error is None:
            score_error = (
                "empty response (200 OK but no content; reasoning models often need "
                "more max_tokens — try --max-tokens 8000)"
            )
            print(f"  ⚠ {score_error}", flush=True)

        sc = score(
            t.name, t.primary_lines, t.bonus_lines, resp,
            relax_indent=relax_indent,
            primary_kinds=t.primary_kinds,
            bonus_kinds=t.bonus_kinds,
            count_comments=count_comments,
        )
        if score_error:
            sc.error = score_error
        scores.append(sc)
        runs.append(
            _Run(
                function=t.name,
                source_path=str(t.source_path) if t.source_path else None,
                prompt_chars=len(prompt),
                response=resp,
                latency_s=latency,
                error=score_error,
            )
        )
        print(render_function(sc), flush=True)

        # Persist what we have before starting the next (possibly long) query.
        _write_dump(in_progress=True)

        # Fail-fast: if N queries in a row error, the rest will too. Bail.
        if score_error:
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if (
            fail_fast_after is not None
            and consecutive_errors >= fail_fast_after
            and i < len(chosen)
        ):
            remaining = len(chosen) - i
            aborted_reason = (
                f"fail-fast: {consecutive_errors} consecutive errors after "
                f"{i}/{len(chosen)} queries"
            )
            print(
                f"\n⚠ {consecutive_errors} consecutive ERROR results — aborting the "
                f"remaining {remaining} queries.",
                flush=True,
            )
            print(
                "  Same prompt size + same model + same params → same outcome. "
                "Likely fixes:",
                flush=True,
            )
            print(
                "    • Reasoning model burning the budget? bump --max-tokens (try 8000–12000)",
                flush=True,
            )
            print(
                "    • Server-side error? check logs and the per-query message above",
                flush=True,
            )
            print(
                "  Pass --no-fail-fast to disable this check and run every query anyway.",
                flush=True,
            )
            break

    if relax_indent:
        print("\n(scored with relax_indent=true — leading whitespace ignored on both sides)",
              flush=True)
    if not count_comments:
        print("(scored with --no-comments — only code lines earn credit)", flush=True)
    print(render_summary(scores), flush=True)

    if dump_path:
        _write_dump(in_progress=False)
        print(f"\nResults dumped to {dump_path}", flush=True)
        if aborted_reason:
            print(
                f"  ⚠ marked INCOMPLETE ({len(scores)}/{len(chosen)} queries) — "
                f"charts will flag it and run-missing.py will re-run it.",
                flush=True,
            )

    return scores


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def source_from_single_file(path: Path) -> Source:
    """Convenience: build a Source from one file (for backwards-compat with the file CLI)."""
    targets = extract(path)
    text = read_text(path)
    from .extract import language_of
    return Source(files=[path], text=text, targets=targets, language=language_of(path))
