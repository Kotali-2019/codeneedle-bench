#!/usr/bin/env python3
"""Positional recall benchmark — CLI entry.

Tests an LLM's ability to reproduce the first N lines of a named function
inside a large source corpus loaded into context. Measures positional recall,
not just named-entity lookup.

Source selection (extract / run / rescore):
    --corpus NAME      a config under configs/corpora/, or a path to one
    --file PATH        single source file (.js/.mjs/.cjs/.py)

Model selection (run only):
    --model NAME       a config under configs/models/, OR a raw model identifier
                       (raw names get sane defaults; create a config for control)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# --- source resolution ---------------------------------------------------


def _resolve_source(args: argparse.Namespace):
    """Return (Source, CorpusConfig|None) from --corpus or --file."""
    from bench.extract import configure_primary_window, load_source_glob
    from bench.runner import source_from_single_file

    corpus_name = getattr(args, "corpus", None) or "http_server"
    if not getattr(args, "file", None):
        from bench.config import load_corpus

        corpus = load_corpus(corpus_name)
        src = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
        configure_primary_window(src, corpus.primary_lines)
        return src, corpus
    return source_from_single_file(Path(args.file)), None


# --- extract -------------------------------------------------------------


def _composition_line(kinds: list[str]) -> str:
    """Summarize what the expected lines are made of.

    Worth showing prominently: on docstring-heavy corpora most of a window can
    be prose or blank, which changes what a score actually measures.
    """
    from collections import Counter

    n = len(kinds)
    if not n:
        return "no lines"
    c = Counter(kinds)
    parts = [
        f"{label} {c.get(key, 0)} ({c.get(key, 0) / n * 100:.0f}%)"
        for key, label in (
            ("code", "code"), ("blank", "blank"),
            ("comment", "comment"), ("docstring", "docstring"),
        )
        if c.get(key, 0)
    ]
    return f"{n} expected lines: " + ", ".join(parts)


def cmd_extract(args: argparse.Namespace) -> int:
    from bench.extract import MIN_BODY_LINES, stratified_sample

    source, corpus = _resolve_source(args)

    if (
        corpus is not None
        and corpus.sample_functions
        and not args.all
        and not args.show
        and (args.k is not None or args.seed is not None)
    ):
        raise SystemExit(
            "error: this corpus uses fixed [sample].functions; "
            "k/seed cannot change that paired cohort"
        )

    if args.show:
        match = next((t for t in source.targets if t.name == args.show), None)
        if match is None:
            print(f"function {args.show!r} not found")
            return 1
        loc = f"  ({match.source_path})" if match.source_path else ""
        print(f"# {match.name} — start_line={match.start_line}  body_lines={len(match.body_lines)}{loc}")
        print(f"# -- primary (first {len(match.primary_lines)}) --")
        kinds = match.primary_kinds
        for i, l in enumerate(match.primary_lines, 1):
            print(f"{i:>3}| {kinds[i-1][:4]:<4}| {l}")
        print(f"# {_composition_line(match.primary_kinds)}")
        if match.bonus_lines:
            print(f"# -- bonus (next {len(match.bonus_lines)}) --")
            for i, l in enumerate(match.bonus_lines, len(match.primary_lines) + 1):
                print(f"{i:>3}|     | {l}")
        return 0

    total_lines = source.text.count("\n") + 1
    window_lines = corpus.primary_lines if corpus else MIN_BODY_LINES
    print(
        f"{len(source.targets)} function(s) with ≥{window_lines} body lines across "
        f"{len(source.files)} file(s) ({len(source.text):,} chars, {total_lines:,} lines)"
    )
    all_kinds = [k for t in source.targets for k in t.primary_kinds]
    print(f"scoreable composition — {_composition_line(all_kinds)}")

    pool = source.targets
    # Mirror the runner: an ambiguous target is never tested, so `extract`
    # must not advertise it either.
    unanswerable = sorted(t.name for t in pool if t.ambiguous)
    if unanswerable:
        pool = [t for t in pool if not t.ambiguous]
        print(f"excluded {len(unanswerable)} unanswerable target(s) — duplicate name "
              f"AND identical signature: {', '.join(unanswerable)}")

    min_code = (
        args.min_code_lines if args.min_code_lines is not None
        else (corpus.min_code_lines if corpus else 0)
    )
    if min_code > 0:
        before = len(pool)
        pool = [t for t in pool if t.code_line_count >= min_code]
        print(f"filtered to {len(pool)} target(s) with ≥{min_code} code line(s) "
              f"({before - len(pool)} dropped)")
    thin = [t for t in pool if t.code_line_count < 5]
    if thin:
        names = ", ".join(f"{t.name}({t.code_line_count})" for t in thin[:8])
        more = f" +{len(thin)-8} more" if len(thin) > 8 else ""
        print(f"⚠ {len(thin)} prose-dominated target(s) (<5 code lines): {names}{more}")
    k = args.k if args.k is not None else (corpus.sample_k if corpus else 16)
    seed = args.seed if args.seed is not None else (corpus.sample_seed if corpus else 42)
    if args.all:
        chosen = pool
    elif corpus is not None and corpus.sample_functions:
        wanted = set(corpus.sample_functions)
        chosen = [target for target in pool if target.name in wanted]
        missing = wanted - {target.name for target in chosen}
        if missing:
            raise SystemExit(
                "error: configured [sample].functions not found: "
                + ", ".join(sorted(missing))
            )
        print(f"configured paired sample of {len(chosen)}:")
    else:
        chosen = stratified_sample(pool, total_lines, k=k, seed=seed)
        print(f"stratified sample of {len(chosen)}:")
    for t in chosen:
        loc = f"  ({t.source_path.name})" if t.source_path else ""
        print(f"  {t.name:<40}  line={t.start_line:>6}  "
              f"body_lines={len(t.body_lines):<4} code_lines={t.code_line_count:>2}/"
              f"{len(t.primary_lines)}{loc}")
    return 0


# --- run ------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "corpus", None) == "all":
        return cmd_run_all(args)
    if getattr(args, "corpus", None) == "tools":
        return cmd_run_tools(args)

    from bench.config import auto_dump_path, load_model
    from bench.runner import run_benchmark

    source, corpus = _resolve_source(args)

    if (
        corpus is not None
        and corpus.sample_functions
        and not args.function
        and (args.k is not None or args.seed is not None)
    ):
        raise SystemExit(
            "error: this corpus uses fixed [sample].functions; use --function "
            "to run an explicit subset, or --function all to run every extracted "
            "function"
        )

    if not args.model:
        raise SystemExit("error: --model is required (a name in configs/models/, a path, or a raw model id)")
    model, model_from_file = load_model(args.model)
    if not model_from_file:
        print(
            f"  (no model config '{args.model}' found; using as raw model identifier with defaults)",
            file=sys.stderr,
        )

    # CLI overrides — applied on top of whichever source the model came from.
    if args.base_url:
        model.client.base_url = args.base_url
    if args.api_key:
        model.client.api_key = args.api_key
    if args.temperature is not None:
        model.client.temperature = args.temperature
    if args.max_tokens is not None:
        model.client.max_tokens = args.max_tokens
    if args.timeout is not None:
        model.client.timeout = args.timeout
    suppress_thinking = model.suppress_thinking and not args.think

    if corpus is not None:
        k = args.k if args.k is not None else corpus.sample_k
        seed = args.seed if args.seed is not None else corpus.sample_seed
    else:
        k = args.k if args.k is not None else 16
        seed = args.seed if args.seed is not None else 42

    if args.dump:
        dump_path = Path(args.dump)
    elif corpus is not None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = auto_dump_path(corpus, model, DEFAULT_RESULTS_DIR)
    else:
        # --file mode: derive corpus stem from filename
        from bench.config import CorpusConfig

        synthetic_corpus = CorpusConfig(
            name=Path(args.file).stem,
            directory=Path(args.file).parent,
            glob=Path(args.file).name,
            limit=1,
            sample_k=k,
            sample_seed=seed,
            primary_lines=20,
        )
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = auto_dump_path(synthetic_corpus, model, DEFAULT_RESULTS_DIR)

    # Scoring policy belongs to the corpus so every model is judged by the
    # same rules. CLI flags are explicit one-run overrides.
    relax_indent = corpus.relax_indent if corpus is not None else False
    if args.relax_indent:
        relax_indent = True
    if args.strict_indent:
        relax_indent = False

    # Scoring policy: corpus config supplies the default, CLI overrides it.
    count_comments = corpus.count_comments if corpus is not None else True
    if args.no_comments:
        count_comments = False
    if args.count_comments:
        count_comments = True

    fn_filter = (
        args.function if args.function
        else (corpus.sample_functions if corpus is not None else None)
    )
    # Expand --function all to every extracted function name.
    if fn_filter == ["all"] and source is not None:
        fn_filter = [t.name for t in source.targets]
    scores = run_benchmark(
        source=source,
        cfg=model.client,
        k=k,
        seed=seed,
        dump_path=dump_path,
        function_filter=fn_filter,
        suppress_thinking=suppress_thinking,
        skip_preflight=args.skip_preflight,
        fail_fast_after=None if args.no_fail_fast else args.fail_fast_after,
        relax_indent=relax_indent,
        count_comments=count_comments,
        corpus_name=corpus.name if corpus is not None else None,
        notes=args.notes,
        min_code_lines=(
            args.min_code_lines if args.min_code_lines is not None
            else (corpus.min_code_lines if corpus is not None else 0)
        ),
        model_label=model.label,
    )
    # Exit 0 means "the benchmark ran"; per-function FAILs are a normal result,
    # not a tool error. Only a run that couldn't produce results exits non-zero.
    errored = sum(1 for s in scores if s.error)
    return 1 if not scores or errored == len(scores) else 0


# --- run tools ------------------------------------------------------------


def cmd_run_tools(args: argparse.Namespace) -> int:
    from bench.config import load_model
    from bench.toolcall import run_toolcall_benchmark

    if not args.model:
        raise SystemExit("error: --model is required")
    model, model_from_file = load_model(args.model)
    if not model_from_file:
        print(
            f"  (no model config '{args.model}' found; using as raw model identifier with defaults)",
            file=sys.stderr,
        )

    if args.base_url:
        model.client.base_url = args.base_url
    if args.api_key:
        model.client.api_key = args.api_key
    if args.temperature is not None:
        model.client.temperature = args.temperature
    if args.max_tokens is not None:
        model.client.max_tokens = args.max_tokens
    if args.timeout is not None:
        model.client.timeout = args.timeout

    if args.dump:
        dump_path = Path(args.dump)
    else:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = DEFAULT_RESULTS_DIR / f"tools__{model.name}.json"

    scores = run_toolcall_benchmark(
        base_url=model.client.base_url,
        model=model.client.model,
        dump_path=dump_path,
        api_key=model.client.api_key,
        temperature=model.client.temperature,
        max_tokens=model.client.max_tokens,
        timeout=model.client.timeout,
    )
    passed = sum(1 for s in scores if s.overall_pass)
    return 0 if passed == len(scores) else 1


# --- run all --------------------------------------------------------------


def _discover_corpora() -> list[str]:
    """Return sorted list of corpus names from configs/corpora/."""
    corpora_dir = REPO_ROOT / "configs" / "corpora"
    return sorted(p.stem for p in corpora_dir.glob("*.toml"))


def cmd_run_all(args: argparse.Namespace) -> int:
    from bench.config import auto_dump_path, load_corpus, load_model
    from bench.extract import load_source_glob
    from bench.report import render_summary
    from bench.runner import run_benchmark

    corpora = _discover_corpora()
    print(f"Running all {len(corpora)} corpora: {', '.join(corpora)}\n", flush=True)

    if not args.model:
        raise SystemExit("error: --model is required")
    model, model_from_file = load_model(args.model)
    if not model_from_file:
        print(
            f"  (no model config '{args.model}' found; using as raw model identifier with defaults)",
            file=sys.stderr,
        )

    if args.base_url:
        model.client.base_url = args.base_url
    if args.api_key:
        model.client.api_key = args.api_key
    if args.temperature is not None:
        model.client.temperature = args.temperature
    if args.max_tokens is not None:
        model.client.max_tokens = args.max_tokens
    if args.timeout is not None:
        model.client.timeout = args.timeout
    suppress_thinking = model.suppress_thinking and not args.think

    all_scores = []
    corpus_results = []
    any_failure = False

    for corpus_name in corpora:
        print(f"\n{'='*60}", flush=True)
        print(f"  CORPUS: {corpus_name}", flush=True)
        print(f"{'='*60}\n", flush=True)

        corpus = load_corpus(corpus_name)
        src = load_source_glob(corpus.directory, corpus.glob, corpus.limit)

        k = args.k if args.k is not None else corpus.sample_k
        seed = args.seed if args.seed is not None else corpus.sample_seed

        # Scoring policy belongs to the corpus so every model is judged by the
        # same rules. CLI flags are explicit one-run overrides.
        relax_indent = corpus.relax_indent
        if args.relax_indent:
            relax_indent = True
        if args.strict_indent:
            relax_indent = False

        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        lang_tag = src.language
        dump_path = DEFAULT_RESULTS_DIR / f"{model.name}__all-{lang_tag}.json"

        # Expand --function all to every extracted function name for this corpus.
        fn_filter = (
            args.function if args.function else None
        )
        if fn_filter == ["all"]:
            fn_filter = [t.name for t in src.targets]

        scores = run_benchmark(
            source=src,
            cfg=model.client,
            k=k,
            seed=seed,
            dump_path=dump_path,
            function_filter=fn_filter,
            suppress_thinking=suppress_thinking,
            skip_preflight=args.skip_preflight,
            fail_fast_after=None if args.no_fail_fast else args.fail_fast_after,
            relax_indent=relax_indent,
        )
        passed = sum(1 for s in scores if s.passed)
        total = len(scores)
        corpus_results.append((corpus_name, passed, total, scores))
        all_scores.extend(scores)
        if passed < total:
            any_failure = True

    # Run tool calling benchmark as part of --corpus all
    print(f"\n{'='*60}", flush=True)
    print(f"  CORPUS: tools", flush=True)
    print(f"{'='*60}\n", flush=True)

    from bench.toolcall import run_toolcall_benchmark
    tool_dump = DEFAULT_RESULTS_DIR / f"{model.name}__all-tools.json"
    tool_scores = run_toolcall_benchmark(
        base_url=model.client.base_url,
        model=model.client.model,
        dump_path=tool_dump,
        api_key=model.client.api_key,
        temperature=model.client.temperature,
        max_tokens=model.client.max_tokens,
        timeout=model.client.timeout,
    )
    tool_passed = sum(1 for s in tool_scores if s.overall_pass)
    tool_total = len(tool_scores)
    corpus_results.append(("tools", tool_passed, tool_total, []))
    if tool_passed < tool_total:
        any_failure = True

    # Final combined report
    print(f"\n{'='*60}", flush=True)
    print(f"  COMBINED RESULTS — {model.name}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for corpus_name, passed, total, scores in corpus_results:
        status = "✓" if passed == total else "✗"
        if corpus_name == "tools":
            print(f"  {status} {corpus_name:<20} {passed}/{total} passed", flush=True)
        else:
            matched = sum(s.primary_matched for s in scores if not s.error)
            possible = sum(s.primary_total for s in scores if not s.error)
            spacing = sum(1 for s in scores if s.spacing_deviation)
            sp_note = f"  spacing={spacing}" if spacing else ""
            print(f"  {status} {corpus_name:<20} {passed}/{total} passed  "
                  f"lines={matched}/{possible}{sp_note}", flush=True)

    print(flush=True)
    print(render_summary(all_scores), flush=True)

    return 1 if any_failure else 0


# --- rescore --------------------------------------------------------------


def cmd_rescore(args: argparse.Namespace) -> int:
    """Re-score a previous run's dump without re-querying the model."""
    import json

    from bench.extract import load_source_glob
    from bench.report import render_function, render_summary
    from bench.runner import source_from_single_file
    from bench.scorer import score

    from bench.textio import read_text

    dump = json.loads(read_text(args.dump))
    from bench.generation import generation_problem

    generation_warning = generation_problem(dump)
    if generation_warning:
        print(
            f"⚠ this dump is not from the current benchmark generation: "
            f"{generation_warning}. It can be inspected here, but its result must "
            "not be compared with current-generation runs.",
            file=sys.stderr,
        )
    if args.corpus:
        from bench.config import load_corpus

        corpus = load_corpus(args.corpus)
        source = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
    elif args.file:
        source = source_from_single_file(Path(args.file))
    else:
        files = dump.get("files") or ([dump["source"]] if dump.get("source") else [])
        if len(files) == 1 and Path(files[0]).is_file():
            source = source_from_single_file(Path(files[0]))
        else:
            raise SystemExit(
                "error: dump references a missing or multi-file corpus; "
                "pass --corpus NAME or --file PATH to re-locate it"
            )

    from bench.extract import configure_primary_window

    primary_lines = int(dump.get(
        "primary_lines", corpus.primary_lines if args.corpus else 20
    ))
    configure_primary_window(source, primary_lines)

    # Honor the original dump's scoring policy unless overridden on the CLI.
    scoring = dump.get("scoring") or {}
    relax_indent = bool(scoring.get("relax_indent", dump.get("relax_indent", False)))
    if args.relax_indent:
        relax_indent = True
    if args.strict_indent:
        relax_indent = False

    count_comments = bool(scoring.get("count_comments", True))
    if args.no_comments:
        count_comments = False
    if args.count_comments:
        count_comments = True

    if not dump.get("complete", True):
        print(
            f"⚠ this dump is INCOMPLETE ({dump.get('queries_run', '?')}/"
            f"{dump.get('queries_planned', '?')} queries): "
            f"{dump.get('aborted_reason')}",
            file=sys.stderr,
        )

    targets = {t.name: t for t in source.targets}
    scores = []
    for r in dump["results"]:
        t = targets.get(r["function"])
        if t is None:
            print(f"skip: {r['function']} not found in source", file=sys.stderr)
            continue
        sc = score(
            t.name, t.primary_lines, t.bonus_lines,
            r.get("response", ""), relax_indent=relax_indent,
            primary_kinds=t.primary_kinds, bonus_kinds=t.bonus_kinds,
            count_comments=count_comments,
        )
        if r.get("error"):
            sc.error = r["error"]
        scores.append(sc)
        print(render_function(sc))
    if relax_indent:
        print("\n(scored with relax_indent=true — leading whitespace ignored on both sides)")
    if not count_comments:
        print("(scored with --no-comments — only code lines earn credit)")
    print(render_summary(scores))
    return 0


# --- argparse -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    color = p.add_mutually_exclusive_group()
    color.add_argument(
        "--no-color", action="store_true",
        help="never emit ANSI colour (also honoured: NO_COLOR=1). Use when "
             "piping output or on a console that prints escape codes literally",
    )
    color.add_argument(
        "--color", action="store_true",
        help="force ANSI colour even when output is not a terminal "
             "(also honoured: FORCE_COLOR=1)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- extract ------------------------------------------------------------
    p_ex = sub.add_parser("extract", help="list functions the extractor would test")
    src_grp = p_ex.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="corpus config name (configs/corpora/<name>.toml) or path")
    src_grp.add_argument("--file", help="single source file")
    p_ex.add_argument("-k", type=int, default=None,
                      help="override corpus sample.k (stratified corpora only)")
    p_ex.add_argument("--seed", type=int, default=None,
                      help="override corpus sample.seed (stratified corpora only)")
    p_ex.add_argument("--all", action="store_true", help="list every extracted function, not a sample")
    p_ex.add_argument("--show", metavar="NAME", help="print expected primary+bonus lines for one function")
    p_ex.add_argument(
        "--min-code-lines", type=int, default=None, metavar="N",
        help="preview the effect of filtering to functions with ≥N code lines",
    )
    p_ex.set_defaults(func=cmd_extract)

    # --- run ----------------------------------------------------------------
    p_run = sub.add_parser("run", help="run the benchmark against an OpenAI-compatible endpoint")
    src_grp = p_run.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="corpus config name (configs/corpora/<name>.toml) or path")
    src_grp.add_argument("--file", help="single source file")
    p_run.add_argument(
        "--model", required=True,
        help="model config name (configs/models/<name>.toml), a path, or a raw model identifier",
    )
    p_run.add_argument("--base-url", default=None, help="overrides model config")
    p_run.add_argument(
        "--api-key", default=None,
        help="overrides model config (visible in shell history/ps — prefer "
             "api_key_file or api_key_env in the model config for real keys)",
    )
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--max-tokens", type=int, default=None)
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.add_argument("-k", type=int, default=None,
                       help="overrides corpus.sample.k (stratified corpora only)")
    p_run.add_argument("--seed", type=int, default=None,
                       help="overrides corpus.sample.seed (stratified corpora only)")
    p_run.add_argument(
        "--dump", default=None,
        help="JSON path for full results (default: results/<corpus>__<model>.json)",
    )
    p_run.add_argument(
        "--function", action="append",
        help="repeatable; overrides sampling. Use 'all' to run every extracted "
             "function in the corpus (shorthand for listing them all).",
    )
    p_run.add_argument("--think", action="store_true", help="allow chain-of-thought (default: suppress)")
    p_run.add_argument(
        "--skip-preflight", action="store_true",
        help="skip the context-fit pre-flight probe (not recommended for local "
             "servers; saves one full-prompt ingest on paid hosted APIs)",
    )
    p_run.add_argument(
        "--fail-fast-after", type=int, default=2, metavar="N",
        help="abort the run after N consecutive ERROR results (default: 2)",
    )
    p_run.add_argument(
        "--no-fail-fast", action="store_true",
        help="disable fail-fast; run every query even if they're all erroring",
    )
    indent = p_run.add_mutually_exclusive_group()
    indent.add_argument(
        "--relax-indent", action="store_true",
        help="ignore leading whitespace when matching (overrides corpus config to true)",
    )
    indent.add_argument(
        "--strict-indent", action="store_true",
        help="enforce verbatim indentation (overrides corpus config to false)",
    )
    comments = p_run.add_mutually_exclusive_group()
    comments.add_argument(
        "--no-comments", action="store_true",
        help="score only code lines; comments and docstrings earn no credit "
             "(blank lines never do, either way)",
    )
    comments.add_argument(
        "--count-comments", action="store_true",
        help="count comments/docstrings toward the score (the default; "
             "overrides a corpus config that turned them off)",
    )
    p_run.add_argument(
        "--min-code-lines", type=int, default=None, metavar="N",
        help="only test functions with ≥N code lines in the primary window; "
             "excludes docstring-dominated targets (overrides corpus config)",
    )
    p_run.add_argument(
        "--notes", default=None, metavar="TEXT",
        help="free-text runtime provenance recorded in the dump, e.g. "
             "\"LM Studio 0.3.x, Q8 KV cache, 131072 ctx\"",
    )
    p_run.set_defaults(func=cmd_run)

    # --- rescore ------------------------------------------------------------
    p_rs = sub.add_parser("rescore", help="re-score a previous --dump without re-querying")
    p_rs.add_argument("dump", help="path to JSON dump from a prior `run`")
    src_grp = p_rs.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="re-locate corpus via this config")
    src_grp.add_argument("--file", help="re-locate corpus from a single file")
    indent = p_rs.add_mutually_exclusive_group()
    indent.add_argument(
        "--relax-indent", action="store_true",
        help="ignore leading whitespace when matching (overrides dump's setting)",
    )
    indent.add_argument(
        "--strict-indent", action="store_true",
        help="enforce verbatim indentation (overrides dump's setting)",
    )
    comments = p_rs.add_mutually_exclusive_group()
    comments.add_argument(
        "--no-comments", action="store_true",
        help="score only code lines (overrides the dump's setting)",
    )
    comments.add_argument(
        "--count-comments", action="store_true",
        help="count comments/docstrings (overrides the dump's setting)",
    )
    p_rs.set_defaults(func=cmd_rescore)

    return p


def main(argv: list[str] | None = None) -> int:
    # Before anything prints: the report uses ≥, ⚠, ✓ and ✗, none of which
    # survive a locale-encoded stream when output is redirected on Windows.
    from bench.textio import use_utf8_stdio

    use_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    # Colour: explicit flag wins over NO_COLOR / FORCE_COLOR and auto-detection.
    from bench.report import set_color_override

    if getattr(args, "no_color", False):
        set_color_override(False)
    elif getattr(args, "color", False):
        set_color_override(True)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
