"""ANSI-colored rendering of scored results."""
from __future__ import annotations

import sys

from .scorer import FunctionScore, LineTag


# Colors chosen to mirror the video's legend:
#   gray   — correct (matched)
#   orange — expected but missing
#   yellow — hallucinated / mangled
#   blue   — extra correct lines past the primary 20
COLOR = {
    LineTag.MATCHED: "\x1b[37m",        # white/gray
    LineTag.MISSING: "\x1b[38;5;208m",  # 256-color orange
    LineTag.HALLUCINATED: "\x1b[33m",   # yellow
    LineTag.BONUS: "\x1b[36m",          # cyan (blue-ish)
    LineTag.IGNORED: "\x1b[2;37m",      # dim — correct but earns no credit
    LineTag.REINDENTED: "\x1b[38;5;117m",  # light blue — content right, spacing differs
}
RESET = "\x1b[0m"
BOLD = "\x1b[1m"


def _colorize(enabled: bool, color: str, text: str) -> str:
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def _composition(score: FunctionScore) -> str:
    """One-line breakdown of where the score came from.

    Surfaces how much of a result is code recall vs. prose recall — on
    docstring-heavy corpora a window can be mostly comments, and that used to
    be invisible in the headline number.
    """
    parts = [f"code {score.code_matched}/{score.code_total}"]
    if score.prose_total:
        label = "prose (not counted)" if score.prose_skipped else "prose"
        parts.append(f"{label} {score.prose_matched}/{score.prose_total}")
    if score.blank_skipped:
        parts.append(f"{score.blank_skipped} blank skipped")
    return "composition: " + " · ".join(parts)


def render_function(score: FunctionScore, color: bool | None = None) -> str:
    if color is None:
        color = sys.stdout.isatty()

    if score.error:
        # Distinguish from a real recall miss — the model never actually answered.
        status = "ERROR"
        status_color = "\x1b[35m"  # magenta — visually distinct from PASS green / FAIL red
        header = (
            f"\n=== {score.name}  "
            f"[{_colorize(color, status_color, status)}]  "
            f"{score.error}"
        )
        return header

    status = "PASS" if score.passed else "FAIL"
    status_color = "\x1b[32m" if score.passed else "\x1b[31m"
    header = (
        f"\n=== {score.name}  "
        f"[{_colorize(color, status_color, status)}]  "
        f"matched={score.primary_matched}/{score.primary_total} "
        f"({score.ratio*100:.0f}%)  "
        f"hallucinated={score.hallucinated}  "
        + (f"reindented={score.reindented}  " if score.reindented else "")
        + f"bonus={score.bonus_matched} ==="
    )
    out = [header, f"  {_composition(score)}", "  -- model output --"]
    for r in score.predicted_tagged:
        out.append("  " + _colorize(color, COLOR[r.tag], r.text))

    missing = [r for r in score.expected_tagged if r.tag == LineTag.MISSING]
    if missing:
        out.append("  -- missing expected lines --")
        for r in missing:
            out.append("  " + _colorize(color, COLOR[r.tag], r.text))

    reindented = [r for r in score.expected_tagged if r.tag == LineTag.REINDENTED]
    if reindented:
        out.append("  -- reproduced, but re-indented (not hallucinations) --")
        for r in reindented:
            out.append("  " + _colorize(color, COLOR[r.tag], r.text))
    return "\n".join(out)


def render_summary(scores: list[FunctionScore], color: bool | None = None) -> str:
    if color is None:
        color = sys.stdout.isatty()
    errored = [s for s in scores if s.error]
    real = [s for s in scores if not s.error]
    passed = sum(1 for s in real if s.passed)
    total_matched = sum(s.primary_matched for s in real)
    total_possible = sum(s.primary_total for s in real)
    total_halluc = sum(s.hallucinated for s in real)
    total_bonus = sum(s.bonus_matched for s in real)
    code_m = sum(s.code_matched for s in real)
    code_t = sum(s.code_total for s in real)
    prose_m = sum(s.prose_matched for s in real)
    prose_t = sum(s.prose_total for s in real)
    blanks = sum(s.blank_skipped for s in real)
    total_reindent = sum(s.reindented for s in real)
    pct = f"{total_matched / total_possible * 100:.0f}%" if total_possible else "n/a"

    lines = [
        "",
        _colorize(color, BOLD, "=== SUMMARY ==="),
        f"  Pass:                  {passed}/{len(real)}"
        + (f"  ({len(errored)} errored)" if errored else ""),
        f"  Scored lines matched:  {total_matched}/{total_possible}  ({pct})",
        f"    · code:              {code_m}/{code_t}",
    ]
    if prose_t:
        note = "  (excluded from score)" if any(s.prose_skipped for s in real) else ""
        lines.append(f"    · comments/docstrings: {prose_m}/{prose_t}{note}")
    if blanks:
        lines.append(f"    · blank lines skipped: {blanks}  (never earn credit)")
    lines += [
        f"  Hallucinated lines:    {total_halluc}",
        f"  Bonus (extra correct): {total_bonus}",
    ]
    if total_reindent:
        affected = sum(1 for s in real if s.spacing_deviation)
        lines.append(
            f"  Re-indented lines:     {total_reindent}"
            f"  (content correct, spacing differs — not hallucinations)"
        )
        lines.append(
            f"    \u21b3 {affected} function(s) affected; scored as misses under strict"
        )
        lines.append(
            "      matching. Re-run with --relax-indent to score by content."
        )

    # Per-function one-liner
    lines.append("")
    lines.append("  per-function:")
    for s in scores:
        if s.error:
            mark = _colorize(color, "\x1b[35m", "!")
            lines.append(f"    {mark} {s.name:<40} ERROR  {s.error}")
        else:
            mark = _colorize(color, "\x1b[32m", "✓") if s.passed else _colorize(color, "\x1b[31m", "✗")
            lines.append(
                f"    {mark} {s.name:<40} "
                f"matched={s.primary_matched:>2}/{s.primary_total:<2} "
                f"({s.ratio*100:>3.0f}%)  "
                f"code={s.code_matched:>2}/{s.code_total:<2}  "
                f"halluc={s.hallucinated:>2}  bonus={s.bonus_matched:>2}"
                + (f"  reindent={s.reindented:>2}" if s.reindented else "")
            )
    return "\n".join(lines)
