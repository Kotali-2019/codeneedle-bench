"""Align model output against ground-truth lines and classify each line."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

from .extract import KIND_BLANK, KIND_CODE, PROSE_KINDS


# The video's threshold was "≥8 of 20 expected lines matched". Expressed as a
# ratio because the denominator is no longer always 20: blank lines never earn
# credit, and comments/docstrings can be excluded via `count_comments=False`.
# On a 20-line, all-code window 0.4 is exactly the original 8/20.
PASS_RATIO = 0.4
PASS_THRESHOLD = 8   # retained for reference; equals PASS_RATIO * 20


class LineTag(str, Enum):
    MATCHED = "matched"            # gray — expected (primary) line reproduced
    MISSING = "missing"            # orange — expected (primary) line not produced
    HALLUCINATED = "hallucinated"  # yellow — produced but not in expected window
    BONUS = "bonus"                # blue — produced, correct, past the primary 20
    IGNORED = "ignored"            # dim — correct, but not eligible for credit
    REINDENTED = "reindented"      # cyan — right content, different leading whitespace


@dataclass
class LineResult:
    tag: LineTag
    text: str


@dataclass
class FunctionScore:
    name: str
    primary_matched: int                 # matched among CREDIT-ELIGIBLE primary lines
    primary_total: int                   # count of credit-eligible primary lines
    hallucinated: int
    bonus_matched: int
    passed: bool
    expected_tagged: list[LineResult]    # expected primary side (matched/missing/ignored)
    predicted_tagged: list[LineResult]   # model output side (matched/halluc/bonus/ignored)
    error: str | None = None             # request errored or returned no usable content; renderers should show ERROR instead of FAIL so it isn't confused with a real recall miss
    # Composition breakdown — how much of the score came from code vs prose.
    code_matched: int = 0
    code_total: int = 0
    prose_matched: int = 0
    prose_total: int = 0
    # A line reproduced correctly but indented differently is not a
    # hallucination (issue #4) — tracked separately so the headline number
    # means what it says.
    reindented: int = 0
    spacing_deviation: bool = False      # True when any line differs only in whitespace
    blank_skipped: int = 0               # expected primary lines excluded as blank
    prose_skipped: int = 0               # expected primary lines excluded as comment/docstring
    raw_total: int = 0                   # every expected primary line, eligible or not

    @property
    def ratio(self) -> float:
        return self.primary_matched / self.primary_total if self.primary_total else 0.0


def _eligible(kind: str, count_comments: bool) -> bool:
    """Whether an expected line of this kind can earn credit.

    Blank lines never do — reproducing whitespace demonstrates no recall, and
    on docstring-heavy corpora they made up ~19% of every window. Comments and
    docstrings do by default (verbatim prose genuinely requires retrieval), but
    `count_comments=False` restricts scoring to code only.
    """
    if kind == KIND_BLANK:
        return False
    if kind in PROSE_KINDS:
        return count_comments
    return True


def score(
    name: str,
    primary: list[str],
    bonus: list[str],
    predicted_text: str,
    relax_indent: bool = False,
    primary_kinds: list[str] | None = None,
    bonus_kinds: list[str] | None = None,
    count_comments: bool = True,
) -> FunctionScore:
    """Score a single function's predicted output against expected lines.

    `relax_indent=True` normalizes both sides with `.strip()` instead of
    `.rstrip()` only — i.e. leading whitespace is ignored when matching. Use
    this for models like Gemma that emit semantically-correct code but
    normalize indentation, where strict verbatim matching would unfairly
    penalize content the model actually got right. Default is strict.

    `primary_kinds` / `bonus_kinds` come from the extractor and drive which
    lines can earn credit. Ineligible lines still take part in the alignment
    (so a model that reproduces blank lines in the right places stays in
    positional sync) — they're only skipped when tallying the score.
    """
    predicted = _clean_output(predicted_text)
    norm = _norm_relaxed if relax_indent else _norm

    exp_primary = [norm(l) for l in primary]
    exp_bonus = [norm(l) for l in bonus]
    exp_full = exp_primary + exp_bonus
    pred = [norm(l) for l in predicted]

    kinds_primary = _resolve_kinds(primary, primary_kinds)
    kinds_bonus = _resolve_kinds(bonus, bonus_kinds)
    kinds_full = kinds_primary + kinds_bonus
    eligible_full = [_eligible(k, count_comments) for k in kinds_full]

    # trim trailing blank lines on prediction (common model artifact)
    while pred and pred[-1] == "":
        pred.pop()

    sm = SequenceMatcher(a=exp_full, b=pred, autojunk=False)

    matched_exp = [False] * len(exp_full)
    # -1 = hallucinated, 0 = primary match, 1 = bonus match,
    #  2 = matched-but-ineligible, 3 = right content, wrong indentation
    pred_kind = [-1] * len(pred)

    for block in sm.get_matching_blocks():
        if block.size == 0:
            continue
        for i in range(block.size):
            ei = block.a + i
            pi = block.b + i
            matched_exp[ei] = True
            if not eligible_full[ei]:
                # Correct, but this line earns no credit — don't let it count
                # toward the score and don't call it a hallucination either.
                pred_kind[pi] = 2
            else:
                pred_kind[pi] = 0 if ei < len(exp_primary) else 1

    # Whitespace-only differences are not hallucinations (issue #4). Under
    # strict matching a re-indented line fails to align, so it lands unmatched
    # on BOTH sides — MISSING expected, HALLUCINATED emitted — for one
    # formatting difference. Pair those back up. Only meaningful in strict
    # mode: with relax_indent they already aligned.
    reindented_exp: set[int] = set()
    if not relax_indent:
        leftover: dict[str, list[int]] = {}
        for i in range(len(exp_full)):
            if matched_exp[i]:
                continue
            key = exp_full[i].strip()
            if key:
                leftover.setdefault(key, []).append(i)
        for pi, kind in enumerate(pred_kind):
            if kind != -1:
                continue
            candidates = leftover.get(pred[pi].strip())
            if not candidates:
                continue
            ei = candidates.pop(0)          # consume: pairs at most once
            if eligible_full[ei]:
                reindented_exp.add(ei)
                pred_kind[pi] = 3
            else:
                # Right content on a line that earns no credit anyway — still
                # not a hallucination.
                pred_kind[pi] = 2

    n_primary = len(exp_primary)
    reindented = sum(1 for k in pred_kind if k == 3)
    primary_total = sum(1 for i in range(n_primary) if eligible_full[i])
    primary_matched = sum(
        1 for i in range(n_primary) if eligible_full[i] and matched_exp[i]
    )
    bonus_matched = sum(
        1 for i in range(n_primary, len(exp_full)) if eligible_full[i] and matched_exp[i]
    )

    code_total = sum(1 for i in range(n_primary) if kinds_full[i] == KIND_CODE)
    code_matched = sum(
        1 for i in range(n_primary) if kinds_full[i] == KIND_CODE and matched_exp[i]
    )
    prose_total = sum(1 for i in range(n_primary) if kinds_full[i] in PROSE_KINDS)
    prose_matched = sum(
        1 for i in range(n_primary) if kinds_full[i] in PROSE_KINDS and matched_exp[i]
    )
    blank_skipped = sum(1 for i in range(n_primary) if kinds_full[i] == KIND_BLANK)
    prose_skipped = prose_total if not count_comments else 0

    hallucinated = sum(1 for k in pred_kind if k == -1)
    # Blank lines shouldn't count as hallucinations (models often insert them).
    hallucinated -= sum(
        1 for i, k in enumerate(pred_kind) if k == -1 and pred[i].strip() == ""
    )

    # Display the ORIGINAL lines (with their actual indentation), not the
    # normalized form used for matching. Otherwise indent-relaxed scoring
    # would render every line lstripped, hiding the model's real output.
    expected_display = [l.rstrip() for l in primary]
    pred_display = [l.rstrip() for l in _clean_output(predicted_text)]
    while pred_display and pred_display[-1] == "":
        pred_display.pop()
    if len(pred_display) != len(pred):
        # Defensive: alignment of pred_display to pred should match because
        # both started from the same _clean_output and stripped trailing blanks.
        pred_display = pred_display[: len(pred)] + [""] * max(0, len(pred) - len(pred_display))

    expected_tagged = []
    for i in range(n_primary):
        if not eligible_full[i]:
            tag = LineTag.IGNORED
        elif matched_exp[i]:
            tag = LineTag.MATCHED
        elif i in reindented_exp:
            tag = LineTag.REINDENTED
        else:
            tag = LineTag.MISSING
        expected_tagged.append(LineResult(tag, expected_display[i]))

    kind_to_tag = {
        0: LineTag.MATCHED,
        1: LineTag.BONUS,
        2: LineTag.IGNORED,
        3: LineTag.REINDENTED,
        -1: LineTag.HALLUCINATED,
    }
    predicted_tagged = [
        LineResult(kind_to_tag[pred_kind[i]], pred_display[i]) for i in range(len(pred))
    ]

    return FunctionScore(
        name=name,
        primary_matched=primary_matched,
        primary_total=primary_total,
        hallucinated=hallucinated,
        bonus_matched=bonus_matched,
        passed=primary_total > 0 and (primary_matched / primary_total) >= PASS_RATIO,
        expected_tagged=expected_tagged,
        predicted_tagged=predicted_tagged,
        code_matched=code_matched,
        code_total=code_total,
        prose_matched=prose_matched,
        prose_total=prose_total,
        blank_skipped=blank_skipped,
        prose_skipped=prose_skipped,
        reindented=reindented,
        spacing_deviation=reindented > 0,
        raw_total=n_primary,
    )


def _resolve_kinds(lines: list[str], kinds: list[str] | None) -> list[str]:
    """Use supplied kinds when they line up; otherwise infer blank-vs-code."""
    if kinds is not None and len(kinds) == len(lines):
        return list(kinds)
    return [KIND_BLANK if l.strip() == "" else KIND_CODE for l in lines]


def _norm(s: str) -> str:
    # Preserve leading indentation; strip trailing whitespace (models are inconsistent there).
    return s.rstrip()


def _norm_relaxed(s: str) -> str:
    # Used when scoring indent-blind. Strips both leading and trailing whitespace.
    # Internal whitespace is preserved so things like `a    b` stay distinct from `a b`.
    return s.strip()


def _clean_output(text: str) -> list[str]:
    """Strip markdown fences and surrounding blank lines. Tolerant of prefix commentary."""
    lines = text.splitlines()

    # If the model wrapped output in fenced code blocks, keep ONLY the fence
    # contents. Pairing fences (rather than slicing first→last) means prose
    # between two separate code blocks isn't scored as hallucinated lines.
    fence_idxs = [i for i, l in enumerate(lines) if l.lstrip().startswith("```")]
    if len(fence_idxs) >= 2:
        kept: list[str] = []
        for open_i, close_i in zip(fence_idxs[0::2], fence_idxs[1::2]):
            kept.extend(lines[open_i + 1 : close_i])
        if len(fence_idxs) % 2 == 1:
            # Unclosed trailing fence — treat it as open to end-of-output.
            kept.extend(lines[fence_idxs[-1] + 1 :])
        lines = kept
    else:
        # Drop any stray fence markers
        lines = [l for l in lines if not l.lstrip().startswith("```")]

    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines
