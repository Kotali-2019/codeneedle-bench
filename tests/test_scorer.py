"""Scoring policy: blank lines, comments, thresholds, fences, alignment."""
from __future__ import annotations

import pytest

from bench.extract import KIND_BLANK, KIND_CODE, KIND_COMMENT, KIND_DOCSTRING
from bench.scorer import PASS_RATIO, LineTag, score

CODE20 = [f"    line_{i} = {i}" for i in range(20)]
K20 = [KIND_CODE] * 20


# --- blank lines earn no credit -------------------------------------------


def test_blank_lines_excluded_from_denominator():
    exp = ["    a = 1", "", "    b = 2", ""]
    kinds = [KIND_CODE, KIND_BLANK, KIND_CODE, KIND_BLANK]
    sc = score("t", exp, [], "\n".join(exp), primary_kinds=kinds)
    assert sc.primary_total == 2, "blank lines must not inflate the denominator"
    assert sc.primary_matched == 2
    assert sc.blank_skipped == 2


def test_blank_only_output_earns_nothing():
    exp = ["    a = 1", "", "    b = 2"]
    kinds = [KIND_CODE, KIND_BLANK, KIND_CODE]
    sc = score("t", exp, [], "\n\n\n\n\n", primary_kinds=kinds)
    assert sc.primary_matched == 0
    assert not sc.passed


def test_garbage_plus_blanks_scores_zero():
    """The exact inflation this policy exists to kill.

    Under the old scheme, interleaving blank lines with garbage harvested free
    matches from the blanks alone.
    """
    exp = ["    a = 1", "", "    b = 2", "", "    c = 3", ""]
    kinds = [KIND_CODE, KIND_BLANK] * 3
    garbage = "\n\n".join(f"WRONG_{i}" for i in range(6))
    sc = score("t", exp, [], garbage, primary_kinds=kinds)
    assert sc.primary_matched == 0, "blanks must not be harvestable"


def test_predicted_blank_lines_are_not_hallucinations():
    sc = score("t", CODE20, [], "\n\n".join(CODE20), primary_kinds=K20)
    assert sc.primary_matched == 20
    assert sc.hallucinated == 0


# --- comments / docstrings ------------------------------------------------


def test_comments_count_by_default():
    exp = ["    a = 1", "    # note"]
    kinds = [KIND_CODE, KIND_COMMENT]
    sc = score("t", exp, [], "\n".join(exp), primary_kinds=kinds)
    assert sc.primary_total == 2
    assert sc.prose_total == 1 and sc.prose_matched == 1
    assert sc.code_total == 1 and sc.code_matched == 1


def test_no_comments_restricts_denominator_to_code():
    exp = ["    a = 1", "    # note", '    """doc"""']
    kinds = [KIND_CODE, KIND_COMMENT, KIND_DOCSTRING]
    sc = score("t", exp, [], "\n".join(exp), primary_kinds=kinds,
               count_comments=False)
    assert sc.primary_total == 1, "only the code line is eligible"
    assert sc.primary_matched == 1
    assert sc.prose_skipped == 2


def test_ineligible_match_is_neither_credited_nor_hallucinated():
    exp = ["    a = 1", "    # note"]
    kinds = [KIND_CODE, KIND_COMMENT]
    sc = score("t", exp, [], "    # note", primary_kinds=kinds,
               count_comments=False)
    assert sc.primary_matched == 0, "excluded line earns nothing"
    assert sc.hallucinated == 0, "but reproducing it correctly is not a hallucination"
    assert sc.predicted_tagged[0].tag is LineTag.IGNORED


def test_prose_only_answer_fails_under_no_comments():
    """Reproducing only the docstring must not pass a code-only run."""
    exp = ['    """doc line"""', "    a = 1", "    b = 2"]
    kinds = [KIND_DOCSTRING, KIND_CODE, KIND_CODE]
    sc = score("t", exp, [], '    """doc line"""', primary_kinds=kinds,
               count_comments=False)
    assert sc.primary_matched == 0 and not sc.passed


# --- threshold ------------------------------------------------------------


def test_ratio_threshold_equals_original_8_of_20():
    sc8 = score("t", CODE20, [], "\n".join(CODE20[:8]), primary_kinds=K20)
    sc7 = score("t", CODE20, [], "\n".join(CODE20[:7]), primary_kinds=K20)
    assert sc8.primary_total == 20 and sc8.primary_matched == 8 and sc8.passed
    assert sc7.primary_matched == 7 and not sc7.passed
    assert PASS_RATIO == pytest.approx(8 / 20)


def test_ratio_property_and_zero_denominator():
    sc = score("t", ["", ""], [], "", primary_kinds=[KIND_BLANK, KIND_BLANK])
    assert sc.primary_total == 0
    assert sc.ratio == 0.0
    assert not sc.passed, "an all-blank window cannot pass"


# --- output cleaning ------------------------------------------------------


def test_single_fenced_block():
    sc = score("t", CODE20, [], "```python\n" + "\n".join(CODE20) + "\n```",
               primary_kinds=K20)
    assert sc.primary_matched == 20 and sc.hallucinated == 0


def test_two_fenced_blocks_prose_between_not_hallucinated():
    out = (
        "```python\n" + "\n".join(CODE20[:10]) + "\n```\n"
        "Here are the rest of the lines:\n"
        "```python\n" + "\n".join(CODE20[10:]) + "\n```"
    )
    sc = score("t", CODE20, [], out, primary_kinds=K20)
    assert sc.primary_matched == 20
    assert sc.hallucinated == 0, "prose between fences must be discarded"


def test_unclosed_fence_is_read_to_end():
    sc = score("t", CODE20, [], "```\n" + "\n".join(CODE20), primary_kinds=K20)
    assert sc.primary_matched == 20


def test_preamble_without_fences_counts_as_hallucination():
    out = "Sure, here you go:\n" + "\n".join(CODE20)
    sc = score("t", CODE20, [], out, primary_kinds=K20)
    assert sc.primary_matched == 20
    assert sc.hallucinated == 1, "unfenced commentary is a real hallucination"


# --- indentation ----------------------------------------------------------


def test_strict_indent_penalizes_reindentation():
    dedented = [l.strip() for l in CODE20]
    sc = score("t", CODE20, [], "\n".join(dedented), primary_kinds=K20)
    assert sc.primary_matched == 0


def test_relax_indent_accepts_reindentation():
    dedented = [l.strip() for l in CODE20]
    sc = score("t", CODE20, [], "\n".join(dedented), relax_indent=True,
               primary_kinds=K20)
    assert sc.primary_matched == 20


def test_relax_indent_display_keeps_original_indentation():
    sc = score("t", CODE20, [], "\n".join(l.strip() for l in CODE20),
               relax_indent=True, primary_kinds=K20)
    assert sc.expected_tagged[0].text.startswith("    "), \
        "expected side must render real indentation, not the normalized form"


# --- bonus ----------------------------------------------------------------


def test_bonus_lines_counted_past_the_primary_window():
    bonus = [f"    extra_{i} = {i}" for i in range(5)]
    sc = score("t", CODE20, bonus, "\n".join(CODE20 + bonus),
               primary_kinds=K20, bonus_kinds=[KIND_CODE] * 5)
    assert sc.primary_matched == 20 and sc.bonus_matched == 5


def test_blank_bonus_lines_do_not_count():
    bonus = ["", "    extra = 1", ""]
    sc = score("t", CODE20, bonus, "\n".join(CODE20 + bonus),
               primary_kinds=K20,
               bonus_kinds=[KIND_BLANK, KIND_CODE, KIND_BLANK])
    assert sc.bonus_matched == 1


# --- robustness -----------------------------------------------------------


def test_kinds_fallback_when_absent_or_mismatched():
    """score() must stay usable without extractor kinds."""
    exp = ["    a = 1", "", "    b = 2"]
    sc = score("t", exp, [], "\n".join(exp))
    assert sc.primary_total == 2, "blank inferred from content"
    sc_bad = score("t", exp, [], "\n".join(exp), primary_kinds=[KIND_CODE])
    assert sc_bad.primary_total == 2, "wrong-length kinds fall back safely"


def test_empty_response_scores_zero():
    sc = score("t", CODE20, [], "", primary_kinds=K20)
    assert sc.primary_matched == 0 and sc.hallucinated == 0 and not sc.passed
