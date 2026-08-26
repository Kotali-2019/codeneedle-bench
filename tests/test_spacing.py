"""Whitespace-only differences are not hallucinations — regression cover for issue #4.

Under strict scoring a re-indented line fails to align, so it lands in the
unmatched bucket on both sides: the expected line reads MISSING and the
emitted line reads HALLUCINATED. That is one formatting difference counted
twice, and the word "hallucination" misdescribes content the model got right.
"""
from __future__ import annotations

import pytest

from bench.report import render_function, render_summary
from bench.scorer import PASS_THRESHOLD, LineTag, score

CODE = [f"\t\tvar line_{i} = {i};" for i in range(20)]


def _tags(results):
    return [r.tag for r in results]


# --- the reported case ----------------------------------------------------


def test_reindented_line_is_not_a_hallucination():
    """The exact shape reported in issue #4."""
    out = list(CODE)
    out[0] = out[0].lstrip()
    sc = score("val", CODE, [], "\n".join(out))

    assert sc.hallucinated == 0, "a re-indented line must not read as a hallucination"
    assert sc.reindented == 1
    assert sc.spacing_deviation is True


def test_the_same_line_is_not_penalized_twice():
    """It used to count as MISSING on one side and HALLUCINATED on the other."""
    out = list(CODE)
    out[3] = out[3].lstrip()
    sc = score("f", CODE, [], "\n".join(out))

    assert _tags(sc.expected_tagged).count(LineTag.MISSING) == 0
    assert _tags(sc.expected_tagged).count(LineTag.REINDENTED) == 1
    assert _tags(sc.predicted_tagged).count(LineTag.HALLUCINATED) == 0
    assert _tags(sc.predicted_tagged).count(LineTag.REINDENTED) == 1


def test_strict_matching_semantics_are_unchanged():
    """Re-indented lines are still misses: verdicts stay comparable with old runs."""
    out = list(CODE)
    out[0] = out[0].lstrip()
    sc = score("f", CODE, [], "\n".join(out))

    assert sc.primary_matched == 19, "strict scoring still requires verbatim"
    assert sc.primary_total == 20
    assert sc.passed is True


def test_relax_indent_still_scores_them_as_matches():
    out = list(CODE)
    out[0] = out[0].lstrip()
    sc = score("f", CODE, [], "\n".join(out), relax_indent=True)

    assert sc.primary_matched == 20
    assert sc.hallucinated == 0
    assert sc.reindented == 0, "nothing left over to pair when already relaxed"
    assert sc.spacing_deviation is False


# --- it must not swallow real errors --------------------------------------


def test_fabricated_content_is_still_a_hallucination():
    out = list(CODE[:10]) + ["\t\tvar totally_made_up = 999;"] + list(CODE[10:])
    sc = score("f", CODE, [], "\n".join(out))

    assert sc.hallucinated == 1
    assert sc.reindented == 0
    assert sc.spacing_deviation is False


def test_mixed_reindent_and_fabrication_counted_separately():
    out = list(CODE)
    out[2] = out[2].lstrip()
    out[5] = out[5].lstrip()
    out.insert(8, "\t\tvar bogus = 1;")
    sc = score("f", CODE, [], "\n".join(out))

    assert sc.reindented == 2
    assert sc.hallucinated == 1


def test_changed_content_at_same_indent_is_not_reindent():
    out = list(CODE)
    out[4] = "\t\tvar line_4 = 999;"      # same spacing, different content
    sc = score("f", CODE, [], "\n".join(out))

    assert sc.reindented == 0
    assert sc.hallucinated == 1


def test_wholly_dedented_output():
    """Gemma-style: every line re-indented, no content wrong."""
    sc = score("f", CODE, [], "\n".join(l.lstrip() for l in CODE))

    assert sc.hallucinated == 0
    assert sc.reindented == 20
    assert sc.primary_matched == 0, "strict: none reproduced verbatim"
    assert sc.spacing_deviation is True

    relaxed = score("f", CODE, [], "\n".join(l.lstrip() for l in CODE),
                    relax_indent=True)
    assert relaxed.primary_matched == 20


# --- pairing correctness --------------------------------------------------


def test_each_expected_line_pairs_at_most_once():
    """Duplicate emitted lines must not consume the same expected line twice."""
    exp = ["\t\tsame();", "\t\tother();"]
    out = ["same();", "same();", "other();"]
    sc = score("f", exp, [], "\n".join(out))

    assert sc.reindented == 2, "two expected lines available to pair"
    assert sc.hallucinated == 1, "the surplus duplicate is a real extra line"


def test_blank_lines_are_never_paired():
    exp = ["\t\ta();", "", "\t\tb();"]
    sc = score("f", exp, [], "\n".join(["a();", "", "b();"]))

    assert sc.reindented == 2
    assert sc.hallucinated == 0


def test_bonus_window_lines_can_reindent():
    bonus = ["\t\tvar extra = 1;"]
    out = "\n".join(CODE + ["var extra = 1;"])
    sc = score("f", CODE, bonus, out)

    assert sc.hallucinated == 0, "a re-indented bonus line is not fabricated"
    assert sc.reindented == 1


def test_trailing_whitespace_already_ignored():
    """rstrip normalization means trailing spaces were never a mismatch."""
    out = [l + "   " for l in CODE]
    sc = score("f", CODE, [], "\n".join(out))

    assert sc.primary_matched == 20
    assert sc.reindented == 0 and sc.hallucinated == 0


# --- reporting ------------------------------------------------------------


def test_render_shows_reindent_instead_of_hallucination():
    out = list(CODE)
    out[0] = out[0].lstrip()
    text = render_function(score("val", CODE, [], "\n".join(out)), color=False)

    assert "reindented=1" in text
    assert "hallucinated=0" in text
    assert "re-indented (not hallucinations)" in text


def test_summary_points_at_the_existing_flag():
    """--relax-indent already existed; users just could not find it."""
    out = list(CODE)
    out[0] = out[0].lstrip()
    text = render_summary([score("val", CODE, [], "\n".join(out))], color=False)

    assert "Re-indented lines:" in text
    assert "--relax-indent" in text


def test_summary_omits_the_section_when_clean():
    text = render_summary([score("f", CODE, [], "\n".join(CODE))], color=False)
    assert "Re-indented" not in text


# --- against the real corpus ----------------------------------------------


def test_real_jquery_target_reindent(js_source):
    """`val` is the function from the issue report."""
    t = next(x for x in js_source.targets if x.name == "val")
    out = list(t.primary_lines)
    i = next(j for j, l in enumerate(out) if l.strip().startswith("var hooks"))
    out[i] = out[i].lstrip()

    sc = score(t.name, t.primary_lines, t.bonus_lines, "\n".join(out),
               primary_kinds=t.primary_kinds, bonus_kinds=t.bonus_kinds)
    assert sc.hallucinated == 0
    assert sc.reindented == 1
    # Denominator excludes blank lines on this branch, so compare relatively.
    assert sc.primary_matched == sc.primary_total - 1


def test_perfect_output_unaffected(js_source):
    for name in ("val", "dataAttr", "propFilter"):
        t = next(x for x in js_source.targets if x.name == name)
        sc = score(t.name, t.primary_lines, t.bonus_lines, "\n".join(t.primary_lines),
                   primary_kinds=t.primary_kinds, bonus_kinds=t.bonus_kinds)
        assert sc.primary_matched == sc.primary_total, name
        assert sc.hallucinated == 0 and sc.reindented == 0, name
        assert sc.passed and not sc.spacing_deviation, name


@pytest.mark.parametrize("n_correct", [0, 7, 8, 20])
def test_pass_threshold_behaviour_preserved(n_correct):
    out = list(CODE[:n_correct]) + [l.lstrip() for l in CODE[n_correct:]]
    sc = score("f", CODE, [], "\n".join(out))
    assert sc.primary_matched == n_correct
    assert sc.passed is (n_correct >= PASS_THRESHOLD)
