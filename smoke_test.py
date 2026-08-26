"""End-to-end smoke test without an LLM.

Generates a synthetic Python source file, runs the extractor, and runs the
scorer with fake model outputs of varying quality. Exits non-zero on failure.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from bench.extract import (
    KIND_BLANK, KIND_CODE, KIND_COMMENT, KIND_DOCSTRING,
    extract, line_kinds, stratified_sample,
)
from bench.scorer import PASS_RATIO, score


def build_fixture(n_funcs: int = 30, body_lines: int = 25) -> str:
    """Emit a Python file with n_funcs functions each having body_lines body lines."""
    out: list[str] = ["'''Generated fixture.'''", ""]
    for i in range(n_funcs):
        out.append(f"def func_{i:03d}(arg_a, arg_b, arg_c):")
        out.append(f"    '''Function number {i}.'''")
        out.append(f"    total = arg_a + arg_b + arg_c")
        for j in range(body_lines - 2):
            out.append(f"    step_{j} = total * {i * 7 + j}  # op in func {i}")
        out.append(f"    return total * {i}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    src = build_fixture()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = Path(f.name)

    try:
        targets = extract(path)
        total_lines = src.count("\n") + 1
        print(f"extracted {len(targets)} functions from fixture ({total_lines} lines)")
        assert len(targets) == 30, f"expected 30, got {len(targets)}"

        sample = stratified_sample(targets, total_lines, k=8, seed=1)
        assert len(sample) == 8

        t = targets[0]
        print(f"\nfirst target: {t.name}, start_line={t.start_line}, body_lines={len(t.body_lines)}")
        print("primary (first 5 of 20):")
        for l in t.primary_lines[:5]:
            print(f"  {l!r}")

        # case 1: perfect output
        perfect = "\n".join(t.primary_lines)
        sc = score(t.name, t.primary_lines, t.bonus_lines, perfect)
        print(f"\nperfect: matched={sc.primary_matched}/{sc.primary_total}  "
              f"halluc={sc.hallucinated}  bonus={sc.bonus_matched}  passed={sc.passed}")
        assert sc.primary_matched == 20 and sc.hallucinated == 0 and sc.passed

        # case 2: perfect + bonus lines past the 20
        perfect_plus = "\n".join(t.primary_lines + t.bonus_lines[:5])
        sc = score(t.name, t.primary_lines, t.bonus_lines, perfect_plus)
        print(f"perfect+bonus: matched={sc.primary_matched}/{sc.primary_total}  "
              f"halluc={sc.hallucinated}  bonus={sc.bonus_matched}  passed={sc.passed}")
        assert sc.primary_matched == 20 and sc.bonus_matched == 5 and sc.hallucinated == 0

        # case 3: truncated to 5 lines (fail threshold of 8)
        truncated = "\n".join(t.primary_lines[:5])
        sc = score(t.name, t.primary_lines, t.bonus_lines, truncated)
        print(f"truncated5: matched={sc.primary_matched}/{sc.primary_total}  "
              f"halluc={sc.hallucinated}  passed={sc.passed}")
        assert sc.primary_matched == 5 and not sc.passed

        # case 4: barely-pass threshold (exactly 8)
        barely = "\n".join(t.primary_lines[:8])
        sc = score(t.name, t.primary_lines, t.bonus_lines, barely)
        assert sc.primary_matched == 8 and sc.passed

        # case 5: hallucinated lines mixed in
        mangled_lines = list(t.primary_lines[:10]) + ["    XYZ = 999  # fake"] * 3 + list(t.primary_lines[10:15])
        sc = score(t.name, t.primary_lines, t.bonus_lines, "\n".join(mangled_lines))
        print(f"mangled: matched={sc.primary_matched}/{sc.primary_total}  "
              f"halluc={sc.hallucinated}  passed={sc.passed}")
        assert sc.primary_matched == 15 and sc.hallucinated == 3 and sc.passed

        # case 6: wrapped in markdown fences
        fenced = "```python\n" + "\n".join(t.primary_lines) + "\n```"
        sc = score(t.name, t.primary_lines, t.bonus_lines, fenced)
        assert sc.primary_matched == 20 and sc.hallucinated == 0

        # case 7: blank lines in output should not count as hallucinations
        with_blanks = "\n\n".join(t.primary_lines)  # double-spaced
        sc = score(t.name, t.primary_lines, t.bonus_lines, with_blanks)
        assert sc.primary_matched == 20 and sc.hallucinated == 0, \
            f"blank-line handling: halluc={sc.hallucinated}"

        # case 8: two fenced blocks with prose between — prose must not be
        # scored as hallucinated lines (only fence contents count).
        two_fences = (
            "```python\n" + "\n".join(t.primary_lines[:10]) + "\n```\n"
            "And here are the remaining lines:\n"
            "```python\n" + "\n".join(t.primary_lines[10:]) + "\n```"
        )
        sc = score(t.name, t.primary_lines, t.bonus_lines, two_fences)
        assert sc.primary_matched == 20 and sc.hallucinated == 0, \
            f"multi-fence handling: matched={sc.primary_matched} halluc={sc.hallucinated}"

        # --- line classification ------------------------------------------
        py_src = (
            'def f():\n'
            '    """Doc line one.\n'
            '\n'
            '    Doc line two.\n'
            '    """\n'
            '    # a comment\n'
            '    x = 1  # trailing comment stays code\n'
            '\n'
            '    return x\n'
        )
        kinds = line_kinds(py_src, "py")
        expected_kinds = [
            KIND_CODE,       # def f():
            KIND_DOCSTRING,  # """Doc line one.
            KIND_BLANK,      # (blank inside docstring)
            KIND_DOCSTRING,  # Doc line two.
            KIND_DOCSTRING,  # """
            KIND_COMMENT,    # # a comment
            KIND_CODE,       # x = 1  # trailing
            KIND_BLANK,
            KIND_CODE,       # return x
        ]
        assert kinds == expected_kinds, f"py kinds: {kinds}"
        print("\npy line kinds: OK")

        js_src = (
            "function g() {\n"
            "\t// line comment\n"
            "\t/* block start\n"
            "\t   block middle\n"
            "\t   block end */\n"
            "\tvar y = 2; // trailing stays code\n"
            "\n"
            "\treturn y;\n"
            "}\n"
        )
        jk = line_kinds(js_src, "js")
        assert jk[1] == KIND_COMMENT, jk
        assert jk[2] == jk[3] == jk[4] == KIND_COMMENT, jk
        assert jk[5] == KIND_CODE, jk       # trailing comment must not blank out code
        assert jk[6] == KIND_BLANK, jk
        assert jk[7] == KIND_CODE, jk
        print("js line kinds: OK")

        # --- scoring policy ------------------------------------------------
        exp = ["    a = 1", "", "    # note", "    b = 2"]
        kinds4 = [KIND_CODE, KIND_BLANK, KIND_COMMENT, KIND_CODE]

        # Blank lines never earn credit: 3 eligible, not 4.
        sc = score("t", exp, [], "\n".join(exp), primary_kinds=kinds4)
        assert sc.primary_total == 3 and sc.primary_matched == 3, \
            f"blank excluded: {sc.primary_matched}/{sc.primary_total}"
        assert sc.blank_skipped == 1 and sc.code_total == 2 and sc.prose_total == 1

        # A blank-only "answer" earns nothing.
        sc = score("t", exp, [], "\n\n\n\n", primary_kinds=kinds4)
        assert sc.primary_matched == 0 and not sc.passed, \
            f"blank-only output scored {sc.primary_matched}"

        # --no-comments restricts the denominator to code.
        sc = score("t", exp, [], "\n".join(exp), primary_kinds=kinds4,
                   count_comments=False)
        assert sc.primary_total == 2 and sc.primary_matched == 2, \
            f"no-comments: {sc.primary_matched}/{sc.primary_total}"
        assert sc.prose_skipped == 1

        # A correct-but-ineligible line is neither credited nor hallucinated.
        sc = score("t", exp, [], "    # note", primary_kinds=kinds4,
                   count_comments=False)
        assert sc.primary_matched == 0 and sc.hallucinated == 0, \
            f"ignored line: matched={sc.primary_matched} halluc={sc.hallucinated}"

        # Ratio threshold equals the old 8/20 on an all-code window.
        all_code = [f"    line_{i}" for i in range(20)]
        k20 = [KIND_CODE] * 20
        sc = score("t", all_code, [], "\n".join(all_code[:8]), primary_kinds=k20)
        assert sc.primary_total == 20 and sc.primary_matched == 8 and sc.passed
        sc = score("t", all_code, [], "\n".join(all_code[:7]), primary_kinds=k20)
        assert sc.primary_matched == 7 and not sc.passed
        print(f"scoring policy (pass ratio {PASS_RATIO}): OK")

        print("\n✅ all smoke checks passed")
        return 0
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
