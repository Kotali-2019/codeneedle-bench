"""Measure what the signature anchor is worth: OLD vs NEW prompt, paired.

Reproduces the measurement behind the anchor fix. The old prompt pointed at
`function <name>(`, which does not exist for property- or assignment-style
definitions; the new one quotes the target's real signature.

    uv run python analysis/ab_anchor.py <model-config> [out.json]

Two groups run against the same model, same corpus, same scorer — the prompt
is the only variable:

  broken   5 jQuery targets with no `function <name>(` text in the file
  control  4 targets whose old anchor was already valid

Design notes, learned the hard way:
  * an empty `content` is NOT a zero score. Reasoning models can spend the
    whole token budget in `reasoning_content` and return nothing, which looks
    identical to total recall failure. This aborts instead of scoring it.
  * raw responses are saved, so any surprising number can be explained.
  * scores are reported strict AND relaxed. Strict conflates recall with
    indentation; relaxed isolates content.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.client import chat_complete
from bench.config import load_corpus, load_model
from bench.extract import load_source_glob
from bench.runner import NO_THINK_SUFFIX, _build_prompt
from bench.scorer import score
from bench.textio import write_text_atomic

OLD_T = ("{file_contents}\n\n---\n\n"
 "Task: reproduce verbatim the first {n} lines of the body of the function named "
 "`{name}` from the source above — i.e., the {n} lines {anchor}.\n\n"
 "Rules:\n- Output ONLY those lines, one per line, in original order.\n"
 "- Preserve original indentation and characters exactly.\n"
 "- Do NOT output the function signature or the line containing `{marker}`.\n"
 "- Do NOT add commentary, line numbers, or markdown code fences.\n"
 "- If there are blank lines in the body, include them as blank lines.\n{think}")
OLD_A = ("starting immediately after the line containing `function {name}(` or the "
         "assignment that introduces it (the line with the opening brace `{{`)")

BROKEN = ["PSEUDO", "init", "then", "val", "parseHTML"]
CONTROL = ["dataAttr", "buildFragment", "domManip", "propFilter"]

c = load_corpus("jquery")
s = load_source_glob(c.directory, c.glob, c.limit)
by = {t.name: t for t in s.targets}
MODEL = sys.argv[1] if len(sys.argv) > 1 else "spark-qwen38-27b"
cfg = load_model(MODEL)[0].client
OUT = sys.argv[2] if len(sys.argv) > 2 else "ab_results.json"
rows, t0 = [], time.time()

def old_prompt(t):
    return OLD_T.format(file_contents=s.text, name=t.name, n=len(t.primary_lines),
                        anchor=OLD_A.format(name=t.name),
                        marker=f"function {t.name}(", think=NO_THINK_SUFFIX)

def ask(t, prompt, arm):
    st = time.time()
    try:
        r = chat_complete(cfg, system=None, user=prompt)
    except Exception as e:
        sys.exit(f"ABORT {t.name}[{arm}]: {type(e).__name__}: {e}")
    # An empty body is NOT a zero score — it means the model never answered.
    if not r or not r.strip():
        sys.exit(f"ABORT {t.name}[{arm}]: empty content after {time.time()-st:.0f}s "
                 f"(reasoning model burning max_tokens? check prefill_no_think)")
    kw = dict(primary_kinds=t.primary_kinds, bonus_kinds=t.bonus_kinds)
    st_sc = score(t.name, t.primary_lines, t.bonus_lines, r, **kw)
    rx_sc = score(t.name, t.primary_lines, t.bonus_lines, r, relax_indent=True, **kw)
    return {"strict": round(st_sc.ratio*100), "relaxed": round(rx_sc.ratio*100),
            "matched": st_sc.primary_matched, "total": st_sc.primary_total,
            "halluc": st_sc.hallucinated, "reindent": st_sc.reindented,
            "secs": round(time.time()-st), "resp_lines": len(r.splitlines()),
            "response": r}

for group, names in (("broken", BROKEN), ("control", CONTROL)):
    for nm in names:
        t = by[nm]
        o = ask(t, old_prompt(t), "old")
        n = ask(t, _build_prompt(t, s.text, False, True), "new")
        rows.append({"group": group, "fn": nm, "old": o, "new": n})
        write_text_atomic(OUT, json.dumps(
            {"elapsed_min": round((time.time()-t0)/60,1), "rows": rows}, indent=2))
        print(f"{group:<8}{nm:<14} strict {o['strict']:>3}%->{n['strict']:>3}%   "
              f"relaxed {o['relaxed']:>3}%->{n['relaxed']:>3}%  ({o['secs']}s/{n['secs']}s)",
              flush=True)
print("DONE", flush=True)
