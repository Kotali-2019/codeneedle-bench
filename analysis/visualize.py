#!/usr/bin/env python3
"""Generate Plotly comparison dashboards from results/*.json.

Layout: one chart per page, grouped under a per-corpus subfolder.

    analysis/charts/
      index.html                          ← top-level: links per corpus
      <corpus>/
        index.html                        ← corpus dashboard with chart links
        leaderboard.html                  ← chart 1 standalone
        per-function.html                 ← chart 2 standalone
        recall-vs-position.html           ← chart 3 standalone

Each chart sizes itself to the data and reserves enough room for a vertical
legend with up to 20 model entries. Every chart is fully interactive — hover,
zoom, pan, click-to-toggle-trace, double-click-to-isolate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# This file lives in analysis/, so REPO_ROOT is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `import bench…` works regardless of cwd

from bench.scorer import PASS_RATIO  # noqa: E402 — needs the sys.path insert above
from bench.generation import generation_problem  # noqa: E402
from bench.scoring_policy import (  # noqa: E402
    DEFAULT_SCORING_POLICY, policy_from_dump,
)
from bench.textio import read_text, write_text, use_utf8_stdio  # noqa: E402

PASS_PCT = PASS_RATIO * 100
LEGEND_ROW_PX = 26    # how much vertical space each legend entry needs

# Stable color palette — assigned once per model so every chart uses the same color.
PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#ff9da6", "#9d755d", "#bab0ac", "#b279a2", "#eeca3b",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@dataclass
class Run:
    path: Path
    model: str
    group_name: str
    data: dict
    group_max: int = 0   # most queries any run in this group has; set by load_runs

    @property
    def n_queries(self) -> int:
        return len(self.data.get("results", []))

    @property
    def complete(self) -> bool:
        """Whether this run finished every query it planned.

        Dumps from schema_version >= 2 say so directly. Older dumps have no
        such field, so fall back to comparing against the busiest run in the
        same corpus group — a run with fewer queries than its peers was cut
        short, and must not be charted as if it were a full result.
        """
        if self.data.get("in_progress"):
            return False
        if "complete" in self.data:
            return bool(self.data["complete"])
        return self.group_max == 0 or self.n_queries >= self.group_max

    @property
    def label(self) -> str:
        """Model name, flagged when the run didn't finish every query."""
        if self.complete:
            return self.model
        planned = self.data.get("queries_planned") or self.group_max or "?"
        return f"{self.model} ⚠ INCOMPLETE {self.n_queries}/{planned}"


def _group_name(data: dict) -> str:
    """Corpus group plus a suffix when its scoring policy is non-default.

    A scoring override creates a separate dashboard instead of silently adding
    an easier or harder trace to the default leaderboard.
    """
    files = data.get("files") or ([data["source"]] if data.get("source") else [])
    if not files:
        base = "unknown"
    elif len(files) == 1:
        base = Path(files[0]).stem
    else:
        base = "+".join(Path(f).stem for f in files[:3])

    policy = policy_from_dump(data)
    if policy == DEFAULT_SCORING_POLICY:
        return base
    return f"{base}__{policy.slug}"


def _label_from_config(result_path: Path) -> str | None:
    """Recover a display label for a dump written before `model_label` existed.

    Result files are named `<corpus>__<model-config-stem>.json`, so the model
    config can be located from the filename and its `label` read directly.
    """
    stem = result_path.stem
    if "__" not in stem:
        return None
    model_stem = stem.split("__", 1)[1]
    try:
        import tomllib

        cfg = REPO_ROOT / "configs" / "models" / f"{model_stem}.toml"
        if not cfg.is_file():
            return None
        return tomllib.loads(read_text(cfg)).get("label")
    except Exception:
        return None


def load_runs(results_dir: Path) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(read_text(p))
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        if not data.get("results"):
            continue
        problem = generation_problem(data)
        if problem:
            print(
                f"skip {p.name}: incompatible benchmark generation ({problem}); "
                "re-run this corpus/model with the current benchmark",
                file=sys.stderr,
            )
            continue
        try:
            group = _group_name(data)
        except ValueError as e:
            print(f"skip {p.name}: incompatible scoring policy ({e})", file=sys.stderr)
            continue
        # Prefer the config's display label: raw server ids can misdescribe a
        # build (an MLX 4-bit registered as plain "qwen3.6-27b", for instance).
        name = (
            data.get("model_label")
            or _label_from_config(p)
            or data.get("model")
            or p.stem
        )
        groups[group].append(Run(path=p, model=name, group_name=group, data=data))

    # Now that every run in a group is known, record the busiest one so legacy
    # dumps (no `complete` field) can still be recognized as cut short.
    for runs in groups.values():
        peak = max((r.n_queries for r in runs), default=0)
        for r in runs:
            r.group_max = peak
    return groups


def resolve_line_positions(runs: list[Run]) -> dict[str, int]:
    """Map function name → start_line. Re-extracts from source so the depth
    chart works for old dumps too. Looks up files by basename under fixtures/
    if the original absolute path no longer exists (e.g. after a repo rename).
    """
    from bench.extract import extract as bench_extract

    fixtures_dirs = [REPO_ROOT / "fixtures"]
    name_to_line: dict[str, int] = {}
    tried: set[Path] = set()
    for run in runs:
        for raw in run.data.get("files") or ([run.data.get("source")] if run.data.get("source") else []):
            if not raw:
                continue
            p = Path(raw)
            if not p.exists():
                for d in fixtures_dirs:
                    alt = d / p.name
                    if alt.exists():
                        p = alt
                        break
                else:
                    continue
            if p in tried:
                continue
            tried.add(p)
            try:
                for t in bench_extract(p):
                    name_to_line.setdefault(t.name, t.start_line)
            except Exception as e:
                print(f"  (couldn't re-extract {p.name} for depth chart: {e})", file=sys.stderr)
    return name_to_line


def assign_colors(runs: list[Run]) -> dict[str, str]:
    models = sorted({r.model for r in runs})
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _legend_kwargs() -> dict:
    """Right-side vertical legend, padded box, room for many entries."""
    return dict(
        orientation="v",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#ddd",
        borderwidth=1,
        font=dict(size=12),
    )


def _chart_height(*, content_rows: int, n_legend_entries: int, base: int = 420) -> int:
    """Pick a height tall enough for both the data rows and the legend.

    `content_rows` is the number of bars/lines/etc. shown vertically.
    `n_legend_entries` is the legend item count.
    """
    by_legend = LEGEND_ROW_PX * n_legend_entries + 120
    by_content = base + 20 * max(0, content_rows - 8)
    return max(base, by_legend, by_content)


# --- charts ---------------------------------------------------------------


def leaderboard(runs: list[Run], colors: dict[str, str]):
    """Horizontal bar chart, one trace per run (so each is independently
    toggleable from the legend). Sorted best → worst by percentage matched.

    Percentage, not absolute count: a run that fail-fasted after 4 of 16
    queries has a much smaller denominator, so comparing raw totals made an
    aborted run look like a terrible model rather than a broken one.
    """
    import plotly.graph_objects as go

    rows = []
    for r in runs:
        matched = sum(x.get("primary_matched", 0) for x in r.data["results"])
        total = sum(x.get("primary_total", 0) for x in r.data["results"])
        code_m = sum(x.get("code_matched", 0) for x in r.data["results"])
        code_t = sum(x.get("code_total", 0) for x in r.data["results"])
        passed = sum(1 for x in r.data["results"] if x.get("passed"))
        queries = len(r.data["results"])
        halluc = sum(x.get("hallucinated", 0) for x in r.data["results"])
        errored = sum(1 for x in r.data["results"] if x.get("error"))
        rows.append({
            "model": r.label, "stem": r.path.stem, "color_key": r.model,
            "matched": matched, "total": total,
            "pct": (matched / total * 100) if total else 0.0,
            "code_m": code_m, "code_t": code_t,
            "passed": passed, "queries": queries,
            "halluc": halluc, "errored": errored,
            "complete": r.complete,
        })
    rows.sort(key=lambda d: d["pct"], reverse=True)

    if not rows:
        return None

    fig = go.Figure()
    for row in rows:
        flag = "" if row["complete"] else " ⚠ INCOMPLETE"
        code_pct = f"{row['code_m']/row['code_t']*100:.0f}%" if row["code_t"] else "n/a"
        annotation = (
            f"{row['pct']:.0f}% ({row['matched']}/{row['total']}) · "
            f"code {code_pct} · "
            f"{row['passed']}/{row['queries']} pass · "
            f"{row['halluc']} halluc"
            + (f" · {row['errored']} err" if row['errored'] else "")
            + flag
        )
        hover = (
            f"<b>{row['model']}</b><br>"
            f"run: {row['stem']}<br>"
            f"scored lines: {row['matched']} / {row['total']} ({row['pct']:.1f}%)<br>"
            f"code lines: {row['code_m']} / {row['code_t']} ({code_pct})<br>"
            f"pass: {row['passed']} / {row['queries']}<br>"
            f"hallucinated: {row['halluc']}<br>"
            f"errored: {row['errored']}"
            + ("" if row["complete"] else "<br><b>⚠ run did not complete</b>")
        )
        fig.add_trace(go.Bar(
            x=[row["pct"]],
            y=[row["stem"]],
            orientation="h",
            name=row["model"],
            legendgroup=row["model"],
            text=[annotation],
            textposition="outside",
            marker_color=colors[row["color_key"]],
            marker_line_color="#c00" if not row["complete"] else "#fff",
            marker_line_width=3 if not row["complete"] else 1,
            hovertext=[hover],
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Leaderboard · % of scored lines matched (blank lines excluded)",
        xaxis=dict(title="% of scored lines matched", range=[0, 190]),
        yaxis=dict(autorange="reversed", automargin=True),
        height=_chart_height(content_rows=len(rows), n_legend_entries=len(rows)),
        margin=dict(l=20, r=40, t=70, b=60),
        legend=_legend_kwargs(),
        bargap=0.25,
    )
    return fig


def per_function_bars(runs: list[Run], colors: dict[str, str]):
    """Grouped bars: one bar per (function × run). Dashed line at pass threshold."""
    import plotly.graph_objects as go

    all_fns: set[str] = set()
    for r in runs:
        for x in r.data["results"]:
            all_fns.add(x["function"])

    def mean_score(fn: str) -> float:
        xs = []
        for r in runs:
            x = next((y for y in r.data["results"] if y["function"] == fn), None)
            if x and x.get("primary_total"):
                xs.append(x["primary_matched"] / x["primary_total"])
        return sum(xs) / len(xs) if xs else 0.0

    fns = sorted(all_fns, key=mean_score, reverse=True)
    if not fns:
        return None

    fig = go.Figure()
    for r in runs:
        y, custom = [], []
        for fn in fns:
            x = next((z for z in r.data["results"] if z["function"] == fn), None)
            if x is None or x.get("error"):
                y.append(None)
                custom.append([r.path.stem, "—"])
            else:
                total = x.get("primary_total") or 0
                # Percentage, because primary_total now varies per function
                # (blank lines are excluded from the denominator).
                y.append(x.get("primary_matched", 0) / total * 100 if total else 0)
                custom.append([r.path.stem,
                               f"{x.get('primary_matched', 0)}/{total}"])
        fig.add_bar(
            x=fns, y=y,
            name=r.label,
            legendgroup=r.label,
            marker_color=colors[r.model],
            customdata=custom,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "model: " + r.label + "<br>"
                "run: %{customdata[0]}<br>"
                "matched: %{customdata[1]} (%{y:.0f}%)<extra></extra>"
            ),
        )

    fig.add_hline(
        y=PASS_PCT, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_PCT:.0f}%)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Per-function score · bars above the dashed line passed",
        xaxis=dict(title="function (sorted by average difficulty)", tickangle=-40,
                   automargin=True),
        yaxis=dict(title="% of scored lines matched", range=[0, 108]),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=_chart_height(content_rows=len(fns), n_legend_entries=len(runs)),
        margin=dict(l=70, r=40, t=70, b=160),
        legend=_legend_kwargs(),
    )
    return fig


def recall_vs_depth(runs: list[Run], colors: dict[str, str], positions: dict[str, int]):
    """Scatter + line: X = function start line in source, Y = % matched."""
    import plotly.graph_objects as go

    fig = go.Figure()
    any_data = False
    max_line = 0
    for r in runs:
        pts = []
        for x in r.data["results"]:
            if x.get("error"):
                continue
            fn = x["function"]
            if fn not in positions:
                continue
            total = x.get("primary_total") or 0
            if not total:
                continue
            pct = x.get("primary_matched", 0) / total * 100
            pts.append((positions[fn], pct, fn, x.get("primary_matched", 0), total))
        if not pts:
            continue
        any_data = True
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        max_line = max(max_line, max(xs))
        ys = [p[1] for p in pts]
        hover = [
            f"<b>{p[2]}</b><br>line {p[0]:,}<br>"
            f"{p[3]}/{p[4]} matched ({p[1]:.0f}%)"
            f"<br>model: {r.label}<br>run: {r.path.stem}"
            for p in pts
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=r.label,
            legendgroup=r.label,
            # Dashed for an aborted run — too few points to read as a trend.
            line=dict(color=colors[r.model], width=2,
                      dash="solid" if r.complete else "dot"),
            marker=dict(size=10, color=colors[r.model], line=dict(color="#fff", width=1)),
            hovertext=hover, hoverinfo="text",
        ))

    if not any_data:
        return None

    fig.add_hline(
        y=PASS_PCT, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_PCT:.0f}%)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Recall vs. position in file · left = near top, right = deep",
        xaxis=dict(title="function start line (deeper in file →)",
                   range=[0, max_line * 1.05]),
        yaxis=dict(title="% primary lines matched", range=[-5, 108]),
        height=_chart_height(content_rows=8, n_legend_entries=len(runs), base=520),
        margin=dict(l=70, r=40, t=70, b=70),
        legend=_legend_kwargs(),
    )
    return fig


# --- HTML assembly --------------------------------------------------------


PAGE_CSS = """
  *{box-sizing:border-box;}
  body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:1.5rem 1.25rem;color:#222;
       background:#fafafa;min-height:100vh;}
  .wrap{max-width:1500px;margin:0 auto;}
  header{font-size:.9rem;color:#666;margin-bottom:.5rem;}
  header a{color:#4c78a8;text-decoration:none;}
  header a:hover{text-decoration:underline;}
  header .corpus{font-weight:600;color:#222;}
  nav{margin:.25rem 0 1.5rem 0;font-size:.95rem;border-bottom:1px solid #e5e5e5;padding-bottom:.5rem;}
  nav a{color:#4c78a8;text-decoration:none;margin-right:1rem;padding:.25rem 0;display:inline-block;}
  nav a.active{color:#222;font-weight:600;border-bottom:2px solid #4c78a8;}
  nav a:hover{text-decoration:underline;}
  h1{margin:.25rem 0;font-size:1.5rem;}
  p.caption{color:#555;margin:.25rem 0 1rem 0;font-size:.95rem;line-height:1.5;}
  .chart{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:.5rem;
         box-shadow:0 1px 3px rgba(0,0,0,.04);overflow-x:auto;}
  ul{padding-left:1.25rem;}
  li{margin:.4rem 0;}
  small{color:#888;}
"""


CHART_PAGES = [
    # (slug, title, caption, chart_fn_key)
    ("leaderboard", "Leaderboard",
     "Total primary lines matched across all tested functions, sorted so the top bar is the best run. "
     "Each model has its own legend entry — click to hide/show, double-click to isolate. "
     "`halluc` = lines the model emitted that don't match the expected window.",
     "leaderboard"),
    ("per-function", "Per-function score",
     "One bar per model for each function, sorted left-to-right easiest → hardest. "
     "Bars above the dashed line passed (≥ 8 of 20 primary lines matched). "
     "Toggle a model in the legend to remove it from every cluster.",
     "per_function"),
    ("recall-vs-position", "Recall vs. position in file",
     "Each marker is a function placed at its line number in the source. "
     "If recall falls off as x increases, the model is losing context as depth grows — the "
     "core finding for sliding-window models. Hover any marker for details.",
     "recall_vs_position"),
]


def write_chart_page(out_path: Path, group: str, slug: str, title: str, caption: str,
                     fig, all_pages: list[tuple[str, str]]) -> None:
    import plotly.io as pio

    # Nav between charts of this corpus.
    nav_links = " ".join(
        f'<a href="{s}.html" class="{"active" if s == slug else ""}">{t}</a>'
        for s, t in all_pages
    )

    # Reference a local plotly.min.js (written once per corpus dir by
    # write_dashboard) so charts render offline — no CDN dependency.
    chart_html = pio.to_html(
        fig,
        include_plotlyjs="plotly.min.js",
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )

    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a> · '
        f'<span class="corpus">{group}</span></header>'
        f'<nav>{nav_links}</nav>'
        f'<h1>{title}</h1>'
        f'<p class="caption">{caption}</p>'
        f'<div class="chart">{chart_html}</div>'
        f'</div>'
    )
    write_text(
        out_path,
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{group} · {title}</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )


def write_corpus_index(out_path: Path, group: str, runs: list[Run],
                       generated_pages: list[tuple[str, str]]) -> None:
    models = sorted({r.model for r in runs})
    queries = sum(len(r.data["results"]) for r in runs)
    policy = policy_from_dump(runs[0].data)
    items = "".join(
        f'<li><a href="{slug}.html">{title}</a></li>'
        for slug, title in generated_pages
    )
    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a></header>'
        f'<h1>{group}</h1>'
        f'<p class="caption">{len(runs)} run(s) · {queries} queries · '
        f'{len(models)} unique model(s): {", ".join(models)}<br>'
        f'scoring: {policy.description}</p>'
        f'<ul>{items}</ul>'
        f'</div>'
    )
    write_text(
        out_path,
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{group} · charts</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )


def write_dashboard(group: str, runs: list[Run], out_dir: Path) -> list[tuple[str, str]]:
    """Write all chart pages for one corpus. Returns list of (slug, title) actually generated."""
    colors = assign_colors(runs)
    positions = resolve_line_positions(runs)

    figs = {
        "leaderboard": leaderboard(runs, colors),
        "per_function": per_function_bars(runs, colors),
        "recall_vs_position": recall_vs_depth(runs, colors, positions),
    }

    chart_dir = out_dir / group
    chart_dir.mkdir(parents=True, exist_ok=True)

    # Bundle plotly.js next to the pages so they work offline (~4MB, gitignored).
    plotly_js = chart_dir / "plotly.min.js"
    if not plotly_js.exists():
        from plotly.offline import get_plotlyjs

        write_text(plotly_js, get_plotlyjs())

    generated: list[tuple[str, str]] = []
    nav_pages: list[tuple[str, str]] = []
    for slug, title, _caption, fig_key in CHART_PAGES:
        if figs.get(fig_key) is not None:
            nav_pages.append((slug, title))

    for slug, title, caption, fig_key in CHART_PAGES:
        fig = figs.get(fig_key)
        if fig is None:
            continue
        page_path = chart_dir / f"{slug}.html"
        write_chart_page(page_path, group, slug, title, caption, fig, nav_pages)
        generated.append((slug, title))

    write_corpus_index(chart_dir / "index.html", group, runs, generated)
    return generated


def write_top_index(groups: dict[str, list[Run]], out_dir: Path) -> Path:
    idx = out_dir / "index.html"
    items = []
    for name in sorted(groups):
        runs = groups[name]
        models = sorted({r.model for r in runs})
        items.append(
            f'<li><a href="{name}/index.html">{name}</a> '
            f'<small>— {len(runs)} run(s), {len(models)} model(s): {", ".join(models)}</small></li>'
        )
    body = (
        f'<div class="wrap">'
        f'<h1>codeneedle · benchmark dashboards</h1>'
        f'<ul>{"".join(items)}</ul>'
        f'</div>'
    )
    write_text(
        idx,
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>codeneedle dashboards</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: analysis/charts/")
    args = ap.parse_args(argv)
    use_utf8_stdio()

    out_dir = args.output_dir or (REPO_ROOT / "analysis" / "charts")
    groups = load_runs(args.results_dir)
    if not groups:
        print(f"no usable result JSON files in {args.results_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = sum(len(r) for r in groups.values())
    print(f"Loaded {total_runs} run(s) in {len(groups)} group(s)")
    for name, runs in sorted(groups.items()):
        generated = write_dashboard(name, runs, out_dir)
        slugs = ", ".join(s for s, _ in generated)
        print(f"  {name}: {len(runs)} run(s) → {out_dir / name}/{{ {slugs} }}.html")

    idx = write_top_index(groups, out_dir)
    print(f"\nopen {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
