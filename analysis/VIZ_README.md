# Analysis & visualizations

```
analysis/
  visualize.py       Plotly dashboard builder
  charts/            generated HTML output (gitignored)
  VIZ_README.md      this file
```

## Run it

```
python3 analysis/visualize.py
```

Reads every `results/*.json`, groups runs by their corpus (the `files` field
in each dump), and writes **one chart per HTML page** under
`analysis/charts/<corpus>/`:

```
analysis/charts/
  index.html                      ← top-level: links per corpus
  http_server/
    index.html                    ← corpus dashboard with chart links
    leaderboard.html              ← chart 1 standalone
    per-function.html             ← chart 2 standalone
    recall-vs-position.html       ← chart 3 standalone
  jquery/
    …same shape…
```

Open with:

```
open analysis/charts/index.html
```

Each chart page is fully self-contained — `plotly.min.js` is written into each
corpus folder, so the pages render offline with no CDN. Pages share
a small horizontal nav at the top so you can jump between charts of the same
corpus without going back to the index.

### Options

```
python3 analysis/visualize.py --results-dir results --output-dir analysis/charts
```

- `--results-dir` (default `results/`) — where to find result JSONs
- `--output-dir`  (default `analysis/charts/`) — where to write HTML

Re-run any time. It overwrites the HTML; nothing is incremental.

Only results carrying the current `benchmark_generation.prompt` and
`benchmark_generation.scorer` values are loaded. Older or unversioned dumps
are skipped with a diagnostic instead of being mixed into a current
leaderboard. Run `python3 run-missing.py` to identify and regenerate those
stale corpus/model combinations.

## What each chart shows

Three charts per corpus, one chart per page. Models get a stable color across
every chart of a corpus, and **every chart is fully interactive**:

- hover over any data point for details
- click a model in the legend to hide/show it
- double-click a model to isolate it
- box-select / lasso to zoom; double-click empty space to reset
- the legend is on the right side and sized so up to 20 models fit cleanly

### 1. Leaderboard

Horizontal bars sort valid runs by the percentage of primary lines matched.
Errored queries are excluded from quality denominators because transport,
context, and empty-response failures are not recall misses. Any run containing
an error remains visible for diagnosis, outlined in red, marked **INELIGIBLE**,
and sorted after valid runs. `run-missing.py` schedules it for rerunning.

Annotation on each bar: `<percent> (<matched>/<total>) · <P>/<N> pass · <H> halluc`.

This is the *who won* view. If you're glancing at one number, this is it.

### 2. Per-function score

One bar per (model × function), grouped by function and sorted left-to-right
from easiest (highest cross-model average) to hardest. A horizontal dashed
line marks the pass threshold (8/20).

This is the *where do models diverge* view. Functions where every model gets
20/20 are uninteresting; the spread on the right side of the chart is where
the benchmark is doing real work.

### 3. Recall vs. position in file

Each marker is a function placed at its **start line in the source**.
Connected with a line per model. Y-axis is `% primary lines matched`.

This is the chart that tests the video's core thesis: **does recall fall off
as functions appear deeper in context?**

How to read it:

| pattern | interpretation |
|---|---|
| flat line near 100% | model is comfortable with the entire context length |
| flat line, **but lower** (e.g. 60%) | quality issue, not depth — model recalls some but not all, equally bad everywhere |
| downward slope from left to right | classic sliding-window collapse — model is forgetting earlier context |
| bouncy / no trend | individual functions are easier or harder; depth isn't the bottleneck |

**Important caveat.** This chart only stresses positional recall when the
corpus is big enough to push the model past its "easy" zone. `http_server` and
`jquery` are public code, so prior training may also inflate their scores. Use
`novel_16k` as the short-context baseline and `novel_64k` / `novel_128k` to
measure long-context decay without known-source memorization. Keep the public
corpora as smoke and real-world controls, not the headline leaderboard.

## Grouping rule

Runs are grouped by **corpus, primary-window length, and scoring policy**, not
model. The base grouping key is the set of file basenames in the dump's
`files` field. The default 20-line, strict/comments-counted policy keeps the
plain corpus name; other contracts receive descriptive suffixes. So:

- All runs against `fixtures/http_server.py` → `analysis/charts/http_server/`
- All runs against `fixtures/jquery.js` → `analysis/charts/jquery/`
- A run against `fixtures/foo.py + fixtures/bar.py` → its own group
- A 48-line jQuery run → `analysis/charts/jquery__48-lines/`
- Content-normalized jQuery runs →
  `analysis/charts/jquery__content-comments-no-blanks-pass-40pct/`

This prevents a model scored with relaxed indentation or code-only scoring
from appearing in the same leaderboard as strictly scored peers.

Within a dashboard, each run shows up as its own colored line / bar even if
it's the same model — re-running is a separate trace, distinguished in the
legend by the result-file stem (`http_server__qwen36-35b`).

## Data sources

The script reads:

- `results/*.json` — all the dumps `bench.py run` produced
- `fixtures/<file>` — re-extracted at viz time so the position chart can map
  function names → start lines (this is robust against repo renames)

The raw response is retained in every dump, but scoring it again cannot repair
a historical prompt difference. That is why unknown prompt generations are
excluded rather than inferred from `schema_version`.

It does *not* require the original `--corpus` config to render — only the
dump JSON and the fixture file. If the fixture moved, the position chart
silently drops the affected functions.

## Adding a new chart

The shape is in `analysis/visualize.py`:

1. Write a function `def my_chart(runs, colors)` that returns a `plotly.graph_objects.Figure`.
2. Append `("Section title", "caption shown under the chart", my_chart(runs, colors))` to the `sections` list inside `write_dashboard`.

The dashboard automatically renders the section, adds it to the top nav, and
shares the model-color palette with the existing charts.
