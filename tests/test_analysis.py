"""Chart generation and the run-missing completeness logic."""
from __future__ import annotations

import importlib.util
import json

import pytest
from bench.generation import current_generation
from bench.textio import read_text, write_text


@pytest.fixture(scope="module")
def rm(repo_root):
    """Import run-missing.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location(
        "run_missing", repo_root / "run-missing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(**over):
    d = {
        "files": ["fixtures/http_server.py"],
        "model": "m", "results": [{"function": "f", "passed": True,
                                   "primary_matched": 5, "primary_total": 10,
                                   "error": None}],
        "benchmark_generation": current_generation(),
    }
    d.update(over)
    return d


def _legacy_dump(**over):
    d = _dump(**over)
    d.pop("benchmark_generation")
    return d


# --- completeness detection ----------------------------------------------


def test_missing_file_is_not_done(rm, tmp_path):
    done, why = rm.result_state(tmp_path / "nope.json")
    assert not done and why == "missing"


def test_unreadable_file_is_not_done(rm, tmp_path):
    p = tmp_path / "bad.json"
    write_text(p, "{not json")
    done, why = rm.result_state(p)
    assert not done and "unreadable" in why


def test_complete_flag_true_is_done(rm, tmp_path):
    p = tmp_path / "d.json"
    write_text(p, json.dumps(_dump(complete=True)))
    assert rm.result_state(p) == (True, "complete")


def test_complete_flag_false_is_not_done(rm, tmp_path):
    p = tmp_path / "d.json"
    write_text(p, json.dumps(_dump(complete=False, queries_run=4,
                                  queries_planned=16)))
    done, why = rm.result_state(p)
    assert not done and "4/16" in why


def test_legacy_dump_all_errored_is_not_done(rm, tmp_path):
    p = tmp_path / "d.json"
    write_text(p, json.dumps(_legacy_dump(results=[
        {"function": "a", "error": "HTTP 400", "passed": False},
        {"function": "b", "error": "HTTP 400", "passed": False},
    ])))
    done, why = rm.result_state(p)
    assert not done and "every query errored" in why


def test_legacy_short_dump_detected_against_corpus(rm, tmp_path):
    """The real bug: a 4-of-16 legacy dump was treated as finished."""
    p = tmp_path / "d.json"
    write_text(p, json.dumps(_legacy_dump(results=[
        {"function": f"f{i}", "error": None, "passed": True} for i in range(4)
    ])))
    done, why = rm.result_state(p, "jquery")
    assert not done, "a short legacy dump must be re-run"
    assert "4/16" in why


def test_legacy_full_dump_is_stale(rm, tmp_path):
    p = tmp_path / "d.json"
    write_text(p, json.dumps(_legacy_dump(results=[
        {"function": f"f{i}", "error": None, "passed": True} for i in range(11)
    ])))
    done, why = rm.result_state(p, "http_server")
    assert done is False
    assert "stale benchmark generation" in why


def test_outdated_explicit_generation_is_stale(rm, tmp_path):
    p = tmp_path / "d.json"
    d = _dump(complete=True)
    d["benchmark_generation"]["prompt"] = "old-prompt"
    write_text(p, json.dumps(d))
    done, why = rm.result_state(p)
    assert done is False
    assert "prompt='old-prompt'" in why


def test_expected_queries_matches_real_corpora(rm):
    assert rm.expected_queries("http_server") == 11
    assert rm.expected_queries("jquery") == 16
    assert rm.expected_queries("no_such_corpus") is None


def test_shipped_partial_result_is_flagged(rm, repo_root):
    """The actual stale artifact in results/, if still present."""
    p = repo_root / "results" / "jquery__gemma-4-31b-bf16.json"
    if not p.is_file():
        pytest.skip("partial result already regenerated")
    done, why = rm.result_state(p, "jquery")
    assert not done, f"expected incomplete, got {why}"


# --- local/hosted detection ----------------------------------------------


@pytest.mark.parametrize("url,is_local", [
    ("http://localhost:1234", True),
    ("http://127.0.0.1:1234", True),
    ("http://0.0.0.0:8080", True),
    ("localhost:1234", True),          # scheme-less
    ("https://api.openai.com", False),
    ("https://api.anthropic.com", False),
    ("", False),                        # unparseable must not trigger lms load
])
def test_is_local_server(rm, url, is_local):
    assert rm.is_local_server(url) is is_local


# --- visualize ------------------------------------------------------------


@pytest.fixture
def viz(repo_root):
    import sys
    sys.path.insert(0, str(repo_root / "analysis"))
    import visualize
    return visualize


def _run(viz, tmp_path, name, n, total_per=10, matched_per=5, **over):
    data = {
        "files": ["fixtures/jquery.js"],
        "model": name,
        "benchmark_generation": current_generation(),
        "results": [
            {"function": f"fn{i}", "passed": True, "error": None,
             "primary_matched": matched_per, "primary_total": total_per,
             "code_matched": matched_per, "code_total": total_per,
             "hallucinated": 0, "bonus_matched": 0}
            for i in range(n)
        ],
    }
    data.update(over)
    p = tmp_path / f"jquery__{name}.json"
    write_text(p, json.dumps(data))
    return p


def test_incomplete_run_flagged_via_complete_field(viz, tmp_path):
    _run(viz, tmp_path, "full", 16)
    _run(viz, tmp_path, "short", 4, complete=False, queries_run=4,
         queries_planned=16)
    groups = viz.load_runs(tmp_path)
    labels = {r.model: r.label for r in groups["jquery"]}
    assert labels["full"] == "full"
    assert "INCOMPLETE 4/16" in labels["short"]


def test_unknown_generation_is_not_charted(viz, tmp_path, capsys):
    _run(viz, tmp_path, "full", 16)
    legacy = _run(viz, tmp_path, "legacyshort", 4)
    data = json.loads(read_text(legacy))
    data.pop("benchmark_generation")
    write_text(legacy, json.dumps(data))
    groups = viz.load_runs(tmp_path)
    by = {r.model: r for r in groups["jquery"]}
    assert by["full"].complete is True
    assert "legacyshort" not in by
    assert "incompatible benchmark generation" in capsys.readouterr().err


def test_outdated_explicit_generation_is_not_charted(viz, tmp_path, capsys):
    p = _run(viz, tmp_path, "old", 4)
    data = json.loads(read_text(p))
    data["benchmark_generation"]["scorer"] = "old-scorer"
    write_text(p, json.dumps(data))
    assert viz.load_runs(tmp_path) == {}
    assert "old-scorer" in capsys.readouterr().err


def test_leaderboard_ranks_by_percentage_not_volume(viz, tmp_path):
    """The core charting bug: an aborted run must not sort last on volume.

    `short` answered 4 questions at 80%; `full` answered 16 at 50%. Ranking by
    raw matched lines puts short (32) far below full (80) — by percentage,
    short correctly leads.
    """
    _run(viz, tmp_path, "full", 16, total_per=10, matched_per=5)
    _run(viz, tmp_path, "short", 4, total_per=10, matched_per=8,
         complete=False, queries_run=4, queries_planned=16)
    runs = viz.load_runs(tmp_path)["jquery"]
    fig = viz.leaderboard(runs, viz.assign_colors(runs))
    order = [t.name for t in fig.data]
    xs = {t.name: t.x[0] for t in fig.data}

    assert order[0].startswith("short"), "percentage must decide the ranking"
    assert xs[order[0]] == pytest.approx(80.0)
    assert xs["full"] == pytest.approx(50.0)


def test_leaderboard_outlines_incomplete_runs(viz, tmp_path):
    _run(viz, tmp_path, "full", 16)
    _run(viz, tmp_path, "short", 4, complete=False, queries_run=4,
         queries_planned=16)
    runs = viz.load_runs(tmp_path)["jquery"]
    fig = viz.leaderboard(runs, viz.assign_colors(runs))
    marks = {t.name: t.marker.line.color for t in fig.data}
    assert marks[[n for n in marks if "INCOMPLETE" in n][0]] == "#c00"


def test_per_function_chart_uses_percentage_axis(viz, tmp_path):
    _run(viz, tmp_path, "a", 4, total_per=10, matched_per=5)
    runs = viz.load_runs(tmp_path)["jquery"]
    fig = viz.per_function_bars(runs, viz.assign_colors(runs))
    assert fig.layout.yaxis.title.text == "% of scored lines matched"
    assert list(fig.data[0].y) == [50.0] * 4
    assert fig.layout.yaxis.range == (0, 108)


def test_charts_have_no_cdn_reference(viz, tmp_path, repo_root):
    _run(viz, tmp_path, "a", 4)
    out = tmp_path / "charts"
    groups = viz.load_runs(tmp_path)
    viz.write_dashboard("jquery", groups["jquery"], out)
    page = read_text(out / "jquery" / "leaderboard.html")
    assert "cdn.plot.ly" not in page, "charts must render offline"
    assert 'src="plotly.min.js"' in page
    assert (out / "jquery" / "plotly.min.js").is_file()


def test_model_label_preferred_over_server_id(viz, tmp_path):
    _run(viz, tmp_path, "qwen3.6-27b", 4, model_label="qwen3.6-27b MLX 4bit")
    runs = viz.load_runs(tmp_path)["jquery"]
    assert runs[0].model == "qwen3.6-27b MLX 4bit"


def test_label_recovered_from_config_for_legacy_dumps(viz, repo_root):
    """Dumps written before `model_label` still chart with the right name."""
    got = viz._label_from_config(
        repo_root / "results" / "jquery__qwen36-27b-mlx-4bit.json")
    assert got == "qwen3.6-27b MLX 4bit"
