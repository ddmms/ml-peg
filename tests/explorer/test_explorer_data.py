"""Browser-free tests for the Model Explorer's data, scoring, figures and layout."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from dash import dcc
from dash.exceptions import PreventUpdate
import pytest

from ml_peg.analysis.explorer import build_explorer_data
from ml_peg.analysis.explorer.build_explorer_data import (
    _benchmark_element_index,
    _iter_parity_traces,
    _xyz_element_union,
    data_file_index,
    data_root,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.utils import explorer_data as explorer_data_module
from ml_peg.app.utils import explorer_layout, explorer_scoring
from ml_peg.app.utils.explorer_data import (
    CATEGORY_DOMAIN,
    _benchmark_lot_class,
    category_page_paths,
    category_titles,
    load_benchmark_scores,
    load_error_matrix,
    load_hardness,
    load_hardness_by_key,
    load_model_benchmark_scores,
    load_similarity,
    model_taxonomy,
    split_structure_key,
    structure_model_ranks,
)
from ml_peg.app.utils.explorer_figures import (
    _LABEL_PAD,
    FINDER_TOP,
    build_coverage_rose_figure,
    build_finder_figure,
    build_h2h_detail_figure,
    build_head_to_head_figure,
    build_report_profile_figure,
    build_similarity_heatmap_figure,
    build_structure_fails_figure,
    graph_style,
    render_fails_list,
    render_finder_picks,
    render_h2h_summary,
)
from ml_peg.app.utils.explorer_ids import (
    FAILS_PICK_TYPE,
    FINDER_EXPERIMENT_CATEGORIES,
    FINDER_GRAPH,
    FINDER_PICK_TYPE,
    H2H_DETAIL_GRAPH,
    STRUCT_FAILS_GRAPH,
)
from ml_peg.app.utils.explorer_layout import _a11y_graph
from ml_peg.app.utils.explorer_scoring import (
    H2H_TIE_BAND,
    MIN_FINDER_BENCH,
    benchmark_error_medians,
    cached_benchmark_error_medians,
    cached_finder_rankings,
    cached_head_to_head_stats,
    cached_structure_fails_ranking,
    h2h_area_options,
    h2h_pair_stats,
    h2h_verdict_counts,
    head_to_head_stats,
    model_finder_rankings,
    report_card_summary,
    representative_models,
    structure_fails_ranking,
)
from ml_peg.app.utils.register_explorer_callbacks import (
    _clicked_structure_key,
    _finder_pick_model,
)
from ml_peg.app.utils.utils import classify_level_of_theory

# --- Data build ----------------------------------------------------------------------


def test_iter_parity_traces_signed_residual() -> None:
    """A parity trace yields signed residuals and skips the y=x reference line."""
    figure = {
        "data": [
            {
                "mode": "markers",
                "name": "model-x",
                "x": [1.0, 2.0, None],
                "y": [1.5, 1.0, 3.0],
                "customdata": [["s1"], ["s2"], ["s3"]],
            },
            {"mode": "lines", "name": "y = x", "x": [0, 1], "y": [0, 1]},
        ]
    }
    traces = list(_iter_parity_traces(figure))
    assert len(traces) == 1
    model, ids, residuals = traces[0]
    assert model == "model-x"
    assert ids == ["s1", "s2"]  # the None point is dropped
    assert residuals == [-0.5, 1.0]


def test_xyz_element_union_reads_every_frame(tmp_path: Path) -> None:
    """The element set is the union over all frames of a multi-frame xyz file."""
    xyz = tmp_path / "multi.xyz"
    xyz.write_text(
        "2\nframe one\nC 0 0 0\nH 0 0 1\n3\nframe two\nO 0 0 0\nH 0 0 1\nN 1 0 0\n"
    )
    assert _xyz_element_union(xyz) == ["C", "H", "N", "O"]
    # Unreadable files resolve to None rather than raising.
    assert _xyz_element_union(tmp_path / "missing.xyz") is None


def test_data_file_index_finds_benchmark_files_once(
    benchmark_data: list[tuple[str, str]],
) -> None:
    """One walk finds every metrics table and figure, and no model-dir files."""
    index = data_file_index()
    tables = {
        (path.relative_to(data_root()).parts[0], path.parent.name)
        for path in index.metrics_tables
    }
    assert tables == set(benchmark_data)
    # The physicality benchmark ships a table but no parity figure.
    assert len(index.figures) == len(benchmark_data) - 1
    # Parity figures sit directly in benchmark directories, never inside a model's
    # output directory (the walk prunes below each benchmark).
    for figure in index.figures:
        rel = figure.relative_to(data_root()).parts
        assert (rel[0], "/".join(rel[1:-1])) in index.model_dirs
    assert len(index.model_dirs[("conformers", "confs")]) == 4  # one per model


def test_data_file_index_skips_excluded_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cost benchmarks such as inference speed never reach the Explorer."""
    for category, benchmark in [
        ("molecular_dynamics", "inference_speed"),
        ("molecular_dynamics", "stability"),
    ]:
        bench = tmp_path / category / benchmark
        bench.mkdir(parents=True)
        (bench / f"{benchmark}_metrics_table.json").write_text("{}")
        (bench / "figure_parity.json").write_text("{}")
    monkeypatch.setattr(build_explorer_data, "DATA_ROOT", tmp_path)
    data_file_index.cache_clear()
    try:
        index = data_file_index()
    finally:
        data_file_index.cache_clear()
    assert [path.parent.name for path in index.metrics_tables] == ["stability"]
    assert [path.parent.name for path in index.figures] == ["stability"]
    assert set(index.model_dirs) == {("molecular_dynamics", "stability")}


def test_compute_explorer_data_shape() -> None:
    """The on-load build produces aligned hardness rows, an error matrix, similarity."""
    rows = load_hardness()
    error_matrix = load_error_matrix()
    similarity = load_similarity()
    assert rows and error_matrix and similarity

    row = rows[0]
    for field in (
        "key",
        "sid",
        "category",
        "benchmark",
        "metric",
        "elements",
        "xyz",
        "median_err",
        "disagreement",
        "hardness_pct",
        "n_models",
        "worst_model",
    ):
        assert field in row
    assert 0.0 <= row["hardness_pct"] <= 1.0
    # Every hardness row has an error-matrix entry with the recorded model count.
    assert row["key"] in error_matrix
    assert len(error_matrix[row["key"]]) == row["n_models"]
    assert row["n_models"] >= build_explorer_data.MIN_MODELS

    corr = similarity["corr"]
    n = len(similarity["models"])
    assert len(corr) == n
    assert all(corr[i][i] == 1.0 for i in range(n)), "correlation diagonal must be 1"
    assert sorted(similarity["order"]) == list(range(n))


def test_elements_recovered_for_most_structures() -> None:
    """The resolver recovers elements for the large majority of structures."""
    rows = load_hardness()
    with_elements = sum(1 for row in rows if row["elements"])
    assert with_elements / len(rows) > 0.8


def test_element_index_prefers_info_json_and_falls_back_to_xyz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aligned info.json elements are used; benchmarks without them parse the .xyz."""
    fast = _benchmark_element_index("conformers", "confs")
    assert fast["confs_1"] == ["C", "H", "O"]
    stem, elements = "confs_1", fast["confs_1"]
    xyz = next((data_root() / "conformers" / "confs").rglob(f"{stem}.xyz"), None)
    assert xyz is not None
    assert _xyz_element_union(xyz) == elements  # the two sources agree
    # adsorption's info.json has no filenames list, so nothing is trusted from it.
    assert _benchmark_element_index("surfaces", "adsorption") == {}
    # Without a usable info.json the resolver falls back to parsing the geometry.
    monkeypatch.setattr(build_explorer_data, "_benchmark_element_index", lambda *_: {})
    path, parsed = build_explorer_data._resolve_structure("conformers", "confs", stem)
    assert path and path.endswith(f"{stem}.xyz")
    assert parsed == elements


def test_element_index_rejects_misaligned_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Element/filename lists of different lengths (multi-frame files) are ignored."""
    monkeypatch.setattr(
        build_explorer_data,
        "read_json",
        lambda _path: {"elements": [["C"], ["H"], ["O"]], "filenames": ["a", "b"]},
    )
    _benchmark_element_index.cache_clear()
    try:
        assert _benchmark_element_index("cat", "bench") == {}
        monkeypatch.setattr(
            build_explorer_data, "read_json", lambda _path: {"elements": ["C", "H"]}
        )
        _benchmark_element_index.cache_clear()
        assert _benchmark_element_index("cat", "union") == {}  # benchmark-wide union
    finally:
        _benchmark_element_index.cache_clear()


# --- Head-to-head --------------------------------------------------------------------


def _broadest_pair() -> tuple[str, str]:
    """Return the two most broadly-evaluated model names."""
    coverage: dict[str, int] = defaultdict(int)
    for resids in load_error_matrix().values():
        for model in resids:
            coverage[model] += 1
    ranked = sorted(coverage, key=coverage.get, reverse=True)
    return ranked[0], ranked[1]


def test_head_to_head_stats_and_verdict_bars() -> None:
    """Head-to-head wins are consistent and the verdict bars encode them."""
    model_a, model_b = _broadest_pair()
    error_matrix = load_error_matrix()
    stats = head_to_head_stats(model_a, model_b, error_matrix)
    assert stats["n_shared"] > 0
    assert stats["wins_a"] + stats["wins_b"] + stats["ties"] == stats["n_shared"]
    # Per-category win counts sum to the overall shared count.
    assert sum(c["n"] for c in stats["per_category"]) == stats["n_shared"]

    figure = build_head_to_head_figure(model_a, model_b, stats)
    bars = figure.data[0]
    # One diverging bar per shared category, keyed by category for click-to-drill.
    assert sorted(bars.y) == sorted(c["category"] for c in stats["per_category"])
    # Bars encode win-rate margin around even (0.5): margin is within ±0.5.
    per_cat = {c["category"]: c for c in stats["per_category"]}
    for category, margin in zip(bars.y, bars.x, strict=True):
        assert margin == pytest.approx(per_cat[category]["win_rate_a"] - 0.5)


def test_value_labels_clear_bars_whiskers_and_ticks() -> None:
    """Value labels start past whiskers and mean ticks, with no gridline across them."""
    model_a, model_b = _broadest_pair()
    stats = head_to_head_stats(model_a, model_b, load_error_matrix())
    h2h = build_head_to_head_figure(model_a, model_b, stats)
    bars = h2h.data[0]
    ends = {
        category: margin + (plus if margin >= 0 else -minus)
        for category, margin, plus, minus in zip(
            bars.y, bars.x, bars.error_x.array, bars.error_x.arrayminus, strict=True
        )
    }
    labels = [trace for trace in h2h.data[1:] if trace.mode == "text"]
    placed = {y: x for trace in labels for y, x in zip(trace.y, trace.x, strict=True)}
    assert set(placed) == set(ends)
    for category, x in placed.items():
        pad = -_LABEL_PAD if ends[category] < 0 else _LABEL_PAD
        assert x == pytest.approx(ends[category] + pad)
    assert all(trace.hoverinfo == "skip" for trace in labels)
    # The axis shows titles while clicks still get the directory name
    assert list(h2h.layout.yaxis.tickvals) == list(bars.y)
    assert "Bulk Crystals" in h2h.layout.yaxis.ticktext

    model = model_a
    report = build_report_profile_figure(model)
    bar, ticks, text = report.data
    means = dict(zip(ticks.y, ticks.x, strict=True))
    for category, value, x in zip(bar.y, bar.x, text.x, strict=True):
        assert x == pytest.approx(max(value, means.get(category, value)) + _LABEL_PAD)

    finder = build_finder_figure(model_finder_rankings())
    fails = build_structure_fails_figure(
        structure_fails_ranking(load_hardness(), None, None, "hardest", top=4)
    )
    for figure in (h2h, report, finder, fails):
        assert figure.layout.yaxis.showgrid is False
        # The label trace must not summon an auto legend
        assert figure.layout.showlegend is False


def test_head_to_head_detail_scatter() -> None:
    """The category drilldown scatter is clickable and matches the category count."""
    model_a, model_b = _broadest_pair()
    error_matrix = load_error_matrix()
    stats = head_to_head_stats(model_a, model_b, error_matrix)
    category = stats["per_category"][0]["category"]
    figure = build_h2h_detail_figure(
        model_a,
        model_b,
        category,
        error_matrix,
        load_hardness_by_key(),
        benchmark_error_medians(error_matrix),
    )
    points = figure.data[0]
    assert len(points.x) == stats["per_category"][0]["n"]
    # Each point carries [key, sid, benchmark]; the key feeds the 3D drilldown.
    assert all(len(cd) == 3 for cd in points.customdata)
    assert all(split_structure_key(cd[0])[0] == category for cd in points.customdata)
    assert all(cd[0] in error_matrix for cd in points.customdata)
    # Log axes with a y=x reference line as the second trace.
    assert figure.layout.xaxis.type == "log"
    assert figure.data[1].mode == "lines"


def test_head_to_head_pair_guards_and_messages() -> None:
    """Not-a-pair and disjoint pairs give placeholders that say why."""
    assert h2h_pair_stats(None, "x") is None
    assert h2h_pair_stats("same", "same") is None
    message = render_h2h_summary("same", "same", None)
    assert "two different models" in message
    assert not build_head_to_head_figure("same", "same", None).data

    disjoint = head_to_head_stats("only_a", "only_b", {"cat/b/m::s": {"only_a": 0.1}})
    assert disjoint["n_shared"] == 0
    assert "share no structures" in render_h2h_summary("only_a", "only_b", disjoint)
    assert not build_head_to_head_figure("only_a", "only_b", disjoint).data

    model_a, model_b = _broadest_pair()
    stats = h2h_pair_stats(model_a, model_b)
    headline = render_h2h_summary(model_a, model_b, stats)
    text = "".join(str(span.children) for span in headline.children)
    assert "leads in" in text and model_a in text and model_b in text


def test_h2h_area_options_list_shared_areas() -> None:
    """The drilldown picker offers only the areas both models were run on."""
    stats = head_to_head_stats(
        "A",
        "B",
        {
            "surfaces/s/m::0": {"A": 0.1, "B": 0.2},
            "bulk_crystal/b/m::0": {"A": 0.1, "B": 0.2},
            "molecular/x/m::0": {"A": 0.1},  # B never ran here
        },
    )
    assert h2h_area_options(stats) == [
        {"label": "bulk_crystal", "value": "bulk_crystal"},
        {"label": "surfaces", "value": "surfaces"},
    ]
    assert h2h_area_options(None) == []


def test_h2h_verdict_counts_follow_the_tie_band() -> None:
    """The headline's lead/too-close counts use the same band as the bar colours."""
    stats = {
        "per_category": [
            {"win_rate_a": 0.5 + 2 * H2H_TIE_BAND},  # A leads
            {"win_rate_a": 0.5 + H2H_TIE_BAND / 2},  # inside the band
            {"win_rate_a": 0.5 - 2 * H2H_TIE_BAND},  # B leads
            {"win_rate_a": 0.5},  # dead even
        ]
    }
    assert h2h_verdict_counts(stats) == (1, 1, 2)


def test_head_to_head_equal_weight_across_benchmarks() -> None:
    """A high-volume benchmark cannot dominate the verdict; ties leave the denominator.

    In one category, benchmark ``big`` (100 structures) is a clean win for A and
    benchmark ``small`` (2 structures) a clean win for B. Pooling raw structures would
    read as a ~98% A win; weighting each benchmark equally makes it an even 50%.
    """
    error_matrix: dict[str, dict[str, float]] = {}
    for i in range(100):
        error_matrix[f"cat/big/mae::{i}"] = {"A": 0.1, "B": 0.2}  # A closer
    for i in range(2):
        error_matrix[f"cat/small/mae::{i}"] = {"A": 0.2, "B": 0.1}  # B closer
    error_matrix["cat/tie/mae::0"] = {"A": 0.3, "B": 0.3}  # exact tie

    stats = head_to_head_stats("A", "B", error_matrix)
    # Overall and the single category both average the two decisive benchmarks (1, 0).
    assert stats["overall_win_rate_a"] == pytest.approx(0.5)
    (row,) = stats["per_category"]
    assert row["win_rate_a"] == pytest.approx(0.5)
    # Raw counts are retained for context; the tie is counted but not in the win rate.
    assert (row["wins_a"], row["wins_b"], row["ties"]) == (100, 2, 1)
    assert row["decisive"] == 102 and row["n_benchmarks"] == 2
    assert stats["n_shared"] == 103


def test_head_to_head_win_rate_symmetric_in_ties() -> None:
    """Excluding ties keeps the win rate symmetric: rate(A) + rate(B) == 1."""
    error_matrix = {
        "cat/b/mae::0": {"A": 0.1, "B": 0.2},  # A wins
        "cat/b/mae::1": {"A": 0.2, "B": 0.1},  # B wins
        "cat/b/mae::2": {"A": 0.5, "B": 0.5},  # tie (must not skew either way)
    }
    rate_a = head_to_head_stats("A", "B", error_matrix)["overall_win_rate_a"]
    rate_b = head_to_head_stats("B", "A", error_matrix)["overall_win_rate_a"]
    assert rate_a == pytest.approx(0.5)
    assert rate_a + rate_b == pytest.approx(1.0)


def test_head_to_head_all_ties_reads_even() -> None:
    """A pair that ties on every shared structure reads 50/50, ties out of the rate."""
    error_matrix = {
        "cat/b/mae::0": {"A": 0.2, "B": 0.2},
        "cat/b/mae::1": {"A": 0.3, "B": 0.3},
    }
    stats = head_to_head_stats("A", "B", error_matrix)
    assert stats["wins_a"] == stats["wins_b"] == 0
    assert stats["ties"] == 2 and stats["n_shared"] == 2
    assert stats["overall_win_rate_a"] == pytest.approx(0.5)
    (row,) = stats["per_category"]
    assert row["decisive"] == 0 and row["win_rate_a"] == pytest.approx(0.5)
    # An all-tie comparison still yields a valid (non-empty) verdict figure.
    figure = build_head_to_head_figure("A", "B", stats)
    assert figure.data


def test_head_to_head_normalization_is_unitless() -> None:
    """Head-to-head axes are relative errors (structure error / benchmark median)."""
    error_matrix = load_error_matrix()
    medians = benchmark_error_medians(error_matrix)
    assert medians and all(m >= 0 for m in medians.values())
    # A benchmark median exists for every structure key's benchmark prefix.
    sample_key = next(iter(error_matrix))
    assert split_structure_key(sample_key)[1] in medians
    assert cached_benchmark_error_medians() == medians


def test_cached_head_to_head_stats_memoizes() -> None:
    """A model pair returns the same cached stats, equal to the direct computation."""
    model_a, model_b = _broadest_pair()
    first = cached_head_to_head_stats(model_a, model_b)
    assert cached_head_to_head_stats(model_a, model_b) is first
    # The wrapper matches head_to_head_stats over the production error matrix.
    assert first == head_to_head_stats(model_a, model_b, load_error_matrix())
    assert first["n_shared"] > 0


# --- Where models fail ---------------------------------------------------------------


def test_structure_fails_ranking_modes() -> None:
    """Difficulty, disagreement, and per-model ranking each order the rows correctly."""
    rows = load_hardness()
    assert rows

    hardest = structure_fails_ranking(rows, None, None, "hardest", top=10)
    assert 0 < len(hardest) <= 10
    scores = [r["fail_score"] for r in hardest]
    assert scores == sorted(scores, reverse=True)  # worst-first
    assert all(r["fail_score"] == r["fail_frac"] for r in hardest)

    contested = structure_fails_ranking(rows, None, None, "contested", top=10)
    c_scores = [r["fail_score"] for r in contested]
    assert c_scores == sorted(c_scores, reverse=True)
    assert all(r["fail_score"] == r["disagreement"] for r in contested)

    # Per-model mode ranks by the model's error over the median model's.
    model = next(iter(next(iter(load_error_matrix().values()))))
    per_model = structure_fails_ranking(rows, None, model, "hardest", top=10)
    assert per_model, "focus model should have failing structures"
    m_scores = [r["fail_score"] for r in per_model]
    assert m_scores == sorted(m_scores, reverse=True)
    assert all(0.0 <= r["model_rank"] <= 1.0 for r in per_model)

    # Category filtering restricts the rows to the requested area.
    category = rows[0]["category"]
    only = structure_fails_ranking(rows, [category], None, "hardest", top=50)
    assert only and all(r["category"] == category for r in only)


def test_hardest_ranking_is_the_share_of_models_failing() -> None:
    """A structure is hardest when most models have an outlier error on it.

    Every model's error is its scale times the structure's, so the scale cancels:
    a structure is an outlier for all models when its error exceeds 3x its
    benchmark's median (lattice 12 > 3 * 2.5, atomisation 30 > 3 * 3) and for none
    otherwise (adsorption 2 <= 3 * 1, confs 8 <= 3 * 5), so only two are listed.
    They tie on share and percentile, so the structure key orders them.
    """
    ranked = structure_fails_ranking(load_hardness(), None, None, "hardest", top=4)
    assert [(r["sid"], r["fail_score"]) for r in ranked] == [
        ("lattice_3", 1.0),
        ("atomisation_3", 1.0),
    ]


def test_field_rankings_keep_two_structures_per_benchmark() -> None:
    """No benchmark places more than two structures in a field-wide ranking."""
    ranked = structure_fails_ranking(load_hardness(), None, None, "contested", top=20)
    per_bench = Counter(r["benchmark"] for r in ranked)
    assert per_bench == dict.fromkeys(
        ("lattice", "adsorption", "confs", "atomisation"), 2
    )


def test_focus_model_ranking_is_error_over_median_model() -> None:
    """A focused model's score is its error over the median model error (1.05x)."""
    ranked = structure_fails_ranking(
        load_hardness(), None, "gamma-3", "hardest", top=20
    )
    assert len(ranked) == 16
    assert all(r["fail_score"] == pytest.approx(1.5 / 1.05) for r in ranked)
    assert all(r["model_rank"] == 1.0 for r in ranked)  # worst of four everywhere
    beta = structure_fails_ranking(load_hardness(), None, "beta-2", "hardest", top=1)
    assert beta[0]["fail_score"] == pytest.approx(0.8 / 1.05)
    assert beta[0]["model_rank"] == 0.0


def test_structure_fails_figure_axes() -> None:
    """Hardest plots the 0-1 failing share; a focused model gets a 1x reference."""
    ranked = structure_fails_ranking(load_hardness(), None, None, "hardest", top=4)
    figure = build_structure_fails_figure(ranked)
    assert list(figure.data[0].x) == [1.0, 1.0]
    assert list(figure.data[1].text) == ["4/4 models", "4/4 models"]
    assert figure.data[0].y[0] == "1. lattice_3 · lattice"
    assert not figure.layout.shapes
    focus = structure_fails_ranking(load_hardness(), None, "gamma-3", "hardest", top=3)
    (line,) = build_structure_fails_figure(focus, model="gamma-3").layout.shapes
    assert (line.x0, line.x1) == (1, 1)


def test_finder_structure_evidence_weights_benchmarks_equally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine structures in one benchmark count the same as one in another.

    A beats B on all nine ``big`` structures and loses the single ``small`` one.
    Pooled, A would stand at 0.9; per benchmark it is (1 + 0) / 2 = 0.5, and B too.
    """
    rows = [
        {
            "key": f"cat/big/m::{i}",
            "category": "cat",
            "benchmark": "big",
            "elements": ["X"],
        }
        for i in range(9)
    ] + [
        {
            "key": "cat/small/m::0",
            "category": "cat",
            "benchmark": "small",
            "elements": ["X"],
        }
    ]
    ranks = {row["key"]: {"A": 0.0, "B": 1.0} for row in rows[:9]}
    ranks["cat/small/m::0"] = {"A": 1.0, "B": 0.0}
    scores = [
        {
            "model": model,
            "category": "cat",
            "domain": "materials",
            "lot_class": "in_domain",
            "model_lot": "PBE",
            "score": 0.5,
        }
        for model in ("A", "B")
        for _ in range(MIN_FINDER_BENCH)
    ]
    monkeypatch.setattr(explorer_scoring, "load_hardness", lambda: rows)
    monkeypatch.setattr(explorer_scoring, "structure_model_ranks", lambda: ranks)
    monkeypatch.setattr(explorer_scoring, "load_model_benchmark_scores", lambda: scores)
    results = {r["model"]: r for r in model_finder_rankings(elements=["X"])}
    assert results["A"]["struct_score"] == pytest.approx(0.5)
    assert results["B"]["struct_score"] == pytest.approx(0.5)
    assert results["A"]["n_struct"] == results["B"]["n_struct"] == 10


def test_build_structure_fails_figure_and_list_carry_keys() -> None:
    """The fails bars and their keyboard list identify structures by key."""
    ranked = structure_fails_ranking(load_hardness(), None, None, "hardest", top=8)
    figure = build_structure_fails_figure(ranked)
    bars = figure.data[0]
    assert 0 < len(bars.y) <= 8
    # The key leads customdata (as in the head-to-head scatter), so both feed the one
    # show_structure handler; sid and benchmark follow for hover.
    key, sid, benchmark = bars.customdata[0][:3]
    assert key in load_error_matrix()
    _category, bench_prefix, parsed_sid = split_structure_key(key)
    assert parsed_sid == sid and bench_prefix.split("/")[1] == benchmark
    # The list offers the same structures as buttons carrying the key in their id.
    details = render_fails_list(ranked).children
    ids = [item.children.id for item in details.children[1].children]
    assert [i["key"] for i in ids] == [row["key"] for row in ranked]
    assert all(i["type"] == FAILS_PICK_TYPE for i in ids)
    # Empty selection yields a valid placeholder figure and no list, not a crash.
    empty = build_structure_fails_figure([])
    assert empty.layout.xaxis.visible is False
    assert render_fails_list([]).children is None


def test_cached_structure_fails_ranking_memoizes() -> None:
    """Identical controls return the same cached ranking, equal to the direct call."""
    first = cached_structure_fails_ranking(None, None, "hardest", 10)
    assert cached_structure_fails_ranking(None, None, "hardest", 10) is first
    assert first == structure_fails_ranking(load_hardness(), None, None, "hardest", 10)
    # A different sort key is a distinct cache entry.
    assert cached_structure_fails_ranking(None, None, "contested", 10) is not first


def test_clicked_structure_key_routing() -> None:
    """The detail-panel router resolves list buttons vs graph clicks and guards."""
    key = "cat/bench/metric::s1"
    list_trigger = {"type": FAILS_PICK_TYPE, "key": key}
    assert _clicked_structure_key(list_trigger, None, None, [0, 1]) == key
    with pytest.raises(PreventUpdate):  # list re-rendered, nothing clicked
        _clicked_structure_key(list_trigger, None, None, [0, 0])
    click = {"points": [{"customdata": [key, "s1", "bench"]}]}
    assert _clicked_structure_key(STRUCT_FAILS_GRAPH, None, click, []) == key
    assert _clicked_structure_key(H2H_DETAIL_GRAPH, click, None, []) == key
    with pytest.raises(PreventUpdate):  # a list button was removed, nothing clicked
        _clicked_structure_key(None, click, click, [])
    with pytest.raises(PreventUpdate):  # the y = x reference line has no customdata
        _clicked_structure_key(STRUCT_FAILS_GRAPH, None, {"points": [{}]}, [])
    with pytest.raises(PreventUpdate):
        _clicked_structure_key(STRUCT_FAILS_GRAPH, None, None, [])


# --- Who fails alike + coverage ------------------------------------------------------


def test_similarity_heatmap_shape_and_order() -> None:
    """The heatmap is a square matrix ordered by the clustering, with unique labels."""
    sim = load_similarity()
    models = sim.get("models", [])
    if len(models) < 2:
        pytest.skip("too few models with shared evaluations to compare")
    figure = build_similarity_heatmap_figure(sim)
    heat = figure.data[0]
    assert len(heat.x) == len(heat.y) == len(models)
    # Labels are unique (heatmap axes collapse duplicate category strings otherwise).
    assert len(set(heat.x)) == len(heat.x)
    # Axis order follows the payload's hierarchical-clustering leaf order.
    order = sim.get("order") or list(range(len(models)))
    assert list(heat.x) == [models[k] for k in order]
    # Diagonal is self-correlation (1.0).
    assert heat.z[0][0] == pytest.approx(1.0)
    # Too few models gives a placeholder rather than a crash.
    assert not build_similarity_heatmap_figure({"models": ["only"], "corr": []}).data


def test_benchmark_scores_derive_from_model_scores() -> None:
    """Each aggregated benchmark row is the mean of its per-model scores."""
    rows = load_benchmark_scores()
    assert rows, "no *_metrics_table.json files found under app/data"
    per_model = load_model_benchmark_scores()
    for row in rows:
        assert row["category"] and row["benchmark"]
        assert 0.0 <= row["mean_score"] <= 1.0
        scores = [
            r["score"]
            for r in per_model
            if (r["category"], r["benchmark"]) == (row["category"], row["benchmark"])
        ]
        assert row["n_models"] == len(scores) > 0
        assert row["mean_score"] == pytest.approx(sum(scores) / len(scores))


def test_benchmark_scores_skip_missing_model_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clear_explorer_caches
) -> None:
    """A NaN or null model ``Score`` is left out, not averaged in as NaN."""
    bench = tmp_path / "molecular" / "toy"
    bench.mkdir(parents=True)
    # Bare NaN token, as the analysis writes it, plus a null
    (bench / "toy_metrics_table.json").write_text(
        '{"data": [{"MLIP": "a", "Score": 0.2}, {"MLIP": "b", "Score": 0.6},'
        ' {"MLIP": "c", "Score": NaN}, {"MLIP": "d", "Score": null}]}'
    )
    monkeypatch.setattr(build_explorer_data, "DATA_ROOT", tmp_path)
    clear_explorer_caches()
    try:
        models = [row["model"] for row in load_model_benchmark_scores()]
        rows = load_benchmark_scores()
    finally:
        monkeypatch.undo()
        clear_explorer_caches()
    assert models == ["a", "b"]
    assert rows == [
        {
            "category": "molecular",
            "benchmark": "toy",
            "mean_score": pytest.approx(0.4),
            "n_models": 2,
        }
    ]


def test_coverage_rose_granularity() -> None:
    """The rose has one wedge per category, or one per benchmark, with aligned data."""
    rows = load_benchmark_scores()
    n_categories = len({r["category"] for r in rows})

    by_category = build_coverage_rose_figure(rows, "category")
    by_benchmark = build_coverage_rose_figure(rows, "benchmark")

    cat_trace = by_category.data[0]
    bench_trace = by_benchmark.data[0]
    assert len(cat_trace.r) == len(cat_trace.theta) == n_categories
    assert len(bench_trace.r) == len(bench_trace.theta) == len(rows)
    # Colour drives the red->green encoding, so it must align with the bar lengths.
    assert list(cat_trace.marker.color) == list(cat_trace.r)
    # No scores at all gives a placeholder rather than a blank canvas.
    assert not build_coverage_rose_figure([], "category").data


# --- Model finder --------------------------------------------------------------------


def test_model_finder_rankings() -> None:
    """The finder ranks broadly-covered models with both evidence streams."""
    all_results = model_finder_rankings()
    assert all_results
    fits = [r["fit"] for r in all_results]
    assert fits == sorted(fits, reverse=True)
    assert all(0.0 <= r["fit"] <= 1.0 for r in all_results)
    assert all(r["n_bench"] >= MIN_FINDER_BENCH for r in all_results)

    # Element evidence engages the structure stream (or flags its absence).
    rows = load_hardness()
    element = next(e for r in rows for e in (r["elements"] or []))
    with_elements = model_finder_rankings(elements=[element])
    assert with_elements
    assert any(r["n_struct"] > 0 for r in with_elements)
    for r in with_elements:
        assert r["no_element_evidence"] == (r["n_struct"] == 0)

    figure = build_finder_figure(all_results)
    assert len(figure.data[0].y) == min(FINDER_TOP, len(all_results))


def test_model_finder_tolerance_narrows_the_evidence() -> None:
    """Tightening the tolerance can only shrink the admissible evidence per model."""
    by_tolerance = {
        tol: {r["model"]: r["n_bench"] for r in model_finder_rankings(tolerance=tol)}
        for tol in ("any", "dft", "in_domain")
    }
    assert by_tolerance["any"]
    for narrower, wider in (("dft", "any"), ("in_domain", "dft")):
        assert set(by_tolerance[narrower]) <= set(by_tolerance[wider])
        for model, n_bench in by_tolerance[narrower].items():
            assert n_bench <= by_tolerance[wider][model]
    # Only lattice and adsorption are in-domain (PBE), one short of the floor, so
    # the strictest tolerance ranks nothing -- and says so.
    assert not by_tolerance["in_domain"]
    empty = build_finder_figure([])
    assert "enough evidence" in empty.layout.annotations[0].text


def test_finder_unknown_elements_fall_back_to_benchmark_only() -> None:
    """An element with no tested structures still ranks models on benchmark evidence."""
    results = model_finder_rankings(elements=["Xx"])  # no structure carries "Xx"
    assert results, "finder should still rank on benchmark evidence"
    for row in results:
        assert row["n_struct"] == 0
        assert row["struct_score"] is None
        assert row["no_element_evidence"] is True
        # With no structure evidence, fit is the (benchmark-only) adjusted mean.
        assert 0.0 <= row["fit"] <= 1.0


def test_finder_picks_are_buttons_keyed_by_model() -> None:
    """The top-pick cards are buttons (keyboard-operable) that carry their model."""
    results = model_finder_rankings()
    cards = render_finder_picks(results, top=2).children
    assert len(cards) == 2
    for card, row in zip(cards, results[:2], strict=True):
        assert card.to_plotly_json()["type"] == "Button"
        assert card.id == {"type": FINDER_PICK_TYPE, "model": row["model"]}
    # No results renders an empty container, not a crash.
    assert render_finder_picks([]).children is None


def test_cached_finder_rankings_memoizes() -> None:
    """Identical use-case controls return the same cached ranking list."""
    first = cached_finder_rankings(None, None, "any")
    assert cached_finder_rankings(None, None, "any") is first
    assert isinstance(first, list) and first, "ranking should be non-empty"
    assert cached_finder_rankings(("molecular",), None, "any") is not first


def test_finder_pick_model_routing() -> None:
    """The finder click router resolves cards vs bars and guards re-renders."""
    card_trigger = {"type": FINDER_PICK_TYPE, "model": "mace-mp-0a"}
    # A card click routes to the card's model.
    assert _finder_pick_model(card_trigger, None, [0, 1, 0]) == "mace-mp-0a"
    # All-zero click counts mean the cards merely re-rendered: no update.
    with pytest.raises(PreventUpdate):
        _finder_pick_model(card_trigger, None, [0, 0, 0])
    # A bar click routes to the bar's y value (the model name).
    bar_click = {"points": [{"y": "orb-v3"}]}
    assert _finder_pick_model(FINDER_GRAPH, bar_click, []) == "orb-v3"
    # An empty/missing click event is a no-op.
    with pytest.raises(PreventUpdate):
        _finder_pick_model(FINDER_GRAPH, None, [])
    # Dash re-fires the callback with no trigger when pick cards are removed; the
    # graph's stale clickData must not reopen the last-clicked model.
    with pytest.raises(PreventUpdate):
        _finder_pick_model(None, bar_click, [])


# --- Report card + taxonomy ----------------------------------------------------------


def test_structure_model_ranks_normalised() -> None:
    """Per-structure ranks span 0 (best) to 1 (worst) over the evaluated cohort."""
    ranks = structure_model_ranks()
    assert ranks
    key, per_model = next(iter(ranks.items()))
    values = sorted(per_model.values())
    assert values[0] == 0.0 and values[-1] == 1.0
    assert set(per_model) <= set(load_error_matrix()[key])


def test_category_page_paths_route_shape() -> None:
    """Every scored category maps to a main-app category route."""
    paths = category_page_paths()
    scored = {row["category"] for row in load_model_benchmark_scores()}
    assert set(paths) == scored
    assert all(path.startswith("/category/") for path in paths.values())
    # Titles (not directory names) drive the slug, matching build_app's routing.
    assert paths["bulk_crystal"] == "/category/bulk-crystals"
    assert category_titles()["bulk_crystal"] == "Bulk Crystals"


def test_category_maps_name_real_categories() -> None:
    """The domain map and the finder presets only name categories that exist."""
    real = {
        path.name
        for path in APP_ROOT.iterdir()
        if (path / f"{path.name}.yml").is_file()
    }
    assert set(CATEGORY_DOMAIN) == real, (
        "CATEGORY_DOMAIN in explorer_data.py must list exactly the category "
        "directories under ml_peg/app"
    )
    bundled = {c for cats in FINDER_EXPERIMENT_CATEGORIES.values() for c in cats}
    assert bundled <= real, "a finder preset names a category that does not exist"


def test_classify_level_of_theory() -> None:
    """The LoT classifier matches the table warning scheme."""
    assert classify_level_of_theory("PBE", "PBE") == "in_domain"
    assert classify_level_of_theory("PBE", "r2SCAN") == "dft"
    assert classify_level_of_theory("PBE", "DLPNO-CCSD(T)/CBS") == "high_level"
    assert classify_level_of_theory("PBE", "Experimental") == "experimental"
    assert classify_level_of_theory("PBE", None) == "in_domain"


def test_model_benchmark_scores_shape() -> None:
    """Per-model benchmark scores carry the score, levels of theory, and LoT class."""
    rows = load_model_benchmark_scores()
    assert rows
    row = rows[0]
    for field in (
        "model",
        "base_id",
        "category",
        "benchmark",
        "domain",
        "score",
        "model_lot",
        "ref_levels",
        "lot_class",
    ):
        assert field in row
    assert 0.0 <= row["score"] <= 1.0
    assert row["lot_class"] in {
        "in_domain",
        "dft",
        "high_level",
        "experimental",
        "none",
    }
    # Physicality benchmarks carry no reference level -> "none".
    phys = [r for r in rows if r["domain"] == "physicality"]
    assert phys and all(r["lot_class"] == "none" for r in phys)


def test_model_taxonomy_groups_variants() -> None:
    """Dispersion/head variants collapse under one architecture family."""
    tax = model_taxonomy()
    assert tax["variants"] and tax["families"]
    # A base model and its +D3 twin share a family.
    variants = tax["variants"]
    if "mace-mp-0a" in variants and "mace-mp-0a-D3" in variants:
        assert variants["mace-mp-0a"]["family"] == variants["mace-mp-0a-D3"]["family"]
    # Every family lists at least one model, ordered by coverage.
    assert all(fam["models"] for fam in tax["families"].values())


def test_report_card_summary_splits_in_domain_and_physicality() -> None:
    """An in-domain-tested model has in-domain tests and separate physicality."""
    rows = load_model_benchmark_scores()
    in_domain = {r["model"] for r in rows if r["lot_class"] == "in_domain"}
    physical = {r["model"] for r in rows if r["domain"] == "physicality"}
    model = sorted(in_domain & physical)[0]
    training = next(r["model_lot"] for r in rows if r["model"] == model)

    summary = report_card_summary(model)
    assert summary["training_lot"] == training
    assert summary["n_benchmarks"] > 0
    in_mean, in_n = summary["in_domain"]
    assert in_n > 0 and in_mean is not None
    assert summary["physicality"][1] > 0
    figure = build_report_profile_figure(model)
    assert len(figure.data[0].y) > 0
    # An unknown model gets a placeholder, not a blank canvas.
    assert not build_report_profile_figure("no-such-model").data


def test_representative_models_collapses_families() -> None:
    """Representatives keep one variant per family, fewer than the full list."""
    models = load_similarity()["models"]
    reps = representative_models(models)
    assert 0 < len(reps) < len(models)
    families = {model_taxonomy()["variants"].get(m, {}).get("family", m) for m in reps}
    assert len(families) == len(reps)  # one per family


def test_split_structure_key() -> None:
    """The key parser splits category, benchmark prefix, and sid; '/' in metric OK."""
    assert split_structure_key("surfaces/adsorb/energy::s1") == (
        "surfaces",
        "surfaces/adsorb/energy",
        "s1",
    )
    # A metric containing "/" stays part of the benchmark prefix.
    assert split_structure_key("molecular/conf/dE/atom::mol-7") == (
        "molecular",
        "molecular/conf/dE/atom",
        "mol-7",
    )


def test_benchmark_lot_class_clamps_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected classifier output degrades to the neutral 'none' style."""
    monkeypatch.setattr(
        explorer_data_module, "classify_level_of_theory", lambda *_: "bogus_class"
    )
    assert _benchmark_lot_class("PBE", {"energy": "PBE"}) == "none"
    # A benchmark with no reference level is still "none" (unchanged behaviour).
    assert _benchmark_lot_class("PBE", {}) == "none"


# --- Startup + layout ----------------------------------------------------------------


def test_warm_explorer_data_primes_all_loaders() -> None:
    """``warm_explorer_data`` primes the base data and every derived cached loader."""
    derived = (
        explorer_data_module.load_benchmark_scores,
        explorer_data_module.load_model_benchmark_scores,
        explorer_data_module.structure_model_ranks,
        explorer_data_module.category_page_paths,
        explorer_data_module.category_titles,
        explorer_data_module.model_taxonomy,
    )
    for fn in derived:
        fn.cache_clear()
    explorer_data_module.warm_explorer_data()
    assert explorer_data_module.explorer_data_ready()
    for fn in derived:
        assert fn.cache_info().currsize == 1, fn.__name__


def test_warm_explorer_data_logs_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A data failure at startup is logged, never raised, so the app still boots."""

    def boom() -> None:
        raise ValueError("bad figure")

    monkeypatch.setattr(explorer_data_module, "_ensure_data", boom)
    with caplog.at_level("ERROR", logger=explorer_data_module.__name__):
        explorer_data_module.warm_explorer_data()
    assert "could not be built" in caplog.text


def test_layout_guards_when_data_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page explains itself when the data was not built or is empty."""
    monkeypatch.setattr(explorer_layout, "explorer_data_ready", lambda: False)
    unbuilt = explorer_layout.build_explorer_layout()
    assert "server log" in unbuilt.children[1].children

    monkeypatch.setattr(explorer_layout, "load_hardness", lambda: [])
    empty = explorer_layout.build_explorer_body()
    assert "No benchmark data" in empty.children[1].children


def test_build_explorer_body_assembles_every_panel() -> None:
    """The page carries an anchor for every jump-nav entry."""
    body = explorer_layout.build_explorer_body()
    ids = {
        child.id for child in body.children if getattr(child, "id", None) is not None
    }
    assert {anchor for _, anchor in explorer_layout._EXPLORER_JUMP_NAV} <= ids


def test_a11y_graph_gives_charts_an_accessible_name() -> None:
    """Interactive graphs are wrapped in a labelled figure for screen readers."""
    wrapped = _a11y_graph(dcc.Graph(id="demo"), "Finder ranking chart")
    props = wrapped.to_plotly_json()["props"]
    assert props["role"] == "figure"
    assert props["aria-label"] == "Finder ranking chart"


def test_graph_style_follows_the_figure_height() -> None:
    """A graph wrapper is sized from its figure, so the two cannot disagree."""
    ranked = structure_fails_ranking(load_hardness(), None, None, "hardest", top=3)
    figure = build_structure_fails_figure(ranked)
    assert graph_style(figure)["height"] == f"min({int(figure.layout.height)}px, 78vh)"
