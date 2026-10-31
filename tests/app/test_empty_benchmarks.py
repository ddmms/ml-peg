"""Unit tests for benchmarks that produced no results."""

from __future__ import annotations

from dash.dash_table import DataTable
from dash.dcc import Store
from dash.html import Div

from ml_peg.app.utils.build_components import build_summary_table, build_test_layout
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names


def _benchmark_table(name: str, scores: dict[str, float] | None) -> DataTable:
    """Build a stand-in benchmark table, with no rows when `scores` is None."""
    data = (
        [{"MLIP": mlip, "Score": score} for mlip, score in scores.items()]
        if scores
        else []
    )
    table = DataTable(id=f"{name}-table", data=data)
    table.description = f"{name} description"
    return table


def test_summary_weights_ignore_benchmark_without_results() -> None:
    """A benchmark with no results must not NaN every score in its group.

    Its column is NaN for every model, and `calc_table_scores` treats any
    weighted NaN as fatal to the whole row, so the group score would otherwise
    be missing for every model whenever one benchmark failed to run.
    """
    mlip = next(iter(get_model_names(current_models)))
    tables = {
        "scored": _benchmark_table("scored", {mlip: 0.5}),
        "empty": _benchmark_table("empty", None),
    }

    table = build_summary_table(tables, table_id="unit-summary-table")

    row = next(row for row in table.data if row["MLIP"] == mlip)
    assert row["empty Score"] == "NaN"
    assert row["Score"] == 0.5
    assert table.weights == {"empty Score": 0.0}
    # Explicit weights are kept; only the empty benchmark is overridden.
    weighted = build_summary_table(
        tables, table_id="unit-summary-table-2", weights={"scored Score": 3.0}
    )
    assert weighted.weights == {"scored Score": 3.0, "empty Score": 0.0}


def test_summary_weights_untouched_when_all_benchmarks_have_results() -> None:
    """A benchmark with results keeps its weight, so scores stay a plain average."""
    mlip = next(iter(get_model_names(current_models)))
    tables = {
        "a": _benchmark_table("a", {mlip: 0.4}),
        "b": _benchmark_table("b", {mlip: 0.6}),
    }

    table = build_summary_table(tables, table_id="unit-summary-table-3")

    row = next(row for row in table.data if row["MLIP"] == mlip)
    assert row["Score"] == 0.5
    assert table.weights is None


def test_layout_without_results_drops_table_and_controls() -> None:
    """A result-less benchmark keeps its docs, but hides table, controls, plots."""
    table = _benchmark_table("empty", None)
    table.thresholds = {}
    table.tooltip_header = {}

    layout = build_test_layout(
        name="empty",
        description="A description",
        framework_ids=["ml_peg"],
        table=table,
        thresholds={},
        extra_components=[Div("plot", id="unit-plot")],
        docs_url="https://example.com/docs",
    )

    rendered = list(layout._traverse())
    assert not [node for node in rendered if isinstance(node, DataTable)]
    assert not [node for node in rendered if isinstance(node, Store)]
    assert not [node for node in rendered if getattr(node, "id", None) == "unit-plot"]
    # `_traverse` yields components, so read the text off their children.
    text = " ".join(
        node.children
        for node in rendered
        if isinstance(getattr(node, "children", None), str)
    )
    assert "Data is still being collected for this benchmark" in text
    assert "https://example.com/docs" in [
        getattr(node, "href", None) for node in rendered
    ]
