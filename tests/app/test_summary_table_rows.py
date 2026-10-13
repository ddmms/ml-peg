"""Unit tests for the initial rows of rendered summary tables."""

from __future__ import annotations

from dash.dash_table import DataTable

from ml_peg.app.utils.build_components import build_loading_summary_table


def _rendered_table(component) -> DataTable:  # noqa: ANN001
    """Return the single DataTable nested inside a layout component."""
    found = [c for c in component._traverse() if isinstance(c, DataTable)]
    assert len(found) == 1
    return found[0]


def test_loading_summary_table_omits_models_without_results() -> None:
    """The first paint hides models with no scores, without touching the source.

    The sync callbacks drop these rows only after mount, so rendering them first
    flashed a fully hatched-out table (every model) on categories where only one
    model has results. The source table's data still seeds the stores, which
    need every row, so it must stay unfiltered.
    """
    rows = [
        {"MLIP": "a", "Score": 0.5, "X Score": 0.5},
        {"MLIP": "b", "Score": "NaN", "X Score": "NaN"},
        {"MLIP": "c", "Score": 0.2, "X Score": 0.2},
    ]
    tooltips = [{"MLIP": "tip a"}, {"MLIP": "tip b"}, {"MLIP": "tip c"}]
    table = DataTable(id="unit-summary-table", data=rows, tooltip_data=tooltips)

    rendered = _rendered_table(build_loading_summary_table(table))

    assert [row["MLIP"] for row in rendered.data] == ["a", "c"]
    # Tooltips are matched to rows by index, so they must be filtered in step.
    assert rendered.tooltip_data == [{"MLIP": "tip a"}, {"MLIP": "tip c"}]
    assert table.data == rows, "source table rows (store seed) were modified"
