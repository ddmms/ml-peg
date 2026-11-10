"""Regression checks for model identity through display ordering and cell clicks."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ml_peg.app.utils import build_callbacks
from ml_peg.app.utils.model_order import order_model_rows
from ml_peg.app.utils.utils import build_level_of_theory_warnings, filter_rows_by_models


@pytest.mark.parametrize("mode", ["alphabetical", "random"])
def test_rows_are_atomic_stable_and_filter_independent(mode):
    """Scores, missing values and extra keys stay with the same canonical model."""
    rows = [
        {"id": "z", "MLIP": "zeta", "Score": None, "extra": "only-z"},
        {"id": "a", "MLIP": "Alpha", "Score": 0.2},
        {"id": "b", "MLIP": "beta-D3", "Score": 0.8},
    ]
    original = deepcopy(rows)
    state = {"mode": mode, "seed": "fixed"}
    ordered = order_model_rows(rows, state)
    assert rows == original
    assert {row["id"]: row for row in ordered} == {row["id"]: row for row in rows}
    assert order_model_rows(list(reversed(rows)), state) == ordered
    for selected in ([], ["a"], ["a", "z"], None):
        assert order_model_rows(filter_rows_by_models(rows, selected), state) == (
            filter_rows_by_models(ordered, selected)
        )
    if mode == "alphabetical":
        assert [row["id"] for row in ordered] == ["a", "b", "z"]
    _, tooltips = build_level_of_theory_warnings(ordered, {}, {}, {})
    for row, tooltip in zip(ordered, tooltips, strict=True):
        assert row["MLIP"] in tooltip["MLIP"]["value"]


def test_seeds_change_order_without_changing_model_identity():
    """New seeds randomize, and aliases sharing an ID have the same relative rank."""
    rows = [{"id": str(i), "MLIP": f"model-{i}"} for i in range(50)]
    one = order_model_rows(rows, {"mode": "random", "seed": "one"})
    two = order_model_rows(rows, {"mode": "random", "seed": "two"})
    assert one != two
    assert {r["id"] for r in one} == {r["id"] for r in two}
    aliases = [{**r, "MLIP": r["MLIP"] + "-D3"} for r in rows]
    assert [r["id"] for r in one] == [
        r["id"] for r in order_model_rows(aliases, {"mode": "random", "seed": "one"})
    ]


def test_clicked_missing_cell_uses_visible_sorted_rows(monkeypatch):
    """A sorted row index must be resolved against derived data, not source order."""
    registrations = []

    def capture(*dependencies, **kwargs):
        def register(function):
            registrations.append((dependencies, function))
            return function

        return register

    monkeypatch.setattr(build_callbacks, "callback", capture)
    monkeypatch.setattr(
        build_callbacks, "register_plot_download_callbacks", lambda: None
    )
    monkeypatch.setattr(
        build_callbacks, "register_plot_settings_callbacks", lambda: None
    )
    monkeypatch.setattr(
        build_callbacks, "plot_with_download_controls", lambda plot: plot
    )
    build_callbacks.plot_from_table_cell(
        "table", "plot", {"present": {"metric": "correct plot"}}
    )
    dependencies, click = registrations[0]
    assert dependencies[-1].component_property == "derived_viewport_data"
    visible = [
        {"id": "present", "metric": 2},
        {"id": "missing", "metric": None},
    ]
    assert click({"row": 0, "row_id": "present", "column_id": "metric"}, visible) == (
        "correct plot",
        None,
    )
    result, _ = click({"row": 1, "row_id": "missing", "column_id": "metric"}, visible)
    assert result.children == "No data available for this model."
