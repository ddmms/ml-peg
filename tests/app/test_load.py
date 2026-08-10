"""Unit tests for figure loading in ``ml_peg.app.utils.load``.

These pin the behaviour of the start-up optimisation that passes saved figures to
``dcc.Graph`` as plain dicts instead of rebuilding validated plotly ``Figure``
objects, guarding against silently shipping empty plots.
"""

from __future__ import annotations

from dash.dcc import Graph

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.load import read_plot


def test_read_plot_returns_dict_figure_with_data() -> None:
    """``read_plot`` returns a Graph whose figure is a dict containing traces."""
    candidates = sorted((APP_ROOT / "data").rglob("figure_*.json"))
    assert candidates, "no saved figure_*.json found under app data"

    graph = read_plot(candidates[0], id="unit-test-figure")

    assert isinstance(graph, Graph)
    assert isinstance(graph.figure, dict), "figure should pass through as a plain dict"
    assert graph.figure.get("data"), "figure should contain trace data"


def test_read_plot_missing_file_returns_empty_graph() -> None:
    """``read_plot`` returns a Graph with no figure when the file is absent."""
    graph = read_plot(APP_ROOT / "data" / "does-not-exist.json", id="missing-figure")

    assert isinstance(graph, Graph)
    assert graph.figure is None
