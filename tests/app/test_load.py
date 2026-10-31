"""Unit tests for figure loading in ``ml_peg.app.utils.load``.

These pin the behaviour of the start-up optimisation that passes saved figures to
``dcc.Graph`` as plain dicts instead of rebuilding validated plotly ``Figure``
objects, guarding against silently shipping empty plots.
"""

from __future__ import annotations

import json
from pathlib import Path

from dash.dcc import Graph

from ml_peg.app.defects.split_vacancy.app_split_vacancy import _trace_models
from ml_peg.app.utils.load import _responsive_mode, read_plot


def test_read_plot_returns_dict_figure_with_data(tmp_path: Path) -> None:
    """``read_plot`` returns a Graph whose figure is a dict containing traces."""
    figure = {
        "data": [{"type": "scatter", "x": [0, 1], "y": [1, 2], "name": "Test model"}],
        "layout": {"title": {"text": "Test figure"}},
    }
    plot_path = tmp_path / "figure_test.json"
    plot_path.write_text(json.dumps(figure))

    graph = read_plot(plot_path, id="unit-test-figure")

    assert isinstance(graph, Graph)
    assert isinstance(graph.figure, dict), "figure should pass through as a plain dict"
    assert graph.figure == figure, "trace data and layout should survive loading"


def test_read_plot_missing_file_returns_empty_graph(tmp_path: Path) -> None:
    """``read_plot`` returns a Graph with no figure when the file is absent."""
    graph = read_plot(tmp_path / "does-not-exist.json", id="missing-figure")

    assert isinstance(graph, Graph)
    assert graph.figure is None


def test_responsive_mode_respects_authored_figure_size() -> None:
    """A figure that sets its own size must not be stretched to its container.

    ``dcc.Graph(responsive=True)`` unsets ``layout.height``/``layout.width``, so
    forcing it would collapse the saved figures that declare a size (e.g. the
    1500x1500 graphene-wetting panels) to the default graph height.
    """
    assert _responsive_mode({"layout": {"height": 1500, "width": 1500}}) == "auto"
    assert _responsive_mode({"layout": {"height": 500}}) == "auto"
    assert _responsive_mode({"layout": {"title": "no size"}}) is True
    assert _responsive_mode({}) is True
    assert _responsive_mode(None) is True


def test_figure_consumers_read_dicts_not_figure_objects() -> None:
    """Consumers of ``read_plot`` must index the figure, not use attributes.

    Passing figures through as dicts is what makes start-up fast, but it changes
    the contract every consumer relies on: ``figure.data`` works on a plotly
    ``Figure`` and raises ``AttributeError`` on a dict. ``get_all_tests`` only
    rescues ``FileNotFoundError``, so such a consumer takes down the whole app
    build rather than skipping its own benchmark — and it stays invisible until
    that benchmark's data ships. ``_trace_models`` is the one Python consumer
    that reaches into traces.
    """
    figure = {"data": [{"name": "mace-mp-0a"}, {"name": None}, {}], "layout": {}}

    assert _trace_models(Graph(figure=figure)) == ["mace-mp-0a", None, None]
    assert _trace_models(Graph(figure=None)) == []
