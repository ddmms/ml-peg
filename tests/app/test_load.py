"""Unit tests for figure loading in ``ml_peg.app.utils.load``.

These pin the behaviour of the start-up optimisation that passes saved figures to
``dcc.Graph`` as plain dicts instead of rebuilding validated plotly ``Figure``
objects, guarding against silently shipping empty plots.
"""

from __future__ import annotations

from dash.dcc import Graph

from ml_peg.app import APP_ROOT
from ml_peg.app.defects.split_vacancy.app_split_vacancy import _trace_models
from ml_peg.app.utils.load import _responsive_mode, read_plot


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
