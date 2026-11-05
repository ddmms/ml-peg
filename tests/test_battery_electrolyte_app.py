"""Battery density plot navigation and per-model structure selection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from dash.html import Iframe
import plotly.graph_objects as go
import pytest

from ml_peg.app.molecular_dynamics.battery_electrolyte_densities import (
    app_battery_electrolyte_densities as benchmark,
)
from ml_peg.app.utils import build_callbacks


@pytest.fixture
def registered_app(tmp_path, monkeypatch):
    """Register real app callbacks against reordered, partly missing assets."""
    callbacks = {}

    def capture(*args, **kwargs):
        def register(function):
            callbacks[function.__name__] = function
            return function

        return register

    renderer = Mock(return_value="<html>structure</html>")
    monkeypatch.setattr(build_callbacks, "callback", capture)
    monkeypatch.setattr(build_callbacks, "generate_weas_html", renderer)
    monkeypatch.setattr(build_callbacks, "_register_point_highlight", Mock())
    monkeypatch.setattr(build_callbacks, "register_plot_download_callbacks", Mock())
    monkeypatch.setattr(build_callbacks, "register_plot_settings_callbacks", Mock())
    monkeypatch.setattr(build_callbacks, "plot_with_download_controls", lambda x: x)
    monkeypatch.setattr(benchmark, "DATA_PATH", tmp_path)
    # The first model has no assets. Model B deliberately has only the second
    # plotted system, plus a stale file that must not influence point ordering.
    model_dir = tmp_path / "model-b"
    model_dir.mkdir()
    (model_dir / "a.xyz").touch()
    (model_dir / "stale.xyz").touch()
    fig = go.Figure(
        [
            go.Scatter(name="model-a", x=[1, 2], y=[1, 2], customdata=[["z"], ["a"]]),
            go.Scatter(name="model-b", x=[1, 2], y=[1, 2], customdata=[["z"], ["a"]]),
            go.Scatter(name="Pure solvent", x=[None], y=[None]),
            go.Scatter(x=[0, 2], y=[0, 2], mode="lines"),
        ]
    )
    fig.write_json(tmp_path / "figure_battery_electrolyte_densities.json")
    benchmark.BatteryElectrolyteDensitiesApp.register_callbacks(
        SimpleNamespace(table_id="table")
    )
    return callbacks, renderer


@pytest.mark.parametrize("metric", ["MAE", "RMSE", "MAPE"])
def test_all_metrics_open_plot(registered_app, metric):
    """Every density metric opens the existing parity graph."""
    callbacks, _ = registered_app
    graph, active_cell = callbacks["show_plot"]({"column_id": metric})
    assert graph.id == f"{benchmark.BENCHMARK_NAME}-figure"
    assert active_cell is None


def test_selects_clicked_model_and_label(registered_app):
    """A missing first model and stale files do not change point/model identity."""
    callbacks, renderer = registered_app
    graph, _ = callbacks["show_plot"]({"column_id": "MAE"})
    customdata = graph.figure["data"][1]["customdata"][1]
    assert customdata[0] == "a"
    view = callbacks["show_struct"](
        {"points": [{"curveNumber": 1, "pointNumber": 1, "customdata": customdata}]}
    )
    renderer.assert_called_once_with(
        "/assets/molecular_dynamics/battery_electrolyte_densities/model-b/a.xyz",
        "struct",
        0,
    )
    assert isinstance(view.children, Iframe)


@pytest.mark.parametrize("curve,point", [(0, 1), (1, 0), (2, 0), (3, 1), (4, 0)])
def test_missing_or_non_model_points_clear_view(registered_app, curve, point):
    """Missing assets, legend/reference traces and highlights do not render."""
    callbacks, renderer = registered_app
    graph, _ = callbacks["show_plot"]({"column_id": "MAE"})
    traces = graph.figure["data"]
    data = traces[curve].get("customdata") if curve < len(traces) else None
    click = {"curveNumber": curve, "pointNumber": point}
    if data is not None:
        click["customdata"] = data[point]
    view = callbacks["show_struct"]({"points": [click]})
    assert not isinstance(view.children, Iframe)
    renderer.assert_not_called()
