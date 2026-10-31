"""Regression checks for missing cells and numeric parity-plot inputs."""

from __future__ import annotations

import json

from dash.dcc import Graph
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import plot_parity
from ml_peg.app.utils import build_callbacks


@pytest.mark.parametrize(
    ("value", "missing"),
    [
        (None, True),
        (float("nan"), True),
        (np.float32("nan"), True),
        ("NaN", True),
        (" nan ", True),
        (0.0, False),
        (np.float32(1.0), False),
        ("1.2", False),
    ],
)
def test_table_cell_plot_rejects_missing_metrics(monkeypatch, value, missing) -> None:
    """Missing values cannot open another model's shared plot; zero stays valid."""
    callbacks = []

    def capture_callback(*_args, **_kwargs):
        def register(func):
            callbacks.append(func)
            return func

        return register

    monkeypatch.setattr(build_callbacks, "callback", capture_callback)
    monkeypatch.setattr(
        build_callbacks, "register_plot_download_callbacks", lambda: None
    )
    monkeypatch.setattr(
        build_callbacks, "register_plot_settings_callbacks", lambda: None
    )
    graph = Graph(id="shared-parity")
    monkeypatch.setattr(
        build_callbacks, "plot_with_download_controls", lambda plot: plot
    )
    build_callbacks.plot_from_table_cell("table", "plot", {"model": {"metric": graph}})

    result, active_cell = callbacks[0](
        {"row": 0, "row_id": "model", "column_id": "metric"},
        [{"id": "model", "metric": value}],
    )

    assert active_cell is None
    if missing:
        assert result.children == "No data available for this model."
    else:
        assert result is graph


def test_log_parity_limits_include_float32_and_exclude_nonfinite(tmp_path) -> None:
    """The parity line spans finite float32 observations rather than fallback limits."""
    output = tmp_path / "parity.json"

    @plot_parity(filename=output, log=True)
    def results():
        return {
            "ref": [np.float32(0.01), np.float32(100), np.inf],
            "model": [np.float32(0.001), np.float32(1000), np.nan],
        }

    results()
    figure = json.loads(output.read_text())
    parity_line = figure["data"][-1]
    assert parity_line["x"] == pytest.approx([0.001, 1000])
    assert parity_line["y"] == pytest.approx([0.001, 1000])


@pytest.mark.parametrize(
    ("structs", "mode", "template", "expected_file", "expected_index"),
    [
        ("trajectory.xyz", "traj", None, "trajectory.xyz", 2),
        (["a.xyz", "b.xyz", "c.xyz"], "struct", None, "c.xyz", 0),
        ({"mp-1": "reference.xyz"}, "struct", None, "reference.xyz", 0),
        ({"mp-1": "trajectory.xyz"}, "traj", None, "trajectory.xyz", 2),
        (None, "struct", "/assets/{id}.xyz", "/assets/mp-1.xyz", 0),
    ],
)
def test_structure_lookup_preserves_mapping_and_template_modes(
    monkeypatch, structs, mode, template, expected_file, expected_index
) -> None:
    """Both branches' lookup modes work with main's positional frame-follow argument."""
    callbacks = []

    def capture_callback(*_args, **_kwargs):
        def register(func):
            callbacks.append(func)
            return func

        return register

    rendered = []

    def render(filename, selected_mode, index):
        rendered.append((filename, selected_mode, index))
        return "structure viewer"

    monkeypatch.setattr(build_callbacks, "callback", capture_callback)
    monkeypatch.setattr(
        build_callbacks, "_register_point_highlight", lambda *a, **k: None
    )
    monkeypatch.setattr(build_callbacks, "generate_weas_html", render)
    build_callbacks.struct_from_scatter(
        "scatter", "viewer", structs, mode, False, struct_template=template
    )
    callbacks[0]({"points": [{"pointNumber": 2, "customdata": ["formula", "mp-1"]}]})

    assert rendered == [(expected_file, mode, expected_index)]
