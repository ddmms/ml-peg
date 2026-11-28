"""Behavioural tests for reusable Plotly rendering helpers."""

from __future__ import annotations

import numpy as np
from plotly.colors import get_colorscale
import pytest

from ml_peg.app.utils.plot_helpers import build_annotated_heatmap


def test_annotated_heatmap_preserves_values_with_custom_colour_limits():
    """Colour limits affect colours, while data, labels and hover stay intact."""
    values = [[-7.0, 0.0, 12.3456789]]
    hover = "Method: %{y}<br>Case: %{x}<br>Error: %{z} eV<extra></extra>"
    fig = build_annotated_heatmap(
        values,
        x_labels=["third", "first", "second"],
        y_labels=["example-method"],
        hovertemplate=hover,
        missing_hovertemplate="Unavailable",
        zmin=-2,
        zmax=2,
        colorscale="Cividis",
        colorbar={"title": {"text": "Error [eV]"}},
    )
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "heatmap"
    assert list(trace.x) == ["third", "first", "second"]
    assert list(trace.y) == ["example-method"]
    assert list(trace.z[0]) == values[0]
    assert trace.zmin == -2
    assert trace.zmax == 2
    assert [stop[1] for stop in trace.colorscale] == [
        stop[1] for stop in get_colorscale("Cividis")
    ]
    assert trace.colorbar.title.text == "Error [eV]"
    assert trace.hovertemplate == hover
    assert trace.texttemplate == "%{text}"
    assert all(trace.text[0])


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf])
def test_annotated_heatmap_distinguishes_unavailable_values_from_zero(missing):
    """Missing cells get N/A annotations without inventing a zero value."""
    values = [[0.0, missing]]
    fig = build_annotated_heatmap(
        values,
        x_labels=["available", "missing"],
        y_labels=["example-method"],
        hovertemplate="Value: %{z}",
        missing_hovertemplate="Value unavailable",
    )
    trace, annotation = fig.data
    assert list(trace.z[0]) == [0.0, None]
    assert trace.hoverongaps is False
    assert trace.connectgaps is False
    assert list(annotation.x) == ["missing"]
    assert list(annotation.y) == ["example-method"]
    assert list(annotation.text) == ["N/A"]
    assert annotation.hovertemplate == "Value unavailable"
    assert values[0][1] is missing


def test_annotated_heatmap_without_rows_has_no_fabricated_data():
    """An empty result set produces no numeric trace or unavailable-value row."""
    fig = build_annotated_heatmap(
        [],
        x_labels=["case-a", "case-b"],
        y_labels=[],
        hovertemplate="Value: %{z}",
        missing_hovertemplate="Value unavailable",
    )
    assert len(fig.data) == 0


@pytest.mark.parametrize("matrix", [[[1.0]], [[1.0, 2.0], [3.0, 4.0]]])
def test_annotated_heatmap_rejects_misaligned_axis_labels(matrix):
    """Do not associate matrix entries with incomplete or mismatched labels."""
    with pytest.raises(ValueError, match="dimensions must match"):
        build_annotated_heatmap(
            matrix,
            x_labels=["case-a", "case-b"],
            y_labels=["example-method"],
            hovertemplate="Value: %{z}",
            missing_hovertemplate="Value unavailable",
        )
