"""Test symmetric log colour scaling for periodic table plots."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import plot_periodic_table
from ml_peg.analysis.utils.utils import symlog_transform

SCALE = 0.1


def build_table(values: dict[str, float], tmp_path: Path, **kwargs) -> dict:
    """
    Write a periodic table plot and read back the figure.

    Parameters
    ----------
    values
        Element-value mapping to plot.
    tmp_path
        Directory to write the figure to.
    **kwargs
        Additional keyword arguments passed to `plot_periodic_table`.

    Returns
    -------
    dict
        Plotly figure, as written to JSON.
    """
    filename = tmp_path / "periodic_table.json"
    plot_periodic_table(
        title="Test",
        colorbar_title="Error",
        filename=str(filename),
        **kwargs,
    )(lambda: values)()
    return json.loads(filename.read_text(encoding="utf8"))


def decode(values: dict | list) -> np.ndarray:
    """
    Decode a numeric array from a figure, which Plotly may base64 encode.

    Parameters
    ----------
    values
        Array as written to JSON, either a plain list or an encoded mapping.

    Returns
    -------
    np.ndarray
        Decoded array.
    """
    if not isinstance(values, dict):
        return np.asarray(values, dtype=float)

    decoded = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    if shape := values.get("shape"):
        decoded = decoded.reshape([int(dim) for dim in shape.split(",")])
    return decoded


def get_cell(figure: dict, element: str) -> float:
    """
    Get the plotted value for an element.

    Parameters
    ----------
    figure
        Plotly figure to read.
    element
        Element symbol to look up.

    Returns
    -------
    float
        Plotted value.
    """
    heatmap = figure["data"][0]
    grid = decode(heatmap["z"])
    for row, hover_row in enumerate(heatmap["text"]):
        for col, hover in enumerate(hover_row):
            if hover and hover.split("<br>")[0] == element:
                return grid[row, col]
    raise AssertionError(f"{element} not found in figure")


def test_symlog_transform_preserves_sign_and_zero():
    """Test transform is sign-preserving and fixes zero."""
    values = np.array([-5.0, -0.1, 0.0, 0.1, 5.0])
    transformed = symlog_transform(values, SCALE)

    assert transformed[2] == 0
    assert np.all(np.sign(transformed) == np.sign(values))
    # Symmetric about zero
    assert transformed[0] == pytest.approx(-transformed[4])
    assert transformed[1] == pytest.approx(-transformed[3])


def test_symlog_transform_is_monotonic():
    """Test ordering of values is unchanged, so the colour scale stays meaningful."""
    values = np.linspace(-20, 20, 101)
    transformed = symlog_transform(values, SCALE)

    assert np.all(np.diff(transformed) > 0)


def test_symlog_transform_near_linear_within_scale():
    """Test values well inside the scale are mapped near-linearly."""
    values = np.array([0.0, 0.0005, 0.001, 0.002])
    transformed = symlog_transform(values, SCALE)

    # log10(1 + x) -> x / ln(10) for small x
    expected = values / (SCALE * np.log(10))
    assert transformed == pytest.approx(expected, rel=0.02)


def test_symlog_transform_compresses_outliers():
    """Test outliers no longer dominate the range, which is the point of the scale."""
    good, outlier = 0.1, 20.0
    transformed = symlog_transform(np.array([good, outlier]), SCALE)

    # Linearly the outlier is 200x the good value, but under 20x once compressed
    assert outlier / good == pytest.approx(200)
    assert transformed[1] / transformed[0] < 20


def test_symlog_transform_preserves_nan():
    """Test missing values stay missing rather than becoming numbers."""
    transformed = symlog_transform(np.array([np.nan, 1.0]), SCALE)

    assert np.isnan(transformed[0])
    assert np.isfinite(transformed[1])


def test_periodic_table_symlog_transforms_grid(tmp_path):
    """Test plotted values are transformed, and empty cells stay empty."""
    values = {"Fe": 2.0, "Pt": -0.05, "H": 0.0}
    figure = build_table(values, tmp_path, symlog_scale=SCALE, zmin=-2.0, zmax=2.0)

    for element, value in values.items():
        expected = symlog_transform(np.array([value]), SCALE)[0]
        assert get_cell(figure, element) == pytest.approx(expected)

    # Elements with no value are not plotted
    grid = decode(figure["data"][0]["z"])
    assert np.count_nonzero(np.isfinite(grid)) == len(values)


def test_periodic_table_symlog_colorbar_labels_original_units(tmp_path):
    """Test the colour bar is labelled in original units at transformed positions."""
    figure = build_table(
        {"Fe": 2.0, "Pt": -2.0}, tmp_path, symlog_scale=SCALE, zmin=-2.0, zmax=2.0
    )
    colorbar = figure["data"][0]["colorbar"]

    ticks = np.array([float(text) for text in colorbar["ticktext"]])
    assert ticks.size
    # Labels span both signs and are within the plotted range
    assert ticks.min() < 0 < ticks.max()
    assert np.all(np.abs(ticks) <= 2.0)
    # Tick positions are the labels mapped through the same transform
    assert decode(colorbar["tickvals"]) == pytest.approx(symlog_transform(ticks, SCALE))


def test_periodic_table_symlog_transforms_limits(tmp_path):
    """Test requested limits are transformed, so they match the transformed grid."""
    figure = build_table(
        {"Fe": 2.0, "Pt": -2.0}, tmp_path, symlog_scale=SCALE, zmin=-2.0, zmax=2.0
    )
    heatmap = figure["data"][0]

    expected = symlog_transform(np.array([-2.0, 2.0]), SCALE)
    assert heatmap["zmin"] == pytest.approx(expected[0])
    assert heatmap["zmax"] == pytest.approx(expected[1])
    # Untransformed limits would leave the data well inside the range, washing it out
    assert get_cell(figure, "Fe") == pytest.approx(heatmap["zmax"])


def test_periodic_table_symlog_rejects_non_positive(tmp_path):
    """Test a non-positive scale is rejected, as the transform divides by it."""
    for scale in (0, -1):
        with pytest.raises(ValueError, match="`symlog_scale` must be positive"):
            build_table({"Fe": 1.0}, tmp_path, symlog_scale=scale)


def test_periodic_table_symlog_all_nan_without_limits(tmp_path):
    """Test an all-NaN grid falls back to finite limits instead of warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = build_table({"Fe": np.nan}, tmp_path, symlog_scale=SCALE)

    assert not [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]

    heatmap = figure["data"][0]
    assert np.isfinite(heatmap["zmin"])
    assert np.isfinite(heatmap["zmax"])
    assert heatmap["zmin"] < heatmap["zmax"]


def test_periodic_table_symlog_all_zero(tmp_path):
    """Test a grid of identical zero values renders rather than raising."""
    figure = build_table({"Fe": 0.0, "Pt": 0.0}, tmp_path, symlog_scale=SCALE)

    heatmap = figure["data"][0]
    assert get_cell(figure, "Fe") == 0
    # The range is degenerate, so no decade labels are possible
    assert heatmap["colorbar"]["ticktext"] == []


def test_periodic_table_linear_by_default(tmp_path):
    """Test values are untransformed and unlabelled without a symlog scale."""
    figure = build_table({"Fe": 2.0, "Pt": -0.05}, tmp_path)
    colorbar = figure["data"][0]["colorbar"]

    assert get_cell(figure, "Fe") == pytest.approx(2.0)
    assert get_cell(figure, "Pt") == pytest.approx(-0.05)
    assert "tickvals" not in colorbar
    assert "ticktext" not in colorbar
