"""Utilities for the thermodynamic properties app."""

from __future__ import annotations

from pathlib import Path

from dash.dcc import Graph

from ml_peg.app.utils.load import read_plot


def load_property_plots(
    data_path: Path,
    benchmark_name: str,
    properties: tuple[str, ...],
) -> dict[str, Graph]:
    """
    Load plots for thermodynamic properties.

    Parameters
    ----------
    data_path
        Path containing the benchmark plot files.
    benchmark_name
        Name of the benchmark used to construct Dash component IDs.
    properties
        Names of the thermodynamic properties to load.

    Returns
    -------
    dict[str, Graph]
        Mapping from property names to Dash graph components.
    """
    return {
        name: read_plot(
            data_path / f"figure_{name}.json",
            id=f"{benchmark_name}-{name}-figure",
        )
        for name in properties
    }


def map_metrics_to_plots(
    plots: dict[str, Graph],
    metrics: tuple[str, ...] = ("MAE", "MAZE"),
) -> dict[str, Graph]:
    """
    Map metric table columns to property plots.

    Parameters
    ----------
    plots
        Mapping from property names to Dash graph components.
    metrics
        Metric names associated with each property.

    Returns
    -------
    dict[str, Graph]
        Mapping from metric table column names to Dash graph components.
    """
    return {
        f"{property_name}_{metric}": plot
        for property_name, plot in plots.items()
        for metric in metrics
    }


def get_structures(
    data_path: Path,
    model: str,
    asset_path: str,
) -> list[str]:
    """
    Get structure asset paths for a model.

    Parameters
    ----------
    data_path
        Path containing benchmark data.
    model
        Name of the model whose structures should be loaded.
    asset_path
        URL path to the benchmark structure assets.

    Returns
    -------
    list[str]
        Structure asset paths ordered by filename.
    """
    model_dir = data_path / model

    if not model_dir.exists():
        return []

    return [
        f"{asset_path}/{model}/{file.stem}.xyz"
        for file in sorted(model_dir.glob("*.xyz"))
    ]
