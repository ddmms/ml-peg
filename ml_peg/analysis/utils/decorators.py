"""Fixtures for MLIP results analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import functools
import json
from json import dump
from pathlib import Path
from typing import Any

from dash import dash_table
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

from ml_peg.analysis.utils.utils import calc_table_scores
from ml_peg.app.utils.utils import Thresholds
from ml_peg.models.get_models import get_model_names, load_model_configs

PERIODIC_TABLE_POSITIONS: dict[str, tuple[int, int]] = {
    # First row
    "H": (0, 0),
    "He": (0, 17),
    # Second row
    "Li": (1, 0),
    "Be": (1, 1),
    "B": (1, 12),
    "C": (1, 13),
    "N": (1, 14),
    "O": (1, 15),
    "F": (1, 16),
    "Ne": (1, 17),
    # Third row
    "Na": (2, 0),
    "Mg": (2, 1),
    "Al": (2, 12),
    "Si": (2, 13),
    "P": (2, 14),
    "S": (2, 15),
    "Cl": (2, 16),
    "Ar": (2, 17),
    # Fourth row
    "K": (3, 0),
    "Ca": (3, 1),
    "Sc": (3, 2),
    "Ti": (3, 3),
    "V": (3, 4),
    "Cr": (3, 5),
    "Mn": (3, 6),
    "Fe": (3, 7),
    "Co": (3, 8),
    "Ni": (3, 9),
    "Cu": (3, 10),
    "Zn": (3, 11),
    "Ga": (3, 12),
    "Ge": (3, 13),
    "As": (3, 14),
    "Se": (3, 15),
    "Br": (3, 16),
    "Kr": (3, 17),
    # Fifth row
    "Rb": (4, 0),
    "Sr": (4, 1),
    "Y": (4, 2),
    "Zr": (4, 3),
    "Nb": (4, 4),
    "Mo": (4, 5),
    "Tc": (4, 6),
    "Ru": (4, 7),
    "Rh": (4, 8),
    "Pd": (4, 9),
    "Ag": (4, 10),
    "Cd": (4, 11),
    "In": (4, 12),
    "Sn": (4, 13),
    "Sb": (4, 14),
    "Te": (4, 15),
    "I": (4, 16),
    "Xe": (4, 17),
    # Sixth row
    "Cs": (5, 0),
    "Ba": (5, 1),
    "La": (8, 3),
    "Hf": (5, 3),
    "Ta": (5, 4),
    "W": (5, 5),
    "Re": (5, 6),
    "Os": (5, 7),
    "Ir": (5, 8),
    "Pt": (5, 9),
    "Au": (5, 10),
    "Hg": (5, 11),
    "Tl": (5, 12),
    "Pb": (5, 13),
    "Bi": (5, 14),
    "Po": (5, 15),
    "At": (5, 16),
    "Rn": (5, 17),
    # Seventh row
    "Fr": (6, 0),
    "Ra": (6, 1),
    "Ac": (9, 3),
    "Rf": (6, 3),
    "Db": (6, 4),
    "Sg": (6, 5),
    "Bh": (6, 6),
    "Hs": (6, 7),
    "Mt": (6, 8),
    "Ds": (6, 9),
    "Rg": (6, 10),
    "Cn": (6, 11),
    "Nh": (6, 12),
    "Fl": (6, 13),
    "Mc": (6, 14),
    "Lv": (6, 15),
    "Ts": (6, 16),
    "Og": (6, 17),
    # Lanthanides (row 8)
    "Ce": (8, 4),
    "Pr": (8, 5),
    "Nd": (8, 6),
    "Pm": (8, 7),
    "Sm": (8, 8),
    "Eu": (8, 9),
    "Gd": (8, 10),
    "Tb": (8, 11),
    "Dy": (8, 12),
    "Ho": (8, 13),
    "Er": (8, 14),
    "Tm": (8, 15),
    "Yb": (8, 16),
    "Lu": (8, 17),
    # Actinides (row 9)
    "Th": (9, 4),
    "Pa": (9, 5),
    "U": (9, 6),
    "Np": (9, 7),
    "Pu": (9, 8),
    "Am": (9, 9),
    "Cm": (9, 10),
    "Bk": (9, 11),
    "Cf": (9, 12),
    "Es": (9, 13),
    "Fm": (9, 14),
    "Md": (9, 15),
    "No": (9, 16),
    "Lr": (9, 17),
}
PERIODIC_TABLE_ROWS = 10
PERIODIC_TABLE_COLS = 18


def plot_parity(
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    hoverdata: dict | None = None,
    filename: str = "parity.json",
) -> Callable:
    """
    Plot parity plot of MLIP results against reference data.

    Parameters
    ----------
    title
        Graph title.
    x_label
        Label for x-axis. Default is `None`.
    y_label
        Label for y-axis. Default is `None`.
    hoverdata
        Hover data dictionary. Default is `{}`.
    filename
        Filename to save plot as JSON. Default is "parity.json".

    Returns
    -------
    Callable
        Decorator to wrap function.
    """

    def plot_parity_decorator(func: Callable) -> Callable:
        """
        Decorate function to plot parity.

        Parameters
        ----------
        func
            Function being wrapped.

        Returns
        -------
        Callable
            Wrapped function.
        """

        @functools.wraps(func)
        def plot_parity_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to plot parity.

            Parameters
            ----------
            *args
                Arguments to pass to the function being wrapped.
            **kwargs
                Key word arguments to pass to the function being wrapped.

            Returns
            -------
            dict
                Results dictionary.
            """
            results = func(*args, **kwargs)
            ref = results["ref"]

            hovertemplate = "<b>Pred: </b>%{x}<br>" + "<b>Ref: </b>%{y}<br>"

            customdata = []
            if hoverdata:
                for i, key in enumerate(hoverdata):
                    hovertemplate += f"<b>{key}: </b>%{{customdata[{i}]}}<br>"
                customdata = list(zip(*hoverdata.values(), strict=True))

            fig = go.Figure()
            for mlip, value in results.items():
                if mlip == "ref":
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=value,
                        y=ref,
                        name=mlip,
                        mode="markers",
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )

            full_fig = fig.full_figure_for_development()
            x_range = full_fig.layout.xaxis.range
            y_range = full_fig.layout.yaxis.range

            lims = [
                np.min([x_range, y_range]),  # min of both axes
                np.max([x_range, y_range]),  # max of both axes
            ]

            fig.add_trace(
                go.Scatter(
                    x=lims,
                    y=lims,
                    mode="lines",
                    showlegend=False,
                )
            )

            fig.update_layout(
                title={"text": title},
                xaxis={"title": {"text": x_label}},
                yaxis={"title": {"text": y_label}},
            )

            fig.update_traces()

            # Write to file
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fig.write_json(filename)

            return results

        return plot_parity_wrapper

    return plot_parity_decorator


def cell_to_scatter(
    *,
    filename: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    title_template: str = "{model} - {metric}",
) -> Callable:
    """
    Pre-generate scatter plots for each table cell (model-metric pair).

    Use this for benchmarks where each table CELL generates its own scatter plot
    (e.g., clicking MACE's ω_max shows a scatter for that specific model-metric
    pair). For benchmarks where clicking a COLUMN shows all models on one scatter
    (like S24 or OC157), use @plot_parity instead.

    This decorator generates complete Plotly figures during analysis instead of
    saving raw points for the app to process on-the-fly.

    Parameters
    ----------
    filename
        Path where JSON data with pre-made figures will be saved.
    x_label
        Label for x-axis (typically "Predicted"). Default is None.
    y_label
        Label for y-axis (typically "Reference"). Default is None.
    title_template
        Template for plot titles with {model} and {metric} placeholders.
        Default is "{model} - {metric}".

    Returns
    -------
    Callable
        Decorator that wraps analysis functions to pre-generate scatter figures.
    """

    def decorator(func: Callable) -> Callable:
        """
        Wrap the decorated callable to pre-generate scatter figures.

        Parameters
        ----------
        func
            Analysis function returning the dataset consumed by the Dash app.

        Returns
        -------
        Callable
            Wrapped function that runs ``func`` and emits Plotly figures.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Execute func and generate scatter plots for each model-metric pair.

            Parameters
            ----------
            *args
                Positional arguments forwarded to ``func``.
            **kwargs
                Keyword arguments forwarded to ``func``.

            Returns
            -------
            dict
                The dataset produced by ``func`` (with ``figures`` entries).
            """
            data_bundle = func(*args, **kwargs)

            # Extract metadata
            metric_labels = data_bundle.get("metrics", {})
            models_data = data_bundle.get("models", {})

            # Create figures for each model-metric pair
            for model_name, model_data in models_data.items():
                model_data["figures"] = {}
                metrics_data = model_data.get("metrics", {})

                for metric_key, metric_info in metrics_data.items():
                    points = metric_info.get("points", [])
                    if not points:
                        continue

                    # Extract ref and pred values
                    refs = [p["ref"] for p in points]
                    preds = [p["pred"] for p in points]
                    ids = [p.get("id", "") for p in points]

                    # Build hovertemplate
                    hovertemplate = (
                        "<b>Pred: </b>%{x}<br><b>Ref: </b>%{y}<br>"
                        "<b>ID: </b>%{customdata[0]}<br>"
                    )
                    customdata = [[id_val] for id_val in ids]

                    # Create figure
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=preds,
                            y=refs,
                            mode="markers",
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                            showlegend=False,
                        )
                    )

                    # Add parity line
                    full_fig = fig.full_figure_for_development()
                    x_range = full_fig.layout.xaxis.range
                    y_range = full_fig.layout.yaxis.range
                    lims = [
                        np.min([x_range, y_range]),
                        np.max([x_range, y_range]),
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=lims,
                            y=lims,
                            mode="lines",
                            showlegend=False,
                        )
                    )

                    # Update layout
                    metric_label = metric_labels.get(metric_key, metric_key)
                    title = title_template.format(model=model_name, metric=metric_label)
                    fig.update_layout(
                        title={"text": title},
                        xaxis={"title": {"text": x_label}},
                        yaxis={"title": {"text": y_label}},
                    )

                    # Store figure as JSON-serializable dict
                    model_data["figures"][metric_key] = json.loads(fig.to_json())

            # Save to file
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "w", encoding="utf8") as f:
                dump(data_bundle, f)

            return data_bundle

        return wrapper

    return decorator


def plot_scatter(
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    show_line: bool = False,
    show_markers: bool = True,
    hoverdata: dict | None = None,
    filename: str = "scatter.json",
    highlight_area: bool = False,
    highlight_range: dict = None,
) -> Callable:
    """
    Plot scatter plot of MLIP results.

    Parameters
    ----------
    title
        Graph title.
    x_label
        Label for x-axis. Default is `None`.
    y_label
        Label for y-axis. Default is `None`.
    show_line
        Whether to show line between points. Default is False.
    show_markers
        Whether to show markers on the plot. Default is True.
    hoverdata
        Hover data dictionary. Default is `{}`.
    filename
        Filename to save plot as JSON. Default is "scatter.json".
    highlight_area
        Whether to add a highlighted rectangle to the plot.
    highlight_range
        Dictionary of rectangle title and x-axis endpoints.

    Returns
    -------
    Callable
        Decorator to wrap function.
    """

    def plot_scatter_decorator(func: Callable) -> Callable:
        """
        Decorate function to plot scatter.

        Parameters
        ----------
        func
            Function being wrapped.

        Returns
        -------
        Callable
            Wrapped function.
        """

        @functools.wraps(func)
        def plot_scatter_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to plot scatter.

            Parameters
            ----------
            *args
                Arguments to pass to the function being wrapped.
            **kwargs
                Key word arguments to pass to the function being wrapped.

            Returns
            -------
            dict
                Results dictionary.
            """
            results = func(*args, **kwargs)

            hovertemplate = "<b>Pred: </b>%{x}<br>" + "<b>Ref: </b>%{y}<br>"
            customdata = []
            if hoverdata:
                for i, key in enumerate(hoverdata):
                    hovertemplate += f"<b>{key}: </b>%{{customdata[{i}]}}<br>"
                customdata = list(zip(*hoverdata.values(), strict=True))

            modes = []
            if show_line:
                modes.append("lines")
            if show_markers:
                modes.append("markers")

            mode = "+".join(modes)

            fig = go.Figure()
            for mlip, value in results.items():
                name = "Reference" if mlip == "ref" else mlip
                fig.add_trace(
                    go.Scatter(
                        x=value[0],
                        y=value[1],
                        name=name,
                        mode=mode,
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )

                colors = pc.qualitative.Plotly

                if highlight_area:
                    for i, (title, range) in enumerate(highlight_range.items()):
                        fig.add_vrect(
                            x0=range[0],
                            x1=range[1],
                            annotation_text=title,
                            annotation_position="top",
                            fillcolor=colors[i],
                            opacity=0.25,
                            line_width=0,
                        )

            fig.update_layout(
                title={"text": title},
                xaxis={"title": {"text": x_label}},
                yaxis={"title": {"text": y_label}},
            )

            fig.update_traces()

            # Write to file
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fig.write_json(filename)

            return results

        return plot_scatter_wrapper

    return plot_scatter_decorator


def plot_density_scatter(
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    filename: str = "density_scatter.json",
    colorbar_title: str = "Density",
    grid_size: int = 80,
    max_points_per_cell: int = 5,
    seed: int = 0,
    hover_metadata: dict[str, str] | None = None,
    annotation_metadata: dict[str, str] | None = None,
) -> Callable:
    """
    Plot density-coloured parity scatter with legend-based model toggling.

    The decorated function must return a mapping of model name to a dictionary with
    ``ref`` and ``pred`` arrays (and optional ``mae``). Each model is rendered as a
    scatter trace with marker colours indicating local data density.
    Only one model is shown at a time; use the legend to toggle models.

    Parameters
    ----------
    title
        Graph title shown above dropdown. Default is None.
    x_label
        Label for x-axis. Default is None.
    y_label
        Label for y-axis. Default is None.
    filename
        Filename to save plot as JSON. Default is "density_scatter.json".
    colorbar_title
        Title shown next to the density colour bar. Default is "Density".
    grid_size
        Number of bins per axis used to estimate local density. Default is 80.
    max_points_per_cell
        Maximum number of examples plotted per cell to keep renders responsive.
    seed
        Seed for deterministic sub-sampling. Default is 0.
    hover_metadata
        Dictionary mapping metadata keys to display labels for hover tooltips.
        Keys are used to look up values in each point's metadata; labels are shown
        in the hover text. Pass ``None`` (default) to omit additional hover metadata.
    annotation_metadata
        Dictionary mapping metadata keys to display labels for model-level
        annotations (shown in the text box on the plot). Keys are used to look up
        values in the model's metadata dict; labels are shown in the annotation.
        Pass ``None`` (default) to omit additional annotation metadata.

    Returns
    -------
    Callable
        Decorator to wrap function.
    """

    def plot_density_decorator(func: Callable) -> Callable:
        """
        Decorate function to plot density scatter.

        Parameters
        ----------
        func
            Function being wrapped.

        Returns
        -------
        Callable
            Wrapped function.
        """

        @functools.wraps(func)
        def plot_density_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to plot density scatter.

            Parameters
            ----------
            *args
                Arguments to pass to the function being wrapped.
            **kwargs
                Key word arguments to pass to the function being wrapped.

            Returns
            -------
            dict
                Results dictionary.
            """

            def _downsample(
                ref_vals: np.ndarray, pred_vals: np.ndarray
            ) -> tuple[list[float], list[float], list[int]]:
                """
                Downsample data points while keeping dense regions representative.

                Parameters
                ----------
                ref_vals
                    Reference (x-axis) values for all systems.
                pred_vals
                    Predicted (y-axis) values for all systems.

                Returns
                -------
                tuple[list[float], list[float], list[int]]
                    Downsampled reference values, predicted values, and density counts
                    corresponding to each retained point.
                """
                if ref_vals.size == 0:
                    return [], [], []

                delta_x = ref_vals.max() - ref_vals.min()
                delta_y = pred_vals.max() - pred_vals.min()
                eps = 1e-9

                norm_x = np.clip(
                    # Normalise to [0, 1). Clamp to avoid hitting the upper bound so
                    # bin indices always live in [0, grid_size - 1].
                    (ref_vals - ref_vals.min()) / max(delta_x, eps),
                    0.0,
                    0.999999,
                )
                norm_y = np.clip(
                    (pred_vals - pred_vals.min()) / max(delta_y, eps),
                    0.0,
                    0.999999,
                )
                bins_x = (norm_x * grid_size).astype(int)
                bins_y = (norm_y * grid_size).astype(int)
                cell_points: dict[tuple[int, int], list[int]] = defaultdict(list)
                for idx, (cx, cy) in enumerate(zip(bins_x, bins_y, strict=True)):
                    cell_points[(int(cx), int(cy))].append(idx)

                rng = np.random.default_rng(seed)
                sampled_x: list[float] = []
                sampled_y: list[float] = []
                sampled_density: list[int] = []
                for indices in cell_points.values():
                    if len(indices) > max_points_per_cell:
                        chosen = rng.choice(
                            indices, size=max_points_per_cell, replace=False
                        )
                    else:
                        chosen = indices
                    density = len(indices)
                    for idx in chosen:
                        sampled_x.append(float(ref_vals[idx]))
                        sampled_y.append(float(pred_vals[idx]))
                        sampled_density.append(density)

                return sampled_x, sampled_y, sampled_density

            results = func(*args, **kwargs)
            if not isinstance(results, dict):
                raise TypeError(
                    "Density plot decorator expects a mapping of model results."
                )

            if not results:
                raise ValueError("No results provided for density plot.")

            global_min = np.inf
            global_max = -np.inf
            processed = {}
            annotations = []
            annotation_fields = annotation_metadata or {}
            hover_fields = hover_metadata or {}

            for model in results:
                data = results[model]
                ref_vals = np.asarray(data.get("ref", []), dtype=float)
                pred_vals = np.asarray(data.get("pred", []), dtype=float)
                meta = data.get("meta") or {}

                # Extract annotation metadata values (for text box)
                annotation_values: list[str] = []
                for meta_key in annotation_fields:
                    meta_raw = meta.get(meta_key)
                    annotation_values.append(
                        "n/a" if meta_raw is None else str(meta_raw)
                    )

                # Extract hover metadata values (for tooltips)
                hover_values: list[str] = []
                for meta_key in hover_fields:
                    meta_raw = meta.get(meta_key)
                    hover_values.append("n/a" if meta_raw is None else str(meta_raw))

                if ref_vals.size == 0 or pred_vals.size == 0:
                    sampled = ([], [], [])
                else:
                    sampled = _downsample(ref_vals, pred_vals)
                    global_min = min(global_min, ref_vals.min(), pred_vals.min())
                    global_max = max(global_max, ref_vals.max(), pred_vals.max())

                # Build annotation text from annotation metadata
                summary_text = ""
                if annotation_fields:
                    summary_text = " | ".join(
                        f"{label}: {value}"
                        for value, label in zip(
                            annotation_values, annotation_fields.values(), strict=True
                        )
                    )
                annotations.append(
                    {
                        "text": f"{model}"
                        + (f" | {summary_text}" if summary_text else ""),
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.02,
                        "y": 0.98,
                        "showarrow": False,
                        "bgcolor": "rgba(255,255,255,0.8)",
                        "bordercolor": "rgba(0,0,0,0.3)",
                        "borderpad": 4,
                    }
                )
                processed[model] = {
                    "samples": sampled,
                    "counts": len(ref_vals),
                    "meta": hover_values if hover_fields else None,
                }

            if not np.isfinite(global_min) or not np.isfinite(global_max):
                global_min, global_max = 0.0, 1.0

            padding = 0.05 * (
                global_max - global_min if global_max != global_min else 1.0
            )
            line_start = global_min - padding
            line_end = global_max + padding

            fig = go.Figure()
            hover_lines = [
                "<b>Reference:</b> %{x:.3f}",
                "<b>Predicted:</b> %{y:.3f}",
                "<b>Density:</b> %{customdata[0]:.0f}",
            ]
            if hover_fields:
                for idx, label in enumerate(hover_fields.values()):
                    hover_lines.append(f"<b>{label}:</b> %{{meta[{idx}]}}")
            hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

            for idx, model in enumerate(results):
                sample_x, sample_y, density = processed[model]["samples"]
                fig.add_trace(
                    go.Scattergl(
                        x=sample_x,
                        y=sample_y,
                        mode="markers",
                        name=model,
                        visible=idx == 0,
                        marker={
                            "size": 6,
                            "color": density,
                            "colorscale": "Viridis",
                            "showscale": True,
                            "colorbar": {"title": colorbar_title},
                        },
                        customdata=np.array(density, dtype=float)[:, None]
                        if density
                        else None,
                        meta=processed[model]["meta"],
                        hovertemplate=hovertemplate,
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[line_start, line_end],
                    y=[line_start, line_end],
                    mode="lines",
                    showlegend=False,
                    line={"color": "black", "dash": "dash"},
                    visible=True,
                )
            )

            # Store all annotations and model order in layout meta so consumers
            # can swap annotation text when filtering per-model on the frontend.
            layout_meta = {
                "annotations": annotations,
                "models": list(results),
            }

            fig.update_layout(
                title={"text": title} if title else None,
                xaxis={"title": {"text": x_label}},
                yaxis={"title": {"text": y_label}},
                annotations=[annotations[0]] if annotations else [],
                meta=layout_meta,
                showlegend=True,
                legend_title_text="Model",
            )

            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fig.write_json(filename)
            return results

        return plot_density_wrapper

    return plot_density_decorator


def plot_periodic_table(
    title: str | None = None,
    colorbar_title: str | None = None,
    hoverdata: dict[str, dict[str, Any]] | None = None,
    filename: str = "periodic_table.json",
    colorscale: str = "Viridis",
) -> Callable:
    """
    Plot a periodic-table heatmap for element-wise metrics.

    Parameters
    ----------
    title
        Plot title.
    colorbar_title
        Label for the colour bar.
    hoverdata
        Optional mapping of hover labels to element-wise values.
    filename
        Output filename for the JSON figure.
    colorscale
        Plotly colourscale name. Default is ``"Viridis"``.

    Returns
    -------
    Callable
        Decorator to wrap function returning element-value mappings.
    """

    def plot_periodic_table_decorator(func: Callable) -> Callable:
        """
        Decorate function to produce periodic-table heatmap.

        Parameters
        ----------
        func
            Function returning mapping of element symbols to numeric values.

        Returns
        -------
        Callable
            Wrapped function.
        """

        @functools.wraps(func)
        def plot_periodic_table_wrapper(*args, **kwargs) -> dict[str, float]:
            """
            Wrap function to render periodic-table figure.

            Parameters
            ----------
            *args
                Positional arguments forwarded to ``func``.
            **kwargs
                Keyword arguments forwarded to ``func``.

            Returns
            -------
            dict[str, float]
                Element-value mapping returned by ``func``.
            """
            values = func(*args, **kwargs)

            grid = np.full((PERIODIC_TABLE_ROWS, PERIODIC_TABLE_COLS), np.nan)
            hover_grid = np.full_like(grid, "", dtype=object)
            text_grid = np.full_like(grid, "", dtype=object)

            for element, value in values.items():
                position = PERIODIC_TABLE_POSITIONS.get(element)
                if position is None:
                    continue
                row, col = position
                grid[row, col] = value

                hover_parts = [f"{element}"]
                if value is not None and not np.isnan(value):
                    hover_parts.append(f"Value: {value:.4g}")

                if hoverdata:
                    for label, mapping in hoverdata.items():
                        extra = mapping.get(element)
                        if extra is None:
                            continue
                        hover_parts.append(f"{label}: {extra}")

                hover_grid[row, col] = "<br>".join(hover_parts)
                text_grid[row, col] = element

            fig = go.Figure(
                data=go.Heatmap(
                    z=grid,
                    x=np.arange(PERIODIC_TABLE_COLS),
                    y=np.arange(PERIODIC_TABLE_ROWS),
                    text=hover_grid,
                    hovertemplate="%{text}<extra></extra>",
                    colorscale=colorscale,
                    colorbar={"title": colorbar_title},
                    showscale=True,
                )
            )

            # Overlay element symbols
            xs, ys, labels = [], [], []
            for element, (row, col) in PERIODIC_TABLE_POSITIONS.items():
                xs.append(col)
                ys.append(row)
                labels.append(text_grid[row, col] or element)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="text",
                    text=labels,
                    textfont={"size": 12, "color": "black"},
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                title={"text": title},
                xaxis={
                    "visible": False,
                    "range": [-0.5, PERIODIC_TABLE_COLS - 0.5],
                    "fixedrange": True,
                },
                yaxis={
                    "visible": False,
                    "autorange": "reversed",
                    "range": [-0.5, PERIODIC_TABLE_ROWS - 0.5],
                    "fixedrange": True,
                },
                margin={"l": 10, "r": 10, "t": 40, "b": 10},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fig.write_json(filename)
            return values

        return plot_periodic_table_wrapper

    return plot_periodic_table_decorator


def render_periodic_table_grid(
    title: str,
    filename_stem: str | Path,
    plot_cell: Callable[[Any, str], bool],
    *,
    figsize: tuple[float, float] | None = None,
    formats: tuple[str, ...] = ("svg", "pdf"),
    rows: int = PERIODIC_TABLE_ROWS,
    cols: int = PERIODIC_TABLE_COLS,
    suptitle_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Render a periodic-table grid where each element's subplot is custom drawn.

    Parameters
    ----------
    title
        Figure title displayed above the grid.
    filename_stem
        Base path for the output files (without extension).
    plot_cell
        Callable receiving ``(axis, element)``; should draw the subplot and
        return ``True`` when content was rendered.
    figsize
        Optional Matplotlib figure size. Defaults to a size proportional to the
        table geometry.
    formats
        Iterable of file extensions to emit (``"svg"``, ``"pdf"``, etc.).
    rows, cols
        Grid dimensions. Defaults to the standard periodic table layout.
    suptitle_kwargs
        Extra keyword arguments forwarded to ``fig.suptitle``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize or (cols * 1.5, rows * 1.2),
        constrained_layout=True,
    )
    axes = axes.reshape(rows, cols)

    for r in range(rows):
        for c in range(cols):
            axes[r, c].axis("off")

    for element, (row, col) in PERIODIC_TABLE_POSITIONS.items():
        ax = axes[row, col]
        rendered = False
        try:
            ax.axis("on")
            rendered = bool(plot_cell(ax, element))
        except Exception:
            rendered = False
        if not rendered:
            ax.axis("off")

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if ax.get_visible() and not ax.has_data():
                ax.axis("off")

    fig.suptitle(title, **(suptitle_kwargs or {}))

    base_path = Path(filename_stem)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base_path.with_suffix(f".{fmt}"), format=fmt)
    plt.close(fig)


def periodic_curve_gallery(
    *,
    curve_dir: Path,
    periodic_dir: Path | None = None,
    overview_title: str | None = None,
    overview_formats: tuple[str, ...] = (),
    overview_figsize: tuple[float, float] | None = (36, 20),
    overview_suptitle: dict[str, Any] | None = None,
    focus_title_template: str | None = None,
    focus_formats: tuple[str, ...] = (),
    focus_figsize: tuple[float, float] = (30, 15),
    focus_dpi: int = 200,
    pair_column: str = "pair",
    element1_column: str = "element_1",
    element2_column: str = "element_2",
    distance_column: str = "distance",
    energy_column: str = "energy",
    pair_separator: str = "-",
    series_columns: dict[str, str] | None = None,
    scalar_columns: dict[str, str] | None = None,
    x_ticks: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0),
    y_ticks: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0),
    x_range: tuple[float, float] = (0.0, 6.0),
    y_range: tuple[float, float] = (-20.0, 20.0),
) -> Callable:
    """
    Decorate a fixture that returns per-model curve data to persist gallery assets.

    Parameters
    ----------
    curve_dir
        Directory where per-pair JSON payloads will be written per model.
    periodic_dir
        Optional directory for periodic-table overviews and per-element figures. When
        omitted, only the curve payloads are written.
    overview_title
        Format string used for the overview title (receives ``model`` keyword). Ignored
        if ``overview_formats`` is empty.
    overview_formats
        File formats to emit for overview figure. Empty tuple disables overview output.
    overview_figsize
        Matplotlib figsize for the overview grid.
    overview_suptitle
        Additional kwargs passed to ``fig.suptitle`` for overview.
    focus_title_template
        Format string for per-element focus plots (receives ``element`` and ``model``).
        Set to ``None`` to disable focus figures.
    focus_formats
        File formats for per-element focus plots. Empty tuple disables focus output.
    focus_figsize
        Matplotlib figsize for focus plots.
    focus_dpi
        DPI for per-element focus images.
    pair_column, element1_column, element2_column, distance_column, energy_column
        Column names describing the curve data.
    pair_separator
        Separator used between elements within the ``pair`` column.
    series_columns
        Mapping of payload keys to column names whose values should be serialised
        as sequences.
    scalar_columns
        Mapping of payload keys to column names serialised as scalars.
    x_ticks, y_ticks
        Tick locations for the plots.
    x_range, y_range
        Axis limits for the plots.

    Returns
    -------
    Callable
        Decorator that wraps a callable returning model/dataframe mappings and
        emits the associated gallery assets.
    """
    curve_dir = Path(curve_dir)
    periodic_dir = Path(periodic_dir) if periodic_dir else None

    default_series = {"distance": distance_column, "energy": energy_column}
    if series_columns:
        default_series.update(series_columns)

    default_scalars = {"element_1": element1_column, "element_2": element2_column}
    if scalar_columns:
        default_scalars.update(scalar_columns)

    def _sorted_pair(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Return distance-sorted samples with duplicate distances removed.

        Parameters
        ----------
        frame
            Dataframe containing at least the ``distance`` column.

        Returns
        -------
        pd.DataFrame
            Frame sorted by distance with only unique distance entries.
        """
        return frame.sort_values(distance_column).drop_duplicates(distance_column)

    def _plot_curve(ax, pair_frame: pd.DataFrame) -> bool:
        """
        Plot a distance-energy curve on the supplied axes.

        Parameters
        ----------
        ax
            Matplotlib axes the curve is plotted onto.
        pair_frame
            Dataframe containing the curve samples.

        Returns
        -------
        bool
            ``True`` when data was plotted, otherwise ``False``.
        """
        if pair_frame.empty or energy_column not in pair_frame:
            return False
        ordered = _sorted_pair(pair_frame)
        if ordered.empty:
            return False
        x = ordered[distance_column].to_numpy()
        y = ordered[energy_column].to_numpy()
        if x.size == 0:
            return False
        shift = y[-1]
        y_shifted = y - shift
        ax.plot(x, y_shifted, linewidth=1, color="tab:blue", zorder=1)
        ax.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
        ax.set_facecolor("white")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(labelsize=7, length=2, pad=1)
        return True

    def _write_curve_payloads(model_name: str, frame: pd.DataFrame) -> None:
        """
        Serialise per-pair JSON payloads for the Dash callbacks.

        Parameters
        ----------
        model_name
            Name of the model associated with ``frame``.
        frame
            Dataframe containing all pair samples for the model.
        """
        model_curve_dir = curve_dir / model_name
        model_curve_dir.mkdir(parents=True, exist_ok=True)
        for pair, group in frame.groupby(pair_column):
            ordered = _sorted_pair(group)
            payload: dict[str, Any] = {"pair": str(pair)}
            for key, column in default_scalars.items():
                if column in ordered:
                    payload[key] = ordered[column].iloc[0]
            for key, column in default_series.items():
                if column in ordered:
                    payload[key] = ordered[column].tolist()
            with (model_curve_dir / f"{pair}.json").open("w", encoding="utf8") as fh:
                json.dump(payload, fh)

    def _render_overview(model_name: str, frame: pd.DataFrame) -> bool:
        """
        Render the homonuclear overview grid for ``model_name``.

        Parameters
        ----------
        model_name
            Name of the model whose overview is drawn.
        frame
            Dataframe containing all pair data for the model.

        Returns
        -------
        bool
            ``True`` when at least one curve was plotted.
        """

        def plot_cell(ax, element: str) -> bool:
            """
            Plot a single homonuclear curve in the periodic-table grid.

            Parameters
            ----------
            ax
                Matplotlib axes for the subplot.
            element
                Chemical symbol identifying the homonuclear pair.

            Returns
            -------
            bool
                ``True`` if data existed for the element.
            """
            pair_label = f"{element}{pair_separator}{element}"
            pair_frame = frame[frame[pair_column] == pair_label]
            rendered = _plot_curve(ax, pair_frame)
            if rendered:
                depth = float(
                    pair_frame[energy_column].min()
                    if energy_column in pair_frame
                    else 0.0
                )
                ax.text(
                    0.02,
                    0.95,
                    f"{element}\n{depth:.2f} eV",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                )
            return rendered

        render_periodic_table_grid(
            title=overview_title.format(model=model_name),
            filename_stem=periodic_dir / model_name / "overview",
            plot_cell=plot_cell,
            figsize=overview_figsize,
            formats=overview_formats,
            suptitle_kwargs=overview_suptitle,
        )
        return True

    def _render_focus(
        model_name: str,
        frame: pd.DataFrame,
        element: str,
        output_path: Path,
    ) -> bool:
        """
        Render per-element heteronuclear plots for a selected ``element``.

        Parameters
        ----------
        model_name
            Model identifier used in titles.
        frame
            Dataframe containing all pairs for the model.
        element
            Element symbol to highlight.
        output_path
            Destination path for the rendered figure.

        Returns
        -------
        bool
            ``True`` if any curve was drawn.
        """
        import matplotlib.pyplot as plt

        pair_groups: dict[str, pd.DataFrame] = {}
        for pair, group in frame.groupby(pair_column):
            try:
                first, second = str(pair).split(pair_separator)
            except ValueError:
                continue
            if element not in {first, second}:
                continue
            other = second if first == element else first
            pair_groups[other] = _sorted_pair(group)

        if not pair_groups:
            return False

        fig, axes = plt.subplots(
            PERIODIC_TABLE_ROWS,
            PERIODIC_TABLE_COLS,
            figsize=focus_figsize,
            constrained_layout=True,
        )
        axes = axes.reshape(PERIODIC_TABLE_ROWS, PERIODIC_TABLE_COLS)
        for ax in axes.ravel():
            ax.axis("off")

        has_data = False
        for other, (row, col) in PERIODIC_TABLE_POSITIONS.items():
            pair_frame = pair_groups.get(other)
            if pair_frame is None or pair_frame.empty:
                continue
            ax = axes[row, col]
            ax.axis("on")
            rendered = _plot_curve(ax, pair_frame)
            if not rendered:
                ax.axis("off")
                continue
            shift = float(
                pair_frame[energy_column].to_numpy()[-1]
                if energy_column in pair_frame and not pair_frame.empty
                else 0.0
            )
            ax.set_title(
                f"{element}-{other}, shift: {shift:.4f}",
                fontsize=8,
            )
            if other == element:
                for spine in ax.spines.values():
                    spine.set_edgecolor("crimson")
                    spine.set_linewidth(2)
            has_data = True

        if not has_data:
            plt.close(fig)
            return False

        if focus_title_template:
            fig.suptitle(
                focus_title_template.format(element=element, model=model_name),
                fontsize=22,
                fontweight="bold",
            )
        fig.savefig(output_path, format=output_path.suffix.lstrip("."), dpi=focus_dpi)
        plt.close(fig)
        return True

    def periodic_curve_gallery_decorator(func: Callable) -> Callable:
        """
        Wrap the supplied callable to emit gallery assets on invocation.

        Parameters
        ----------
        func
            Callable returning a mapping of model names to dataframes.

        Returns
        -------
        Callable
            Wrapped callable that additionally persists gallery assets.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Execute the wrapped callable and persist curve/plot assets.

            Parameters
            ----------
            *args
                Positional arguments forwarded to ``func``.
            **kwargs
                Keyword arguments forwarded to ``func``.

            Returns
            -------
            dict[str, pd.DataFrame]
                Mapping produced by the original callable.
            """
            model_frames = func(*args, **kwargs) or {}
            if not isinstance(model_frames, dict):
                raise TypeError(
                    "periodic_curve_gallery expects the wrapped function to return "
                    "a mapping of model names to pandas.DataFrame instances."
                )

            curve_dir.mkdir(parents=True, exist_ok=True)
            if periodic_dir:
                periodic_dir.mkdir(parents=True, exist_ok=True)

            for model_name, frame in model_frames.items():
                if frame is None or frame.empty:
                    continue
                _write_curve_payloads(model_name, frame)

                # Skip image/manifest generation when no formats are requested or
                # when no periodic_dir is supplied.
                if not periodic_dir or (
                    not overview_formats
                    and not (focus_title_template and focus_formats)
                ):
                    continue

                model_periodic_dir = periodic_dir / model_name
                model_periodic_dir.mkdir(parents=True, exist_ok=True)
                elements_dir = model_periodic_dir / "elements"
                elements_dir.mkdir(parents=True, exist_ok=True)

                manifest: dict[str, Any] = {"elements": {}}
                if overview_formats:
                    overview_rendered = _render_overview(model_name, frame)
                    if overview_rendered:
                        manifest["overview"] = f"overview.{overview_formats[0]}"

                if focus_title_template and focus_formats:
                    available_elements: set[str] = set()
                    if element1_column in frame:
                        available_elements |= {
                            str(e)
                            for e in frame[element1_column].dropna().astype(str)
                            if e
                        }
                    if element2_column in frame:
                        available_elements |= {
                            str(e)
                            for e in frame[element2_column].dropna().astype(str)
                            if e
                        }
                    available_elements = sorted(available_elements)
                    extension = focus_formats[0]
                    for element in available_elements:
                        output_path = elements_dir / f"{element}.{extension}"
                        if _render_focus(model_name, frame, element, output_path):
                            manifest["elements"][element] = (
                                f"elements/{element}.{extension}"
                            )

                manifest_path = model_periodic_dir / "manifest.json"
                with manifest_path.open("w", encoding="utf8") as fh:
                    json.dump(manifest, fh, indent=2)

            return model_frames

        return wrapper

    return periodic_curve_gallery_decorator


def build_table(
    *,
    thresholds: Thresholds,
    filename: str = "table.json",
    metric_tooltips: dict[str, str] | None = None,
    normalize: bool = True,
    normalizer: Callable[[float, float, float], float] | None = None,
    weights: dict[str, float] | None = None,
    mlip_name_map: dict[str, str] | None = None,
) -> Callable:
    """
    Build DataTable, including optional metric normalisation.

    If `normalize` is `True`, by default each metric is normalised to 0-1 scale where:
    - Values <= Y get score 0
    - Values >= X get score 1
    - Values between Y and X scale linearly, by default.

    Parameters
    ----------
    thresholds
        Mapping of metric names to dictionaries containing ``good``, ``bad``, and a
        ``unit`` entry (the unit value may explicitly be ``None``). All metrics must be
        covered so downstream rendering is consistent.
    filename
        Filename to save table. Default is "table.json".
    metric_tooltips
        Tooltips for table metric headers. Defaults are set for "MLIP" and "Score".
    normalize
        Whether to apply normalisation when calculating the score. Default is True.
    normalizer
        Optional function to map (value, X, Y) -> normalised score. Default is
        ml_peg.analysis.utils.utils.normalize_metric.
        Tooltips for table metric headers.
    weights
        Default weights for metrics. Default is 1 for all metrics.
    mlip_name_map
        Optional mapping of model identifier -> display name. Use this to annotate
        table rows (e.g. append a suffix) without changing the underlying model
        configuration metadata.

    Returns
    -------
    Callable
        Decorator to wrap function.
    """

    def build_table_decorator(func: Callable) -> Callable:
        """
        Decorate function to build table.

        Parameters
        ----------
        func
            Function being wrapped.

        Returns
        -------
        Callable
            Wrapped function.
        """

        @functools.wraps(func)
        def build_table_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to build table.

            Parameters
            ----------
            *args
                Arguments to pass to the function being wrapped.
            **kwargs
                Key word arguments to pass to the function being wrapped.

            Returns
            -------
            dict
                Results dictionary.
            """
            results = func(*args, **kwargs)

            missing_metrics = results.keys() - thresholds.keys()
            if missing_metrics:
                raise KeyError(
                    "Missing threshold entries for metrics: "
                    f"{', '.join(sorted(missing_metrics))}"
                )
            for metric_name, threshold_info in thresholds.items():
                if metric_name not in results:
                    continue
                if not isinstance(threshold_info, dict):
                    raise TypeError(
                        "Thresholds for metric "
                        f"'{metric_name}' must be provided as a mapping containing "
                        "'good', 'bad', and 'unit' entries."
                    )
                for required_key in ("good", "bad", "unit"):
                    if required_key not in threshold_info:
                        raise KeyError(
                            f"Threshold definition for '{metric_name}' is missing "
                            f"the '{required_key}' entry. Include the key even when "
                            "its value should be None."
                        )
            # Form of results is
            # results = {
            #     metric_1: {mlip_1: value_1, mlip_2: value_2},
            #     metric_2: {mlip_1: value_3, mlip_2: value_4},
            # }

            metrics_columns = ("MLIP",) + tuple(results)

            # Get all models (including those without results for this benchmark)
            mlips = tuple(get_model_names())

            name_map = mlip_name_map if mlip_name_map else {}
            display_names = {mlip: name_map.get(mlip, mlip) for mlip in mlips}
            display_values = list(display_names.values())
            if len(display_values) != len(set(display_values)):
                raise ValueError(
                    "Non-unique MLIP display names detected. Provide unique names via "
                    "'mlip_name_map'."
                )

            metrics_data = []
            for mlip in mlips:
                display_name = display_names[mlip]
                # For models without results, set metric values to None
                row_data = {"MLIP": display_name}
                for key, value in results.items():
                    row_data[key] = value.get(mlip, None)
                # Store the original model name in the row ID for callbacks, instead of
                # the display name (e.g. store mace-mp-0a instead of mace-mp-0a-D3)
                row_data["id"] = mlip
                metrics_data.append(row_data)

            summary_tooltips = {
                "MLIP": "Model identifier, hover for configuration details.",
            }
            if normalize:
                summary_tooltips["Score"] = (
                    "Weighted score across metrics, "
                    "Higher is better (normalised 0 to 1)."
                )
            else:
                summary_tooltips["Score"] = (
                    "Weighted score across metrics, higher is better."
                )

            if metric_tooltips:
                tooltip_header = metric_tooltips | summary_tooltips
            else:
                tooltip_header = summary_tooltips

            # Calculate scores, including any normalisation
            if normalize:
                metrics_data = calc_table_scores(
                    metrics_data=metrics_data,
                    thresholds=thresholds,
                    normalizer=normalizer,
                )
            else:
                metrics_data = calc_table_scores(metrics_data)

            metrics_columns += ("Score",)

            metric_weights = weights if weights else {}
            for column in results:
                metric_weights.setdefault(column, 1)

            table = dash_table.DataTable(
                metrics_data,
                [{"name": i, "id": i} for i in metrics_columns if i != "id"],
                id="metrics",
                tooltip_header=tooltip_header,
            )

            model_configs, model_levels = load_model_configs(mlips)

            model_configs = {
                display_names[name]: config for name, config in model_configs.items()
            }
            model_levels = {
                display_names[name]: level for name, level in model_levels.items()
            }

            # Extract metric level of theory from thresholds
            metric_levels = {}
            for metric_name in results:
                metric_levels[metric_name] = thresholds.get(metric_name, {}).get(
                    "level_of_theory"
                )

            # Save dict of table to be loaded
            model_name_map = {
                display_name: original_name
                for original_name, display_name in display_names.items()
            }

            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "w") as fp:
                dump(
                    {
                        "data": table.data,
                        "columns": table.columns,
                        "tooltip_header": tooltip_header,
                        "thresholds": thresholds,
                        "weights": metric_weights,
                        "model_levels_of_theory": model_levels,
                        "metric_levels_of_theory": metric_levels,
                        "model_configs": model_configs,
                        "model_name_map": model_name_map,
                    },
                    fp,
                )

            return results

        return build_table_wrapper

    return build_table_decorator
