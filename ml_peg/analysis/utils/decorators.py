"""Fixtures for MLIP results analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import functools
from json import dump
from pathlib import Path
from typing import Any

from dash import dash_table
import numpy as np
import plotly.graph_objects as go

from ml_peg.analysis.utils.utils import calc_table_scores
from ml_peg.app.utils.utils import Thresholds


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


def plot_scatter(
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    show_line: bool = False,
    hoverdata: dict | None = None,
    filename: str = "scatter.json",
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
    hoverdata
        Hover data dictionary. Default is `{}`.
    filename
        Filename to save plot as JSON. Default is "scatter.json".

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

            mode = "lines+markers" if show_line else "markers"

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
                    (ref_vals - ref_vals.min()) / max(delta_x, eps), 0.0, 0.999999
                )
                norm_y = np.clip(
                    (pred_vals - pred_vals.min()) / max(delta_y, eps), 0.0, 0.999999
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

            model_names = list(results)
            if not model_names:
                raise ValueError("No results provided for density plot.")

            global_min = np.inf
            global_max = -np.inf
            processed = {}
            annotations = []
            for model in model_names:
                data = results[model]
                ref_vals = np.asarray(data.get("ref", []), dtype=float)
                pred_vals = np.asarray(data.get("pred", []), dtype=float)
                meta = data.get("meta") or {}
                excluded = meta.get("excluded")
                excluded_text = str(excluded) if excluded is not None else "n/a"
                if ref_vals.size == 0 or pred_vals.size == 0:
                    sampled = ([], [], [])
                else:
                    sampled = _downsample(ref_vals, pred_vals)
                    global_min = min(global_min, ref_vals.min(), pred_vals.min())
                    global_max = max(global_max, ref_vals.max(), pred_vals.max())
                mae_value = data.get("mae")
                if mae_value is None and ref_vals.size:
                    mae_value = float(np.mean(np.abs(ref_vals - pred_vals)))
                mae_text = f"{mae_value:.3f}" if mae_value is not None else "n/a"
                annotations.append(
                    {
                        "text": f"{model} MAE: {mae_text} | Excluded: {excluded_text}",
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
                    "meta": excluded_text,
                }

            if not np.isfinite(global_min) or not np.isfinite(global_max):
                global_min, global_max = 0.0, 1.0

            padding = 0.05 * (
                global_max - global_min if global_max != global_min else 1.0
            )
            line_start = global_min - padding
            line_end = global_max + padding

            fig = go.Figure()
            hovertemplate = (
                "<b>Reference:</b> %{x:.3f}<br>"
                "<b>Predicted:</b> %{y:.3f}<br>"
                "<b>Density:</b> %{customdata[0]:.0f}<br>"
                "<b>Excluded:</b> %{meta[0]}<extra></extra>"
            )

            for idx, model in enumerate(model_names):
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
                        meta=[processed[model]["meta"]],
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
                "models": model_names,
            }

            fig.update_layout(
                title={"text": title} if title else None,
                xaxis={"title": {"text": x_label}},
                yaxis={"title": {"text": y_label}},
                annotations=[annotations[0]],
                meta=layout_meta,
                showlegend=True,
                legend_title_text="Model",
            )

            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fig.write_json(filename)
            return results

        return plot_density_wrapper

    return plot_density_decorator


def build_table(
    *,
    thresholds: Thresholds,
    filename: str = "table.json",
    metric_tooltips: dict[str, str] | None = None,
    normalize: bool = True,
    normalizer: Callable[[float, float, float], float] | None = None,
    weights: dict[str, float] | None = None,
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
            # Use MLIP keys from first (any) metric keys
            mlips = next(iter(results.values())).keys()

            metrics_data = []
            for mlip in mlips:
                metrics_data.append(
                    {"MLIP": mlip}
                    | {key: value[mlip] for key, value in results.items()}
                    | {"id": mlip},
                )

            summary_tooltips = {"MLIP": "Name of the model"}
            if normalize:
                summary_tooltips["Score"] = (
                    "Average of normalised metrics (higher is better)"
                )
            else:
                summary_tooltips["Score"] = "Average of metrics"

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

            # Save dict of table to be loaded
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "w") as fp:
                dump(
                    {
                        "data": table.data,
                        "columns": table.columns,
                        "tooltip_header": tooltip_header,
                        "thresholds": thresholds,
                        "weights": metric_weights,
                    },
                    fp,
                )

            return results

        return build_table_wrapper

    return build_table_decorator
