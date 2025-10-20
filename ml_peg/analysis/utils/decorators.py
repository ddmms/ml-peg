"""Fixtures for MLIP results analysis."""

from __future__ import annotations

from collections.abc import Callable
import functools
from json import dump
from pathlib import Path
from typing import Any

from dash import dash_table
import numpy as np
import plotly.graph_objects as go

from ml_peg.analysis.utils.utils import calc_ranks, calc_scores, normalize_metric


def calc_normalized_scores(
    rows: list[dict[str, Any]],
    normalization_thresholds: dict[str, tuple[float, float]],
    normalizer: Callable[[float, float, float], float] | None = None,
) -> list[dict[str, Any]]:
    """
    Normalize metric values row-by-row and compute composite scores.

    Parameters
    ----------
    rows
        Table rows containing metric values.
    normalization_thresholds
        Normalization thresholds keyed by metric name, where each value is
        a (good_threshold, bad_threshold) tuple.
    normalizer
        Optional function to map (value, good, bad) -> normalized score.
        If None, uses normalize_metric from utils.

    Returns
    -------
    list[dict[str, Any]]
        Updated rows including normalized ``Score`` entries.
    """
    _normalizer = normalizer if normalizer is not None else normalize_metric

    out = []
    for row in rows:
        norm_scores = []
        for key, value in row.items():
            if (
                key not in ("MLIP", "Score", "Rank", "id")
                and key in normalization_thresholds
            ):
                good_threshold, bad_threshold = normalization_thresholds[key]
                norm_scores.append(_normalizer(value, good_threshold, bad_threshold))
        row = row.copy()
        row["Score"] = float(np.mean(norm_scores)) if norm_scores else 0.0
        out.append(row)
    return out


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


def build_table(
    filename: str = "table.json",
    metric_tooltips: dict[str, str] | None = None,
) -> Callable:
    """
    Build table MLIP results.

    Parameters
    ----------
    filename
        Filename to save table.
    metric_tooltips
        Tooltips for table metric headers.

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
            # Form of results is
            # results = {
            #     metric_1: {mlip_1: value_1, mlip_2: value_2},
            #     metric_2: {mlip_1: value_3, mlip_2: value_4},
            # }

            metrics_columns = ("MLIP",) + tuple(results.keys())
            # Use MLIP keys from first (any) metric keys
            mlips = next(iter(results.values())).keys()

            metrics_data = []
            for mlip in mlips:
                metrics_data.append(
                    {"MLIP": mlip}
                    | {key: value[mlip] for key, value in results.items()}
                    | {"id": mlip},
                )

            summary_tooltips = {
                "MLIP": "Name of the model",
                "Score": "Average of metrics (lower is better)",
                "Rank": "Model rank based on score (lower is better)",
            }
            if metric_tooltips:
                tooltip_header = metric_tooltips | summary_tooltips
            else:
                tooltip_header = summary_tooltips

            metrics_data = calc_scores(metrics_data)
            metrics_data = calc_ranks(metrics_data)
            metrics_columns += ("Score", "Rank")

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
                    },
                    fp,
                )

            return results

        return build_table_wrapper

    return build_table_decorator


def build_normalized_table(
    filename: str = "normalized_table.json",
    metric_tooltips: dict[str, str] | None = None,
    normalization_ranges: dict[str, tuple[float, float]] | None = None,
    normalizer: Callable[[float, float, float], float] | None = None,
) -> Callable:
    """
    Build table with metric normalization capabilities.

    Each metric gets normalized to 0-1 scale where:
    - Values <= Y get score 0
    - Values >= X get score 1
    - Values between Y and X scale linearly

    Parameters
    ----------
    filename
        Filename to save table.
    metric_tooltips
        Tooltips for table metric headers.
    normalization_ranges
        Mapping of metric names to (X, Y) tuples where X is the upper threshold and
        Y is the lower threshold.
    normalizer
        Optional function to map (value, X, Y) -> normalized score. If None, uses
        ml_peg.analysis.utils.utils.normalize_metric. Allows easy swapping later.

    Returns
    -------
    Callable
        Decorator to wrap function.
    """

    def build_normalized_table_decorator(func: Callable) -> Callable:
        """
        Decorate function to build normalized table.

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
        def build_normalized_table_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to build normalized table.

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
            # Form of results is
            # results = {
            #     metric_1: {mlip_1: value_1, mlip_2: value_2},
            #     metric_2: {mlip_1: value_3, mlip_2: value_4},
            # }

            metrics_columns = ("MLIP",) + tuple(results.keys())
            # Use MLIP keys from first (any) metric keys
            mlips = next(iter(results.values())).keys()

            metrics_data = []
            for mlip in mlips:
                metrics_data.append(
                    {"MLIP": mlip}
                    | {key: value[mlip] for key, value in results.items()}
                    | {"id": mlip},
                )

            summary_tooltips = {
                "MLIP": "Name of the model",
                "Score": "Average of normalized metrics (higher is better)",
                "Rank": "Model rank based on score (lower is better)",
            }
            if metric_tooltips:
                tooltip_header = metric_tooltips | summary_tooltips
            else:
                tooltip_header = summary_tooltips

            # Set default normalization ranges if not provided
            default_ranges = {}
            if normalization_ranges:
                default_ranges = normalization_ranges.copy()

            # Create default ranges for any missing metrics
            for metric_name in results.keys():
                if metric_name not in default_ranges:
                    # Default: assume lower is better, set reasonable defaults
                    values = [
                        results[metric_name][mlip]
                        for mlip in mlips
                        if not np.isnan(results[metric_name][mlip])
                    ]
                    if values:
                        min_val, max_val = min(values), max(values)
                        # X (upper threshold) = min value (best case)
                        # Y (lower threshold) = max value (worst case)
                        default_ranges[metric_name] = (min_val, max_val)
                    else:
                        default_ranges[metric_name] = (0.0, 1.0)

            # Apply normalization and calculate scores using the utility function
            metrics_data = calc_normalized_scores(
                metrics_data, default_ranges, normalizer=normalizer
            )
            metrics_data = calc_ranks(metrics_data)
            metrics_columns += ("Score", "Rank")

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
                        "normalization_ranges": default_ranges,
                    },
                    fp,
                )

            return results

        return build_normalized_table_wrapper

    return build_normalized_table_decorator
