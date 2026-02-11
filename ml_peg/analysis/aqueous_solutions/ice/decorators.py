"""Fixtures for MLIP results analysis for ice."""

from __future__ import annotations

from collections.abc import Callable
import functools
import json
from json import dump
from pathlib import Path
from typing import Any

import plotly.graph_objects as go


def cell_to_bar(
    *,
    filename: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    title_template: str = "{model} - {metric}",
) -> Callable:
    """
    Pre-generate bar plots for each table cell (model-metric pair).

    Use this for benchmarks where each table CELL generates its own bar plot
    (e.g., clicking MACE's ω_max shows a bar for that specific model-metric
    pair). For benchmarks where clicking a COLUMN shows all models on one bar
    (like S24 or OC157), use @plot_parity instead.

    This decorator generates complete Plotly figures during analysis instead of
    saving raw points for the app to process on-the-fly.

    Parameters
    ----------
    filename
        Path where JSON data with pre-made figures will be saved.
    x_label
        Label for x-axis (typically "Score"). Default is None.
    y_label
        Label for y-axis (typically "Names of Variables"). Default is None.
    title_template
        Template for plot titles with {model} and {metric} placeholders.
        Default is "{model} - {metric}".

    Returns
    -------
    Callable
        Decorator that wraps analysis functions to pre-generate bar figures.
    """

    def decorator(func: Callable) -> Callable:
        """
        Wrap the decorated callable to pre-generate bar figures.

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
            Execute func and generate bar plots for each model-metric pair.

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
                    # Extract ref and pred values

                    labels = [minfo["label"] for minfo in metric_info]
                    values = [minfo["value"] for minfo in metric_info]

                    # Build hovertemplate
                    hovertemplate = (
                        "<b>Score: </b>%{x}<br><b>Ref: </b>%{y}<br>"
                        "<b>Label: </b>%{customdata[0]}<br>"
                    )
                    # Create figure
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=labels,
                            y=values,
                            customdata=[
                                [lab, minfo["data"]]
                                for lab, minfo in zip(labels, metric_info, strict=False)
                            ],
                            hovertemplate=hovertemplate,
                            showlegend=False,
                        )
                    )
                    # Set ylimits as 0 to 105
                    fig.update_yaxes(range=[0, 105])

                    # Update layout
                    metric_label = metric_labels.get(metric_key, metric_key)
                    title = title_template.format(model=model_name, metric=metric_label)
                    fig.update_layout(
                        title={"text": title},
                        xaxis={"title": {"text": x_label}},
                        yaxis={"title": {"text": y_label}},
                    )

                    # Store figure as JSON-serializable dict
                    models_data[model_name]["figures"][metric_key] = json.loads(
                        fig.to_json()
                    )

            # Save to file
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "w", encoding="utf8") as f:
                dump(data_bundle, f)

            return data_bundle

        return wrapper

    return decorator


def plot_hist(
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    hoverdata: dict | None = None,
    filename: str = "histogram.json",
    nbins: int = 50,
    bin_range: tuple[float, float] | None = None,
) -> Callable:
    """
    Plot histogram of MLIP results against reference data.

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
    nbins
        Number of bins for histogram. Default is 50.
    bin_range
        Tuple specifying the range of bins (min, max). Default is `None`
        (automatically determined).

    Returns
    -------
    Callable
        Decorator to wrap function.
    """

    def plot_hist_decorator(func: Callable) -> Callable:
        """
        Decorate function to plot histogram.

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
        def plot_hist_wrapper(*args, **kwargs) -> dict[str, Any]:
            """
            Wrap function to plot histogram.

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

            hovertemplate = "<b>Bin: </b>%{x}<br>" + "<b>Density: </b>%{y}<br>"

            customdata = []
            if hoverdata:
                for i, key in enumerate(hoverdata):
                    hovertemplate += f"<b>{key}: </b>%{{customdata[{i}]}}<br>"
                customdata = list(zip(*hoverdata.values(), strict=True))

            fig = go.Figure()
            for mlip, value in results.items():
                fig.add_trace(
                    go.Scatter(
                        x=value["bins"],
                        y=value["hist"],
                        name=mlip,
                        mode="lines",
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

        return plot_hist_wrapper

    return plot_hist_decorator
