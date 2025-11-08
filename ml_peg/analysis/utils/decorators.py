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

from ml_peg.analysis.utils.utils import calc_ranks, calc_table_scores
from ml_peg.app.utils.utils import Thresholds

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


def build_table(
    *,
    thresholds: Thresholds,
    filename: str = "table.json",
    metric_tooltips: dict[str, str] | None = None,
    normalize: bool = True,
    normalizer: Callable[[float, float, float], float] | None = None,
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
        Tooltips for table metric headers. Defaults are set for "MLIP", "Score", and
        "Rank".
    normalize
        Whether to apply normalisation when calculating the score. Default is True.
    normalizer
        Optional function to map (value, X, Y) -> normalised score. Default is
        ml_peg.analysis.utils.utils.normalize_metric.

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
                "Rank": "Model rank based on score (lower is better)",
            }
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
                        "thresholds": thresholds,
                    },
                    fp,
                )

            return results

        return build_table_wrapper

    return build_table_decorator
