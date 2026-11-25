"""Shared layout helpers for the NMR benchmark apps."""

from __future__ import annotations

import json
from pathlib import Path

from dash import Input, Output, callback
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div

from ml_peg.app.utils.build_callbacks import (
    plot_with_download_controls,
    register_plot_settings_callbacks,
    struct_from_scatter,
)
from ml_peg.app.utils.plot_helpers import INSTRUCTION_STYLE, TABLE_HINT
from ml_peg.app.utils.register_callbacks import register_plot_download_callbacks

TEXT_STYLE = {"textAlign": "justify"}


# Plots for a clicked cell sit side by side, each square, and wrap onto separate
# rows when the page is too narrow for two.
PLOT_ROW = {"display": "flex", "flexWrap": "wrap", "gap": "16px"}
PLOT_CELL = {"flex": "1 1 420px", "maxWidth": "540px", "minWidth": "0"}


def square(graph: Graph) -> Graph:
    """
    Make a plot square, filling the width of its container.

    Parameters
    ----------
    graph
        Plot to resize (changed in place).

    Returns
    -------
    Graph
        The same plot, with a 1:1 aspect ratio and a responsive Plotly layout.
    """
    graph.style = {"width": "100%", "aspectRatio": "1 / 1"}
    graph.config = {"responsive": True}
    figure = graph.figure
    if figure is not None:
        # ``read_plot`` returns a plain dict; build a new one so the cached figure
        # is not modified.
        if hasattr(figure, "to_plotly_json"):
            figure = figure.to_plotly_json()
        layout = {
            key: value
            for key, value in (figure.get("layout") or {}).items()
            if key not in ("height", "width")
        }
        # Legend above the plot, so the plotting area itself stays square.
        layout["legend"] = {
            **layout.get("legend", {}),
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
        }
        layout["margin"] = {"l": 60, "r": 20, "t": 80, "b": 60}
        graph.figure = {**figure, "layout": layout}
    return graph


def plots_from_table_column(
    table_id: str, plot_id: str, column_to_plots: dict[str, list[Graph]]
) -> None:
    """
    Show one or more plots side by side when a table column is clicked.

    As ml-peg's ``plot_from_table_column``, which shows a single plot, so that a
    column can open its parity plot and the shielding-vs-experiment plot.

    Parameters
    ----------
    table_id
        ID of the Dash table being clicked.
    plot_id
        ID of the placeholder Div for the plots.
    column_to_plots
        Table column to the plots to show, in order.
    """
    register_plot_download_callbacks()
    register_plot_settings_callbacks()

    @callback(
        Output(plot_id, "children"),
        Output(table_id, "active_cell"),
        Input(table_id, "active_cell"),
    )
    def show_plots(active_cell) -> Div:
        """
        Show the plots of the clicked column.

        Parameters
        ----------
        active_cell
            Clicked cell in the Dash table.

        Returns
        -------
        Div
            Message explaining interactivity, or the plots on table click.
        """
        if not active_cell:
            return Div(TABLE_HINT, style=INSTRUCTION_STYLE), None
        column_id = active_cell.get("column_id")
        if column_id not in column_to_plots:
            raise PreventUpdate
        return (
            Div(
                [
                    Div(plot_with_download_controls(square(g)), style=PLOT_CELL)
                    for g in column_to_plots[column_id]
                ],
                style=PLOT_ROW,
            ),
            None,
        )


def link_structures(graphs: list[Graph], struct_id: str, links: dict) -> None:
    """
    Open the crystal structure of a clicked point, with its site highlighted.

    Parameters
    ----------
    graphs
        Plots whose points share one order.
    struct_id
        ID of the placeholder Div for the structure viewer.
    links
        ``{"structs": [...], "highlights": [[...], ...]}`` in point order, as
        written by the analysis (``structure_links.json``).
    """
    for graph in graphs:
        struct_from_scatter(
            scatter_id=graph.id,
            struct_id=struct_id,
            structs=links["structs"],
            highlights=links["highlights"],
        )


def read_links(path: Path) -> dict:
    """
    Read the analysis's structure links, if present.

    Parameters
    ----------
    path
        ``structure_links.json``.

    Returns
    -------
    dict
        Plot key to links; empty if the analysis has not been run.
    """
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
