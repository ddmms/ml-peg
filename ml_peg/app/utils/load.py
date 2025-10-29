"""Helpers to load components into Dash app."""

from __future__ import annotations

import json
from pathlib import Path

from dash.dash_table import DataTable
from dash.dcc import Graph
from plotly.io import read_json

from ml_peg.analysis.utils.utils import calc_metric_scores, get_table_style
from ml_peg.app.utils.utils import (
    calculate_column_widths,
    clean_thresholds,
    is_numeric_column,
    rank_format,
    sig_fig_format,
)


def rebuild_table(filename: str | Path, id: str) -> DataTable:
    """
    Rebuild saved dash table.

    Parameters
    ----------
    filename
        Name of json file with saved table data.
    id
        ID for table.

    Returns
    -------
    DataTable
        Loaded Dash DataTable.

    Raises
    ------
    ValueError
        If the table JSON omits required ``thresholds`` metadata.
    """
    # Load JSON file
    with open(filename) as f:
        table_json = json.load(f)

    data = table_json["data"]
    columns = table_json["columns"]

    for column in columns:
        column_id = column.get("id")
        if column_id is None:
            continue
        if column_id == "Rank":
            column["type"] = "numeric"
            column.setdefault("format", rank_format())
        elif column.get("type") == "numeric" or is_numeric_column(data, column_id):
            column["type"] = "numeric"
            column.setdefault("format", sig_fig_format())
    tooltip_header = table_json["tooltip_header"]

    scored_data = calc_metric_scores(data)
    style = get_table_style(data, scored_data=scored_data)
    column_widths = calculate_column_widths([cols["name"] for cols in columns])

    style_cell_conditional: list[dict[str, object]] = []
    for column_id, width in column_widths.items():
        if width is None:
            continue
        col_width = f"{width}px"
        alignment = "left" if column_id == "MLIP" else "center"
        style_cell_conditional.append(
            {
                "if": {"column_id": column_id},
                "width": col_width,
                "minWidth": col_width,
                "maxWidth": col_width,
                "textAlign": alignment,
            }
        )

    table = DataTable(
        data=data,
        columns=columns,
        tooltip_header=tooltip_header,
        tooltip_delay=100,
        tooltip_duration=None,
        editable=True,
        id=id,
        style_data_conditional=style,
        style_cell_conditional=style_cell_conditional,
        sort_action="native",
        persistence=True,
        persistence_type="session",
        persisted_props=["data"],
    )

    thresholds = clean_thresholds(table_json.get("thresholds"))
    if thresholds is None or not thresholds:
        raise ValueError(f"No thresholds defined in table JSON: {filename}")

    table.thresholds = thresholds

    return table


def read_plot(filename: str | Path, id: str = "figure-1") -> Graph:
    """
    Read preprepared plotly Figure.

    Parameters
    ----------
    filename
        Name of json file with saved plot data.
    id
        ID for plot.

    Returns
    -------
    Graph
        Loaded plotly Graph.
    """
    return Graph(id=id, figure=read_json(filename))
