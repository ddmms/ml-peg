"""Helpers to load components into Dash app."""

from __future__ import annotations

import json
from pathlib import Path

from dash.dash_table import DataTable
from dash.dcc import Graph
from plotly.io import read_json

from ml_peg.analysis.utils.utils import get_table_style


def calculate_column_widths(
    columns: list[str],
    widths: dict[str, float] | None = None,
    *,
    char_width: int = 9,
    padding: int = 40,
    min_metric_width: int = 140,
) -> dict[str, int]:
    """
    Calculate column widths based on column titles with minimum width enforcement.

    Parameters
    ----------
    columns
        List of column names from DataTable.
    widths
        Dictionary of column widths. Default is {}.
    char_width
        Approximate pixel width per character.
    padding
        Extra padding to add to calculated width.
    min_metric_width
        Minimum width for metric columns in pixels.

    Returns
    -------
    dict[str, int]
        Mapping of column IDs to pixel widths.
    """
    widths = widths if widths else {}
    # Fixed widths for static columns
    widths.setdefault("MLIP", 150)
    widths.setdefault("Score", 100)
    widths.setdefault("Rank", 100)

    for col in columns:
        if col not in ("MLIP", "Score", "Rank"):
            # Calculate width based on column title length
            calculated_width = len(col) * char_width + padding
            # Enforce minimum width
            widths.setdefault(col, max(calculated_width, min_metric_width))

    return widths


def clean_thresholds(
    raw_thresholds: dict[str, dict[str, float]],
) -> dict[str, tuple[float, float]] | None:
    """
    Convert a raw normalization mapping into ``(good, bad)`` float tuples.

    Parameters
    ----------
    raw_thresholds
        Raw normalization structure read from json file.

    Returns
    -------
    dict[str, tuple[float, float]] | None
        Cleaned mapping or ``None`` when conversion fails.
    """
    if not isinstance(raw_thresholds, dict):
        return None

    thresholds = {}

    for metric, bounds in raw_thresholds.items():
        try:
            if isinstance(bounds, dict):
                good_val = float(bounds["good"])
                bad_val = float(bounds["bad"])
            elif isinstance(bounds, list | tuple) and len(bounds) == 2:
                good_val = float(bounds[0])
                bad_val = float(bounds[1])
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue

        thresholds[metric] = (good_val, bad_val)
    return thresholds


def rebuild_table(
    filename: str | Path,
    id: str,
) -> DataTable:
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
    tooltip_header = table_json["tooltip_header"]

    style = get_table_style(data)
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
