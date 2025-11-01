"""Helpers to load components into Dash app."""

from __future__ import annotations

import json
from pathlib import Path

from dash.dash_table import DataTable
from dash.dcc import Graph
from plotly.io import read_json

from ml_peg.analysis.utils.utils import calc_metric_scores, get_table_style
from ml_peg.app.utils.utils import (
    build_level_of_theory_warnings,
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
    thresholds = clean_thresholds(table_json.get("thresholds"))
    if thresholds is None or not thresholds:
        raise ValueError(f"No thresholds defined in table JSON: {filename}")

    width_labels: list[str] = []

    for column in columns:
        column_id = column.get("id")
        column_name = column.get("name", column_id)
        label_source = column_id if column_id is not None else column_name
        if not isinstance(label_source, str):
            label_source = str(label_source) if label_source is not None else ""
        width_labels.append(label_source)
        if column_id is None:
            continue
        if column_id == "Rank":
            column["type"] = "numeric"
            column.setdefault("format", rank_format())
        elif column.get("type") == "numeric" or is_numeric_column(data, column_id):
            column["type"] = "numeric"
            column.setdefault("format", sig_fig_format())

        base_name = column_name if isinstance(column_name, str) else None

        # Append unit labels to display names when available
        if base_name and thresholds and column_id in thresholds:
            unit_val = thresholds[column_id].get("unit")
            if unit_val and f"[{unit_val}]" not in base_name:
                column["name"] = f"{base_name} [{unit_val}]"

    tooltip_header = table_json["tooltip_header"]

    scored_data = calc_metric_scores(data)
    style = get_table_style(data, scored_data=scored_data)
    column_widths = calculate_column_widths(width_labels)
    model_levels = table_json.get(
        "model_levels_of_theory", table_json.get("model_levels") or {}
    )
    metric_levels = table_json.get(
        "metric_levels_of_theory", table_json.get("metric_levels") or {}
    )
    model_configs = table_json.get("model_configs") or {}
    warning_styles, tooltip_rows = build_level_of_theory_warnings(
        data, model_levels, metric_levels, model_configs
    )
    style_with_warnings = style + warning_styles

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
        style_data_conditional=style_with_warnings,
        style_cell_conditional=style_cell_conditional,
        sort_action="native",
        persistence=True,
        persistence_type="session",
        persisted_props=["data"],
    )

    table.thresholds = thresholds
    table.model_levels_of_theory = model_levels
    table.metric_levels_of_theory = metric_levels
    table.model_configs = model_configs
    table.tooltip_data = tooltip_rows

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
