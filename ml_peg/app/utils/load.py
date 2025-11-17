"""Helpers to load components into Dash app."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from dash.dash_table import DataTable
from dash.dcc import Graph
from plotly.io import read_json

from ml_peg.analysis.utils.utils import calc_metric_scores, get_table_style
from ml_peg.app.utils.utils import (
    calculate_column_widths,
    clean_thresholds,
    clean_weights,
    is_numeric_column,
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
    if not thresholds:
        raise ValueError(f"No thresholds defined in table JSON: {filename}")

    width_labels: list[str] = []

    for column in columns:
        column_id = column.get("id")
        column_name = column.get("name", column_id)
        label_source = column_id if column_id is not None else column_name
        if not isinstance(label_source, str):
            raise TypeError(
                "Column identifiers must be strings. "
                f"Encountered {label_source!r} in {filename}."
            )
        width_labels.append(label_source)
        if column_id is None:
            continue
        if column.get("type") == "numeric" or is_numeric_column(data, column_id):
            column["type"] = "numeric"
            column.setdefault("format", sig_fig_format())
        if column_name is not None and not isinstance(column_name, str):
            raise TypeError(
                "Column display names must be strings. "
                f"Encountered {column_name!r} in {filename}."
            )
        base_name = column_name

        # Append unit labels to display names when available
        if base_name and column_id in thresholds:
            unit_val = thresholds[column_id].get("unit")
            if unit_val and f"[{unit_val}]" not in base_name:
                column["name"] = f"{base_name} [{unit_val}]"

    tooltip_header = table_json["tooltip_header"]

    scored_data = calc_metric_scores(data, thresholds)
    style = get_table_style(data, scored_data=scored_data)
    column_widths = calculate_column_widths(width_labels)

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
    weights = clean_weights(table_json.get("weights"))
    if not thresholds or not weights:
        raise ValueError(f"No thresholds defined in table JSON: {filename}")

    table.thresholds = thresholds
    table.weights = weights

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


def _filter_density_figure_for_model(fig_dict: dict, model: str) -> dict:
    """
    Filter a density-plot figure dict to a single model trace.

    Keeps the y=x reference line and swaps to the annotation matching the model,
    using metadata stored by ``plot_density_scatter``.

    Parameters
    ----------
    fig_dict
        Figure dictionary loaded from saved density-plot JSON.
    model
        Model name to keep visible in the filtered figure.

    Returns
    -------
    dict
        Filtered figure dictionary with only the requested model trace and reference
        line.
    """
    data = fig_dict.get("data", [])
    layout = deepcopy(fig_dict.get("layout", {}))
    annotations_meta = (
        layout.get("meta") if isinstance(layout.get("meta"), dict) else {}
    )

    fig_data = []
    for trace in data:
        name = trace.get("name")
        if name is None:
            line_trace = deepcopy(trace)
            line_trace["visible"] = True
            line_trace["showlegend"] = False  # keep reference line, no legend
            fig_data.append(line_trace)
        elif name == model:
            model_trace = deepcopy(trace)
            model_trace["visible"] = True
            model_trace["showlegend"] = False  # hide legend to avoid overlap
            fig_data.append(model_trace)

    # Pick the matching annotation when available; otherwise keep a simple fallback.
    chosen_annotation = None
    stored_annotations = annotations_meta.get("annotations")
    model_order = annotations_meta.get("models")
    if isinstance(stored_annotations, list) and isinstance(model_order, list):
        try:
            idx = model_order.index(model)
            if idx < len(stored_annotations):
                chosen_annotation = stored_annotations[idx]
        except ValueError:
            pass
    if chosen_annotation:
        layout["annotations"] = [chosen_annotation]
    elif layout.get("annotations"):
        fallback = deepcopy(layout["annotations"][0])
        if isinstance(fallback, dict):
            fallback["text"] = model
            layout["annotations"] = [fallback]

    # Hide legend entirely to prevent overlap with the density colorbar.
    layout["showlegend"] = False

    return {"data": fig_data, "layout": layout}


def read_density_plot_for_model(
    filename: str | Path, model: str, id: str = "figure-1"
) -> Graph:
    """
    Read a density-plot JSON and return a Graph filtered to a single model.

    Parameters
    ----------
    filename
        Path to saved density-plot JSON.
    model
        Model name to keep visible in the returned figure.
    id
        Dash component id for the Graph.

    Returns
    -------
    Graph
        Dash Graph displaying only the requested model (plus reference line).
    """
    with open(filename) as f:
        fig_dict = json.load(f)
    return Graph(id=id, figure=_filter_density_figure_for_model(fig_dict, model))
