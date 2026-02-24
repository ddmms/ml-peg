"""Helpers to load components into Dash app."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from warnings import warn

from dash.dash_table import DataTable
from dash.dcc import Graph
from plotly.io import read_json

from ml_peg.analysis.utils.utils import calc_metric_scores, get_table_style
from ml_peg.app.utils.utils import (
    build_level_of_theory_warnings,
    calculate_column_widths,
    clean_thresholds,
    clean_weights,
    is_numeric_column,
    sig_fig_format,
)
from ml_peg.models.get_models import get_model_names, load_model_configs


def rebuild_table(
    filename: str | Path, id: str, description: str | None = None
) -> DataTable:
    """
    Rebuild saved dash table.

    Parameters
    ----------
    filename
        Name of json file with saved table data.
    id
        ID for table.
    description
        Description of table. Default is None.

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
    model_name_map = dict(table_json.get("model_name_map") or {})
    thresholds = clean_thresholds(table_json.get("thresholds"))
    if not thresholds:
        raise ValueError(f"No thresholds defined in table JSON: {filename}")

    # Pad table with all models from registry (models without data will be
    # grayed/hashed out)
    all_registry_models = get_model_names()
    existing_display_names = {row.get("MLIP") for row in data}

    # Model names come in two forms:
    # - Original model name: Base name from models.yml (e.g., "mace-mp-0a")
    # - Display name: UI name, may have suffix (e.g., "mace-mp-0a-D3")

    # model_name_map stores: display_name -> original model name
    # We need the inverse for lookups: original name -> display name
    original_to_display = {v: k for k, v in model_name_map.items()}

    # Determine which metrics exist (excluding MLIP, Score, id)
    metric_columns = [
        col["id"] for col in columns if col.get("id") not in {"MLIP", "Score", "id"}
    ]

    # Add missing models with None for all metrics
    for original_model in all_registry_models:
        # Convert original model name (from registry) to display name (for table)
        display_name = original_to_display.get(original_model, original_model)
        if display_name not in existing_display_names:
            # Create row with None for all metrics (will appear grayed out) while
            # storing the original model name in ``id`` so callbacks have a stable key
            new_row = {"MLIP": display_name, "id": original_model}
            for metric in metric_columns:
                new_row[metric] = None
            # Score will be None (calculated later by calc_table_scores)
            new_row["Score"] = None
            data.append(new_row)

            # Update model_name_map if this is a new model not in original JSON
            if original_model not in model_name_map.values():
                model_name_map[display_name] = original_model

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
    model_levels = table_json.get("model_levels_of_theory") or {}
    metric_levels = table_json.get("metric_levels_of_theory") or {}
    model_configs = table_json.get("model_configs") or {}

    # Add model configs and levels for newly added models
    # (models that were padded but not in original JSON)
    all_display_names = {row.get("MLIP") for row in data}
    missing_models = []
    for display_name in all_display_names:
        # Convert display name to the original model name for config lookup
        original_name = model_name_map.get(display_name, display_name)
        if display_name not in model_configs:
            missing_models.append(original_name)

    if missing_models:
        # Load configs using the original names (as they appear in models.yml)
        new_configs, new_levels = load_model_configs(missing_models)
        for original_name in missing_models:
            # Convert back to display name for storage in table metadata
            display_name = original_to_display.get(original_name, original_name)
            if original_name in new_configs:
                model_configs[display_name] = new_configs[original_name]
            if original_name in new_levels:
                model_levels[display_name] = new_levels[original_name]

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
        editable=False,
        id=id,
        style_data_conditional=style_with_warnings,
        style_cell_conditional=style_cell_conditional,
        style_header={
            "whiteSpace": "normal",
            "height": "auto",
            "minHeight": "70px",
            "textAlign": "center",
            "verticalAlign": "middle",
            "lineHeight": "1.4",
            "padding": "8px",
        },
        style_header_conditional=[
            {
                "if": {"column_id": "MLIP"},
                "textAlign": "left",
            }
        ],
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
    table.description = description
    table.model_levels_of_theory = model_levels
    table.metric_levels_of_theory = metric_levels
    table.model_configs = model_configs
    table.tooltip_data = tooltip_rows
    table.model_name_map = model_name_map

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
    figure = read_json(filename) if Path(filename).exists() else None
    return Graph(id=id, figure=figure)


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
    layout = deepcopy(fig_dict.get("layout"))
    annotations_meta = layout.get("meta")

    fig_data = []
    for trace in data:
        name = trace.get("name")
        if name is None or name == model:
            # ``name`` is ``None`` for the y=x reference line; keep that and the
            # requested model trace visible while hiding their legend entries.
            trace_copy = deepcopy(trace)
            trace_copy["visible"] = True
            trace_copy["showlegend"] = False
            fig_data.append(trace_copy)

    # Pick the matching annotation (Plotly layout annotation with MAE/exclusion text)
    stored_annotations = (
        annotations_meta.get("annotations") if annotations_meta else None
    )
    model_order = annotations_meta.get("models") if annotations_meta else None
    chosen_annotation = None
    if isinstance(stored_annotations, list) and isinstance(model_order, list):
        try:
            idx = model_order.index(model)
            if idx < len(stored_annotations):
                chosen_annotation = stored_annotations[idx]
        except ValueError:
            pass
    if chosen_annotation:
        layout["annotations"] = [chosen_annotation]

    # Hide legend entirely to prevent overlap with the density colorbar.
    layout["showlegend"] = False

    return {"data": fig_data, "layout": layout}


def read_density_plot_for_model(
    filename: str | Path, model: str, id: str = "figure-1"
) -> Graph | None:
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
    Graph | None
        Dash Graph displaying only the requested model (plus reference line).
        Returns None if the model has no data in the plot.
    """
    with open(filename) as f:
        fig_dict = json.load(f)

    filtered_fig = _filter_density_figure_for_model(fig_dict, model)

    # Check if model has actual data (not just the reference line)
    # If only 1 trace (the y=x line) or 0 traces, model has no data
    if len(filtered_fig.get("data", [])) <= 1:
        warn(f"No model data found for {model}", stacklevel=2)
        return None

    return Graph(id=id, figure=filtered_fig)
