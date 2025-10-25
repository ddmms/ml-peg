"""Helpers to load components into Dash app."""

from __future__ import annotations

import json
from pathlib import Path

from dash.dash_table import DataTable
from dash.dcc import Graph
from plotly.io import read_json

from ml_peg.analysis.utils.utils import get_table_style


def calculate_column_widths(
    columns: list[dict],
    char_width: int = 9,
    padding: int = 40,
    min_metric_width: int = 140,
) -> dict[str, int]:
    """
    Calculate column widths based on column titles with minimum width enforcement.

    Parameters
    ----------
    columns
        List of column dictionaries from DataTable (each has 'id' and 'name').
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
    widths = {}

    for col in columns:
        col_id = col.get("id")
        col_name = col.get("name", col_id)

        # Fixed widths for static columns
        if col_id == "MLIP":
            widths[col_id] = 150
        elif col_id == "Score":
            widths[col_id] = 100
        elif col_id == "Rank":
            widths[col_id] = 100
        else:
            # Calculate width based on column title length
            calculated_width = len(col_name) * char_width + padding
            # Enforce minimum width
            widths[col_id] = max(calculated_width, min_metric_width)

    return widths


def _coerce_thresholds(raw_ranges):
    """
    Convert a raw normalization mapping into ``(good, bad)`` float tuples.

    Parameters
    ----------
    raw_ranges
        Raw normalization structure read from disk.

    Returns
    -------
    dict[str, tuple[float, float]] | None
        Cleaned mapping or ``None`` when conversion fails.
    """
    if not isinstance(raw_ranges, dict):
        return None
    result: dict[str, tuple[float, float]] = {}
    for metric, bounds in raw_ranges.items():
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

        result[metric] = (good_val, bad_val)
    return result


def rebuild_table(
    filename: str | Path,
    id="table-1",
    column_widths: dict[str, int] | None = None,
) -> DataTable:
    """
    Rebuild saved dash table.

    Parameters
    ----------
    filename
        Name of json file with saved table data.
    id
        ID for table.
    column_widths
        Optional column width metadata currently unused but kept for parity.

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

    style_cell_conditional: list[dict[str, object]] = []
    if column_widths:
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

    thresholds = _coerce_thresholds(table_json.get("thresholds"))
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


def infer_column_widths_from_json(
    filename: str | Path,
    char_px: int = 9,
    padding_px: int = 50,
    mlip_min: int = 250,
    mlip_max: int = 400,
    metric_min: int = 250,
    metric_max: int = 400,
    score_w: int = 150,
    rank_w: int = 150,
    min_total: int | None = None,
    overrides: dict[str, int] | None = None,
) -> tuple[dict[str, int], list[str]]:
    """
    Infer column widths from a saved table JSON snapshot.

    Parameters
    ----------
    filename
        Path to the serialized table JSON.
    char_px
        Pixel-per-character estimate for width calculations.
    padding_px
        Additional padding applied to every computed width.
    mlip_min
        Minimum pixel width permitted for the MLIP column.
    mlip_max
        Maximum pixel width permitted for the MLIP column.
    metric_min
        Lower bound on metric column widths.
    metric_max
        Upper bound on metric column widths.
    score_w
        Width assigned to the Score column.
    rank_w
        Width assigned to the Rank column.
    min_total
        Optional minimum total width across all columns.
    overrides
        Optional explicit column width overrides.

    Returns
    -------
    tuple[dict[str, int], list[str]]
        A mapping of column IDs to pixel widths and the ordered metric column list.
    """
    with open(filename) as f:
        table_json = json.load(f)

    data = table_json.get("data", [])
    columns = [c.get("id") for c in table_json.get("columns", [])]

    widths: dict[str, int] = {}

    def px_for_len(n: int) -> int:
        """
        Convert a character length to pixels using heuristic padding.

        Parameters
        ----------
        n
            Number of characters.

        Returns
        -------
        int
            Estimated pixel width.
        """
        return int(n * char_px + padding_px)

    # MLIP width from longest name/header
    mlip_header_len = len("MLIP")
    if data and "MLIP" in data[0]:
        mlip_max_len = max(
            mlip_header_len, max(len(str(row.get("MLIP", ""))) for row in data)
        )
        widths["MLIP"] = max(mlip_min, min(mlip_max, px_for_len(mlip_max_len)))
    else:
        widths["MLIP"] = mlip_min

    # Metric widths and order
    metric_order: list[str] = []
    for col in columns:
        if col in ("MLIP", "Score", "Rank", "id"):
            continue
        metric_order.append(col)

        max_val_len = 0
        for row in data:
            val = row.get(col, "")
            s = str(val)
            max_val_len = max(max_val_len, len(s))

        header_len = len(str(col or ""))
        candidate = max(header_len, max_val_len)
        est_px = px_for_len(candidate)
        widths[col] = max(metric_min, min(metric_max, est_px))

    # Score/Rank
    widths["Score"] = score_w
    widths["Rank"] = rank_w

    # Optional total width target
    if isinstance(min_total, int):
        metric_cols_count = len(metric_order)
        current_total = sum(
            widths.get(c, 0) for c in ["MLIP", *metric_order, "Score", "Rank"]
        )
        if current_total < min_total and metric_cols_count >= 0:
            deficit = min_total - current_total
            add_mlip = int(deficit * 0.6)
            add_each_metric = int(deficit * 0.4 / max(1, metric_cols_count))
            widths["MLIP"] = min(mlip_max, widths["MLIP"] + add_mlip)
            for col in metric_order:
                widths[col] = min(metric_max, widths[col] + add_each_metric)

    if overrides:
        widths.update(overrides)

    return widths, metric_order


def get_metric_columns_from_json(filename: str | Path) -> list[str]:
    """
    Return ordered metric columns from a saved table JSON.

    Parameters
    ----------
    filename
        Path to the serialized table JSON.

    Returns
    -------
    list[str]
        Ordered list of metric column IDs (excluding MLIP/Score/Rank).
    """
    with open(filename) as f:
        table_json = json.load(f)
    cols = [c.get("id") for c in table_json.get("columns", [])]
    return [c for c in cols if c not in ("MLIP", "Score", "Rank")]
