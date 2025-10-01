"""Helpers to load components into Dash app."""

from __future__ import annotations

import json
from pathlib import Path

from dash.dash_table import DataTable
from dash.dcc import Graph
from plotly.io import read_json

from ml_peg.analysis.utils.utils import get_table_style


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
    Infer column widths from saved Table JSON by estimating the longest display
    string per column, with sane per-column clamps and optional overrides.

    - MLIP: based on longest model name (bounded by mlip_min/max)
    - Metrics: based on max(header length, max displayed value length), bounded
    - Score/Rank: compact fixed widths

    Returns: (widths map, metric order list)
    """
    with open(filename) as f:
        table_json = json.load(f)

    data = table_json.get("data", [])
    columns = [c.get("id") for c in table_json.get("columns", [])]

    widths: dict[str, int] = {}

    def px_for_len(n: int) -> int:
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
    Return ordered metric columns from a saved table JSON, excluding
    MLIP/Score/Rank.
    """
    with open(filename) as f:
        table_json = json.load(f)
    cols = [c.get("id") for c in table_json.get("columns", [])]
    return [c for c in cols if c not in ("MLIP", "Score", "Rank")]


def rebuild_table(
    filename: str | Path,
    id: str = "table-1",
    column_widths: dict[str, int] | None = None,
    outer_scroll: bool = False,
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
    """
    # Load JSON file
    with open(filename) as f:
        table_json = json.load(f)

    data = table_json["data"]
    columns = table_json["columns"]
    tooltip_header = table_json["tooltip_header"]

    style = get_table_style(data)

    style_cell_conditional = []
    if column_widths:
        # Apply fixed widths so external UI (e.g., thresholds row) can align
        for col, px in column_widths.items():
            style_cell_conditional.append(
                {
                    "if": {"column_id": col},
                    "width": f"{px}px",
                    "minWidth": f"{px}px",
                    "maxWidth": f"{px}px",
                }
            )

    return DataTable(
        data=data,
        columns=columns,
        tooltip_header=tooltip_header,
        tooltip_delay=100,
        tooltip_duration=None,
        editable=True,
        id=id,
        style_data_conditional=style,
        style_cell_conditional=style_cell_conditional,
        # When explicit widths are provided, prevent the table from stretching
        # columns to fill remaining space, which breaks alignment with external UI.
        fill_width=False if column_widths else True,
        # If wrapped in an outer scroll container, keep inner table from scrolling
        style_table={"overflowX": "visible"} if outer_scroll else {"overflowX": "auto"},
        sort_action="native",
    )


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


def build_ag_grid_from_table_json(
    filename: str | Path,
    grid_id: str,
    column_size: str = "responsiveSizeToFit",
):
    """
    Build a Dash AG Grid from a saved DataTable JSON (data/columns).

    Parameters
    ----------
    filename
        Path to the saved table JSON (with keys: data, columns, tooltip_header).
    grid_id
        Component id to assign to the AG Grid.
    column_size
        One of "sizeToFit" or "responsiveSizeToFit".

    Returns
    -------
    Component
        A dash_ag_grid.AgGrid component. If dash-ag-grid is not installed, raises a
        helpful ImportError.
    """
    try:
        import dash_ag_grid as dag  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "dash-ag-grid is required for the AG Grid preview. Install with `pip install dash-ag-grid`."
        ) from exc

    with open(filename) as f:
        table_json = json.load(f)

    data = table_json.get("data", [])
    columns = table_json.get("columns", [])

    # Preserve order and names from DataTable definition
    column_defs = []
    for col in columns:
        # DataTable columns have {"name": header, "id": field}
        field = col.get("id")
        header = col.get("name")
        if field is None:
            continue
        column_defs.append(
            {
                "field": field,
                "headerName": header,
                "sortable": True,
                "filter": False,
                "resizable": True,
                # Keep editable False for preview to avoid confusion
                "editable": False,
            }
        )

    grid = dag.AgGrid(
        id=grid_id,
        rowData=data,
        columnDefs=column_defs,
        columnSize=column_size,
        defaultColDef={
            "resizable": True,
            "sortable": True,
            "filter": False,
            "suppressHeaderMenuButton": True,
        },
        dashGridOptions={
            "ensureDomOrder": True,
            "suppressDragLeaveHidesColumns": True,
        },
        style={"width": "100%"},
    )
    return grid
