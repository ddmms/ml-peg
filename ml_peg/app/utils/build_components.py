"""Utility functions for building app components."""

from __future__ import annotations

from dash import html
from dash.dash_table import DataTable
from dash.dcc import Checklist, Store
from dash.dcc import Input as DCC_Input
from dash.development.base_component import Component
from dash.html import H2, H3, Br, Button, Details, Div, Label, Summary

from ml_peg.app.utils.register_callbacks import (
    register_normalization_callbacks,
    register_summary_table_callbacks,
    register_tab_table_callbacks,
    register_weight_callbacks,
)


def _default_width(label: str, *, base: int = 120) -> int:
    """
    Return a heuristic width based on the column label.

    Parameters
    ----------
    label : str
        Column header used to estimate the width.
    base : int, optional
        Minimum pixel width that the heuristic will return.

    Returns
    -------
    int
        Estimated pixel width for the column.
    """
    approx = len(label) * 9 + 40
    return max(base, approx)


def _resolved_column_widths(
    metrics: list[str],
    column_widths: dict[str, int] | None,
) -> dict[str, int]:
    """
    Build a complete width mapping for MLIP, metric, Score, and Rank columns.

    Parameters
    ----------
    metrics : list[str]
        Ordered metric column names to include in the grid.
    column_widths : dict[str, int] or None
        Optional explicit widths keyed by column name.

    Returns
    -------
    dict[str, int]
        Mapping from column names to resolved pixel widths.
    """
    resolved: dict[str, int] = {}
    if column_widths:
        for key, value in column_widths.items():
            if isinstance(value, int | float):
                resolved[key] = int(value)

    resolved.setdefault("MLIP", resolved.get("MLIP", 150))
    resolved.setdefault("Score", resolved.get("Score", 100))
    resolved.setdefault("Rank", resolved.get("Rank", 100))

    for metric in metrics:
        if metric not in resolved:
            resolved[metric] = _default_width(metric)

    return resolved


def _grid_template_from_widths(
    resolved_widths: dict[str, int],
    column_order: list[str],
) -> str:
    """
    Compose a CSS grid template string from resolved column widths.

    Parameters
    ----------
    resolved_widths : dict[str, int]
        Mapping of column names to pixel widths.
    column_order : list[str]
        Ordered metric column names to render between the MLIP, Score, and Rank columns.

    Returns
    -------
    str
        CSS grid template definition using `minmax` tracks.
    """
    tracks: list[tuple[str, int]] = [("MLIP", resolved_widths["MLIP"])]
    tracks.extend((col, resolved_widths[col]) for col in column_order)
    tracks.append(("Score", resolved_widths["Score"]))
    tracks.append(("Rank", resolved_widths["Rank"]))

    template_parts: list[str] = []
    for _, width in tracks:
        min_px = max(width, 40)
        weight = max(width / 10, 1)
        template_parts.append(f"minmax({int(min_px)}px, {weight:.3f}fr)")
    return " ".join(template_parts)


def build_weight_input(
    input_id: str,
    default_value: float | None,
    *,
    cell_width: int | None = None,
) -> Div:
    """
    Build numeric input for a metric weight.

    Parameters
    ----------
    input_id
        ID for text box input component.
    default_value
        Default value for the text box input.
    cell_width
        Optional width hint retained for signature compatibility; unused.

    Returns
    -------
    Div
        Div wrapping the input box.
    """
    wrapper_style: dict[str, str] = {
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "boxSizing": "border-box",
        "border": "1px solid transparent",
    }
    wrapper_style.update(
        {
            "width": "100%",
            "minWidth": "0",
            "maxWidth": "100%",
            "height": "100%",
        }
    )

    return Div(
        DCC_Input(
            id=input_id,
            type="number",
            value=default_value,
            step=0.1,
            style={
                "width": "80px",
                "fontSize": "12px",
                "padding": "2px 4px",
                "border": "1px solid #6c757d",
                "borderRadius": "3px",
                "textAlign": "center",
            },
        ),
        style=wrapper_style,
    )


def build_weight_components(
    header: str,
    table: DataTable,
    *,
    use_threshold_store: bool = False,
    column_widths: dict[str, int] | None = None,
) -> Div:
    """
    Build weight sliders, text boxes and reset button.

    Parameters
    ----------
    header
        Header for above sliders.
    table
        DataTable to build weight components for.
    use_threshold_store
        Whether this table also exposes normalization thresholds. When True,
        weight callbacks will reuse the raw-data store and normalization store to
        recompute Scores consistently.
    column_widths
        Optional mapping of table column IDs to pixel widths used to align the
        inputs with the rendered table.

    Returns
    -------
    Div
        Div containing header, weight sliders, text boxes and reset button.
    """
    # Identify metric columns (exclude reserved columns)
    reserved = {"MLIP", "Score", "Rank", "id"}
    columns = [col["id"] for col in table.columns if col.get("id") not in reserved]

    if not columns:
        return Div()

    input_ids = [f"{table.id}-{col}" for col in columns]

    resolved_widths = _resolved_column_widths(columns, column_widths)
    grid_template = _grid_template_from_widths(resolved_widths, columns)

    weight_inputs = [
        build_weight_input(
            input_id=f"{input_id}-input",
            default_value=1.0,
        )
        for _, input_id in zip(columns, input_ids, strict=True)
    ]

    container = Div(
        [
            Div(
                [
                    Div(
                        header,
                        style={
                            "fontWeight": "bold",
                            "fontSize": "13px",
                            "padding": "2px 4px",
                            "color": "#212529",
                            "whiteSpace": "nowrap",
                            "boxSizing": "border-box",
                            "border": "1px solid transparent",
                        },
                    ),
                    Button(
                        "Reset Weights",
                        id=f"{table.id}-reset-button",
                        n_clicks=0,
                        style={
                            "fontSize": "11px",
                            "padding": "4px 8px",
                            "marginTop": "6px",
                            "backgroundColor": "#6c757d",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "3px",
                            "width": "fit-content",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-start",
                    "gap": "4px",
                    "boxSizing": "border-box",
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "border": "1px solid transparent",  # #dee2e6 or transparent
                },
            ),
            *weight_inputs,
            Div(
                "",
                style={
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            ),
            Div(
                "",
                style={
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": grid_template,
            "alignItems": "start",
            "columnGap": "0px",
            "rowGap": "4px",
            "marginTop": "8px",
            "padding": "10px 12px",
            "backgroundColor": "#f8f9fa",
            "border": "1px solid #dee2e6",
            "borderRadius": "6px",
            "width": "100%",
            "minWidth": "0",
            "boxSizing": "border-box",
        },
    )

    layout = [
        Br(),
        container,
        Store(
            id=f"{table.id}-weight-store",
            storage_type="session",
            data=dict.fromkeys(columns, 1.0),
        ),
    ]

    # Callbacks to update table scores when table weight dicts change
    if table.id != "summary-table":
        register_tab_table_callbacks(
            table_id=table.id, use_threshold_store=use_threshold_store
        )
    else:
        register_summary_table_callbacks()

    # Callbacks to sync sliders, text boxes, and stored table weights
    for column, input_id in zip(columns, input_ids, strict=True):
        register_weight_callbacks(input_id=input_id, table_id=table.id, column=column)

    return Div(layout)


def build_test_layout(
    name: str,
    description: str,
    table: DataTable,
    extra_components: list[Component] | None = None,
    docs_url: str | None = None,
    column_widths: dict[str, int] | None = None,
    normalization_ranges: dict[str, tuple[float, float]] | None = None,
) -> Div:
    """
    Build app layout for a test.

    Parameters
    ----------
    name
        Name of test.
    description
        Description of test.
    table
        Dash Table with metric results.
    extra_components
        List of Dash Components to include after the metrics table.
    docs_url
        URL to online documentation. Default is None.
    column_widths
        Optional column-width mapping inferred from analysis output. Used to align
        threshold controls beneath the table columns when available.
    normalization_ranges
        Optional normalization metadata (metric -> (good, bad)) supplied via the
        analysis pipeline. When provided, inline threshold controls are rendered
        automatically.

    Returns
    -------
    Div
        Layout for test layout.
    """
    layout_contents = [
        H2(name, style={"color": "black"}),
        H3(description),
    ]

    layout_contents.extend(
        [
            Details(
                [
                    Summary(
                        "Click for more information",
                        style={
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "padding": "5px",
                        },
                    ),
                    Label(
                        [html.A("Online documentation", href=docs_url, target="_blank")]
                    ),
                ],
                style={
                    # "border": "1px solid #ddd",
                    "padding": "10px",
                    # "borderRadius": "5px",
                },
            ),
            Br(),
        ]
    )

    layout_contents.append(Div(table))
    layout_contents.append(
        Store(
            id=f"{table.id}-computed-store",
            storage_type="session",
            data=table.data,
        )
    )

    # Inline normalization thresholds when metadata is supplied
    if normalization_ranges is not None:
        reserved = {"MLIP", "Score", "Rank", "id"}
        metric_columns = [
            col["id"] for col in table.columns if col.get("id") not in reserved
        ]
        layout_contents.append(
            Store(
                id=f"{table.id}-raw-data-store",
                storage_type="session",
                data=table.data,
            )
        )
        threshold_controls = build_threshold_inputs_under_table(
            table_columns=metric_columns,
            normalization_ranges=normalization_ranges,
            table_id=table.id,
            column_widths=column_widths,
        )
        layout_contents.append(threshold_controls)

    # Add metric-weight controls for every benchmark table
    metric_weights = build_weight_components(
        header="Metric Weights",
        table=table,
        use_threshold_store=(normalization_ranges is not None),
        column_widths=column_widths,
    )
    if metric_weights:
        layout_contents.append(metric_weights)

    layout_contents.append(
        Store(
            id="summary-table-scores-store",
            storage_type="session",
        ),
    )

    if extra_components:
        layout_contents.extend(extra_components)

    return Div(layout_contents)


def build_threshold_input(
    metric_name: str, x_default: float, y_default: float, table_id: str
) -> Div:
    """
    Build threshold input boxes for metric normalization.

    Parameters
    ----------
    metric_name
        Name of the metric.
    x_default
        Default value for X threshold (upper bound).
    y_default
        Default value for Y threshold (lower bound).
    table_id
        ID for associated table.

    Returns
    -------
    Div
        Div containing X and Y threshold inputs.
    """
    return Div(
        [
            Label(f"{metric_name}:", style={"fontWeight": "bold"}),
            Div(
                [
                    Div(
                        [
                            Label("Good:", style={"fontSize": "12px"}),
                            DCC_Input(
                                id=f"{table_id}-{metric_name}-good-threshold",
                                type="number",
                                value=x_default,
                                step=0.001,
                                style={"width": "80px", "marginLeft": "5px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    Div(
                        [
                            Label("Bad:", style={"fontSize": "12px"}),
                            DCC_Input(
                                id=f"{table_id}-{metric_name}-bad-threshold",
                                type="number",
                                value=y_default,
                                step=0.001,
                                style={"width": "80px", "marginLeft": "5px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                style={"display": "flex", "gap": "10px"},
            ),
        ],
        style={
            "marginBottom": "10px",
            "padding": "5px",
            "border": "1px solid #ddd",
            "marginLeft": "auto",
            "marginRight": "auto",
            "width": "fit-content",
        },
    )


def build_threshold_inputs_under_table(
    table_columns: list[str],
    normalization_ranges: dict[str, tuple[float, float]],
    table_id: str,
    column_widths: dict[str, int] | None = None,
) -> Div:
    """
    Build inline Good/Bad threshold inputs aligned to the table columns.

    Parameters
    ----------
    table_columns : list[str]
        Ordered metric column names in the table.
    normalization_ranges : dict[str, tuple[float, float]]
        Default (good, bad) threshold ranges keyed by metric column.
    table_id : str
        Identifier prefix for threshold inputs and related controls.
    column_widths : dict[str, int] or None
        Optional pixel widths used to align the grid with the table columns.

    Returns
    -------
    Div
        Container with threshold inputs and associated controls.
    """
    resolved_widths = _resolved_column_widths(table_columns, column_widths)
    grid_template = _grid_template_from_widths(resolved_widths, table_columns)

    container_style = {
        "display": "grid",
        "gridTemplateColumns": grid_template,
        "alignItems": "start",
        "justifyItems": "center",
        "columnGap": "0px",
        "rowGap": "0px",
        "marginTop": "10px",
        "padding": "4px 8px",
        "backgroundColor": "#f8f9fa",
        "border": "1px solid #dee2e6",
        "borderRadius": "5px",
        "width": "100%",
        "minWidth": "0",
        "boxSizing": "border-box",
    }

    cells: list[Div] = []
    defaults: dict[str, tuple[float, float]] = {}

    cells.append(
        Div(
            [
                Div(
                    "Thresholds",
                    style={
                        "fontWeight": "bold",
                        "fontSize": "13px",
                        "padding": "2px 4px",
                        "whiteSpace": "nowrap",
                        "boxSizing": "border-box",
                    },
                ),
                Button(
                    "Reset",
                    id=f"{table_id}-reset-thresholds-button",
                    n_clicks=0,
                    style={
                        "fontSize": "11px",
                        "padding": "4px 8px",
                        "marginTop": "4px",
                        "backgroundColor": "#6c757d",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "3px",
                        "width": "fit-content",
                    },
                ),
                # Toggle to view normalized metric values in the table
                Checklist(
                    id=f"{table_id}-normalized-toggle",
                    options=[{"label": "Show normalized values", "value": "norm"}],
                    value=[],
                    style={"marginTop": "6px", "fontSize": "11px"},
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "flex-start",
                "justifyContent": "center",
                "padding": "1px 2px",
                "justifySelf": "start",
                "width": "100%",
                "minWidth": "0",
                "maxWidth": "100%",
                "boxSizing": "border-box",
                "border": "1px solid transparent",
            },
        )
    )

    for metric in table_columns:
        raw_bounds = normalization_ranges.get(metric, (None, None))
        try:
            x_val = float(raw_bounds[0])
            y_val = float(raw_bounds[1])
        except (TypeError, ValueError, IndexError):
            x_val, y_val = 0.0, 1.0
        defaults[metric] = (x_val, y_val)

        cells.append(
            Div(
                [
                    Div(
                        [
                            Label(
                                "Good:",
                                style={
                                    "fontSize": "13px",
                                    "color": "lightseagreen",
                                    "textAlign": "right",
                                    "position": "absolute",
                                    "right": "calc(50% + 45px)",
                                },
                            ),
                            DCC_Input(
                                id=f"{table_id}-{metric}-good-threshold",
                                type="number",
                                value=x_val,
                                step=0.001,
                                style={
                                    "width": "80px",
                                    "fontSize": "12px",
                                    "padding": "2px 4px",
                                    "border": "1px solid lightseagreen",
                                    "borderRadius": "3px",
                                    "marginLeft": "auto",
                                    "marginRight": "auto",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "marginBottom": "2px",
                            "position": "relative",
                        },
                    ),
                    Div(
                        [
                            Label(
                                "Bad:",
                                style={
                                    "fontSize": "13px",
                                    "color": "#dc3545",
                                    "textAlign": "right",
                                    "position": "absolute",
                                    "right": "calc(50% + 45px)",
                                },
                            ),
                            DCC_Input(
                                id=f"{table_id}-{metric}-bad-threshold",
                                type="number",
                                value=y_val,
                                step=0.001,
                                style={
                                    "width": "80px",
                                    "fontSize": "12px",
                                    "padding": "2px 4px",
                                    "border": "1px solid #dc3545",
                                    "borderRadius": "3px",
                                    "marginLeft": "auto",
                                    "marginRight": "auto",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "position": "relative",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "padding": "2px 0",
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "height": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            )
        )

    for _ in ("Score", "Rank"):
        cells.append(
            Div(
                "",
                style={
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            )
        )

    store = Store(
        id=f"{table_id}-normalization-store",
        storage_type="session",
        data=defaults,
    )

    # Register callbacks for these metrics, pass defaults for reset
    register_normalization_callbacks(
        table_id,
        table_columns,
        defaults,
        register_toggle=False,
    )

    # Score sync callbacks will be implemented later

    return Div(
        [
            Div(cells, id=f"{table_id}-threshold-grid", style=container_style),
            store,
        ]
    )


def build_normalization_components(
    metrics: list[str],
    normalization_ranges: dict[str, tuple[float, float]],
    table_id: str,
) -> Div:
    """
    Build normalization threshold components for all metrics.

    DEPRECATED: Use build_table_with_threshold_rows instead for inline editing.

    Parameters
    ----------
    metrics
        List of metric names.
    normalization_ranges
        Dictionary mapping metric names to (X, Y) threshold tuples.
    table_id
        ID for associated table.

    Returns
    -------
    Div
        Div containing all threshold inputs and controls.
    """
    layout = [
        Br(),
        H3("Metric Normalization Thresholds"),
        Div(
            "Adjust X (score=1) and Y (score=0) thresholds for each metric. "
            "Values between X and Y are normalized linearly.",
            style={"marginBottom": "15px", "fontStyle": "italic"},
        ),
    ]

    for metric in metrics:
        if metric in normalization_ranges:
            x_default, y_default = normalization_ranges[metric]
            layout.append(build_threshold_input(metric, x_default, y_default, table_id))

    layout.extend(
        [
            Button(
                "Reset Thresholds",
                id=f"{table_id}-reset-thresholds-button",
                n_clicks=0,
                style={"marginTop": "20px"},
            ),
            Store(
                id=f"{table_id}-normalization-store",
                storage_type="session",
                data=normalization_ranges,
            ),
        ]
    )

    # Register normalization callbacks
    register_normalization_callbacks(table_id, metrics, normalization_ranges)

    return Div(layout)


def _controls_table_common(columns, data, table_id_suffix: str) -> DataTable:
    """
    Create a simple Dash DataTable used for threshold and weight controls.

    Parameters
    ----------
    columns
        Column definition dictionaries copied from the benchmark table.
    data
        Row data to populate the controls table.
    table_id_suffix
        Suffix appended to the table identifier for uniqueness.

    Returns
    -------
    DataTable
        Dash table configured with hidden Score/Rank styling.
    """
    style_header_conditional = [
        {"if": {"column_id": "Score"}, "color": "rgba(0,0,0,0)"},
        {"if": {"column_id": "Rank"}, "color": "rgba(0,0,0,0)"},
    ]
    style_cell_conditional = [
        {"if": {"column_id": "Score"}, "color": "rgba(0,0,0,0)"},
        {"if": {"column_id": "Rank"}, "color": "rgba(0,0,0,0)"},
    ]
    return DataTable(
        id=table_id_suffix,
        data=data,
        columns=columns,
        editable=True,
        sort_action="none",
        style_table={"overflowX": "auto"},
        fill_width=False,
        style_header_conditional=style_header_conditional,
        style_cell_conditional=style_cell_conditional,
    )


def build_thresholds_table(
    table_id: str,
    metrics: list[str],
    normalization_ranges: dict[str, tuple[float, float]] | None = None,
) -> DataTable:
    """
    Build a separate thresholds table with Good/Bad rows (no weights).

    Parameters
    ----------
    table_id
        Identifier of the benchmark table the controls align with.
    metrics
        Metric columns that should expose threshold controls.
    normalization_ranges
        Optional default (good, bad) ranges for each metric.

    Returns
    -------
    DataTable
        Dash DataTable with two rows labelled ``Good`` and ``Bad``.
    """
    thresholds_id = f"{table_id}-thresholds"
    norm = normalization_ranges or {}
    good_row = {m: norm.get(m, (0.0, 1.0))[0] for m in metrics}
    bad_row = {m: norm.get(m, (0.0, 1.0))[1] for m in metrics}

    data = [
        {
            "MLIP": "Good",
            "id": "__controls_good__",
            **good_row,
            "Score": "",
            "Rank": "",
        },
        {"MLIP": "Bad", "id": "__controls_bad__", **bad_row, "Score": "", "Rank": ""},
    ]
    columns = (
        [{"name": "", "id": "MLIP"}]
        + [{"name": m, "id": m} for m in metrics]
        + [{"name": "Score", "id": "Score"}, {"name": "Rank", "id": "Rank"}]
    )
    return _controls_table_common(columns, data, thresholds_id)
