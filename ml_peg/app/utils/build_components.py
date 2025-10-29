"""Utility functions for building app components."""

from __future__ import annotations

from dash import html
from dash.dash_table import DataTable
from dash.dcc import Checklist, Store
from dash.dcc import Input as DCC_Input
from dash.development.base_component import Component
from dash.html import H2, H3, Br, Button, Details, Div, Label, Summary

from ml_peg.app.utils.register_callbacks import (
    register_category_table_callbacks,
    register_normalization_callbacks,
    register_summary_table_callbacks,
    register_weight_callbacks,
)
from ml_peg.app.utils.utils import calculate_column_widths


def grid_template_from_widths(
    widths: dict[str, int],
    column_order: list[str],
) -> str:
    """
    Compose a CSS grid template string from column widths.

    Parameters
    ----------
    widths
        Mapping of column names to pixel widths.
    column_order
        Ordered metric column names to render between the MLIP, Score, and Rank columns.

    Returns
    -------
    str
        CSS grid template definition using `minmax` tracks.
    """
    tracks: list[tuple[str, int]] = [("MLIP", widths["MLIP"])]
    tracks.extend((col, widths[col]) for col in column_order)
    tracks.append(("Score", widths["Score"]))
    tracks.append(("Rank", widths["Rank"]))

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
    use_thresholds: bool = False,
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
    use_thresholds
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

    widths = calculate_column_widths(columns, column_widths)
    grid_template = grid_template_from_widths(widths, columns)

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
        register_category_table_callbacks(
            table_id=table.id, use_thresholds=use_thresholds
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
    thresholds: dict[str, tuple[float, float]] | None = None,
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
    thresholds
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
    if thresholds is not None:
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
        threshold_controls = build_threshold_inputs(
            table_columns=metric_columns,
            thresholds=thresholds,
            table_id=table.id,
            column_widths=column_widths,
        )
        layout_contents.append(threshold_controls)

    # Add metric-weight controls for every benchmark table
    metric_weights = build_weight_components(
        header="Metric Weights",
        table=table,
        use_thresholds=True,
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


def build_threshold_inputs(
    table_columns: list[str],
    thresholds: dict[str, tuple[float, float]],
    table_id: str,
    column_widths: dict[str, int] | None = None,
) -> Div:
    """
    Build inline Good/Bad threshold inputs aligned to the table columns.

    Parameters
    ----------
    table_columns : list[str]
        Ordered metric column names in the table.
    thresholds : dict[str, tuple[float, float]]
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
    widths = calculate_column_widths(table_columns, column_widths)
    grid_template = grid_template_from_widths(widths, table_columns)

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
    default_thresholds: dict[str, tuple[float, float]] = {}

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
        raw_bounds = thresholds.get(metric, (None, None))
        x_val = float(raw_bounds[0])
        y_val = float(raw_bounds[1])
        default_thresholds[metric] = (x_val, y_val)

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
        id=f"{table_id}-thresholds-store",
        storage_type="session",
        data=default_thresholds,
    )

    # Register callbacks for these metrics, pass default_thresholds for reset
    register_normalization_callbacks(
        table_id,
        table_columns,
        default_thresholds,
        register_toggle=False,
    )

    return Div(
        [
            Div(cells, id=f"{table_id}-threshold-grid", style=container_style),
            store,
        ]
    )
