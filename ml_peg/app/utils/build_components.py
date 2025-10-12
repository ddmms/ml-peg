"""Utility functions for building app components."""

from __future__ import annotations

from dash import html
from dash.dash_table import DataTable
from dash.dcc import Input as DCC_Input
from dash.dcc import Store
from dash.development.base_component import Component
from dash.html import H2, H3, Br, Button, Details, Div, Label, Summary

from ml_peg.app.utils.register_callbacks import (
    register_summary_table_callbacks,
    register_tab_table_callbacks,
    register_weight_callbacks,
)


def build_weight_input(input_id: str, default_value: float | None) -> Div:
    """
    Build numeric input for a metric weight.

    Parameters
    ----------
    input_id
        ID for text box input component.
    default_value
        Default value for the text box input.

    Returns
    -------
    Div
        Div wrapping the input box.
    """
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
        style={"display": "flex", "justifyContent": "center"},
    )


def build_weight_components(
    header: str,
    table: DataTable,
) -> Div:
    """
    Build weight sliders, text boxes and reset button.

    Parameters
    ----------
    header
        Header for above sliders.
    table
        DataTable to build weight components for.

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

    weight_inputs = [
        build_weight_input(
            input_id=f"{input_id}-input",
            default_value=1.0,
        )
        for input_id in input_ids
    ]

    column_count = len(weight_inputs)
    metric_columns_template = " ".join(["minmax(80px, 1fr)"] * column_count)
    grid_template = f"minmax(140px, auto) {metric_columns_template}".strip()

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
                },
            ),
            *weight_inputs,
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": grid_template,
            "alignItems": "start",
            "columnGap": "8px",
            "rowGap": "4px",
            "marginTop": "8px",
            "padding": "10px 12px",
            "backgroundColor": "#f8f9fa",
            "border": "1px solid #dee2e6",
            "borderRadius": "6px",
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
        register_tab_table_callbacks(table_id=table.id)
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

    # Add metric-weight controls for every benchmark table
    metric_weights = build_weight_components(header="Metric weights", table=table)
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


def build_metric_weight_components(
    table: DataTable, header: str = "Metric weights"
) -> Div:
    """
    Discover metric columns and render weight controls and callbacks.

    Uses `build_weight_components` to create sliders, inputs, a reset button,
    and a weight store tied to the supplied `DataTable`.

    Parameters
    ----------
    table
        Benchmark results DataTable.
    header
        Header label shown above the sliders. Default is "Metric weights".

    Returns
    -------
    Div
        Div containing sliders, inputs, reset button and weight store.
    """
    # Identify metric columns (exclude reserved columns)
    reserved = {"MLIP", "Score", "Rank", "id"}
    metric_columns = [
        col["id"] for col in table.columns if col.get("id") not in reserved
    ]

    if not metric_columns:
        return Div()

    input_ids = [f"{table.id}-{col}" for col in metric_columns]

    return build_weight_components(
        header=header,
        columns=metric_columns,
        input_ids=input_ids,
        table_id=table.id,
    )
