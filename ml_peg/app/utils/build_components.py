"""Utility functions for building app components."""

from __future__ import annotations

from dash import html
from dash.dash_table import DataTable
from dash.dcc import Input as DCC_Input
from dash.dcc import Slider, Store
from dash.development.base_component import Component
from dash.html import H2, H3, Br, Button, Details, Div, Label, Summary

from ml_peg.app.utils.register_callbacks import (
    register_summary_table_callbacks,
    register_tab_table_callbacks,
    register_weight_callbacks,
)


def build_slider(
    label: str, slider_id: str, input_id: str, default_value: float | None
) -> Div:
    """
    Build slider and input box.

    Parameters
    ----------
    label
        Slider label.
    slider_id
        ID for slider component.
    input_id
        ID for text box input component.
    default_value
        Default value for slider/text box input.

    Returns
    -------
    Div
        Slider and input text box.
    """
    return Div(
        [
            Label(label),
            Div(
                [
                    Div(
                        Slider(
                            id=slider_id,
                            min=0,
                            max=5,
                            step=0.1,
                            value=default_value,
                            tooltip={"always_visible": False},
                            marks=None,
                        ),
                        style={"flex": "1 1 80%"},
                    ),
                    DCC_Input(
                        id=input_id,
                        type="number",
                        value=default_value,
                        step=0.1,
                        style={"width": "80px"},
                    ),
                ],
                style={"display": "flex", "gap": "10px", "alignItems": "center"},
            ),
        ]
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
    layout = [Br(), Div(header), Br()]

    # Identify metric columns (exclude reserved columns)
    reserved = {"MLIP", "Score", "Rank", "id"}
    columns = [col["id"] for col in table.columns if col.get("id") not in reserved]

    if not columns:
        return Div()

    input_ids = [f"{table.id}-{col}" for col in columns]

    for column, input_id in zip(columns, input_ids, strict=True):
        layout.append(
            build_slider(
                label=column,
                slider_id=f"{input_id}-slider",
                input_id=f"{input_id}-input",
                default_value=None,  # Set by stored value/default
            )
        )

    layout.extend(
        [
            Button(
                "Reset Weights",
                id=f"{table.id}-reset-button",
                n_clicks=0,
                style={"marginTop": "20px"},
            ),
            Store(
                id=f"{table.id}-weight-store",
                storage_type="session",
                data=dict.fromkeys(columns, 1.0),
            ),
        ]
    )

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
