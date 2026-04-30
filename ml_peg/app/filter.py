"""Build data components."""

from __future__ import annotations

from ase.data import chemical_symbols
from dash.dcc import Dropdown
from dash.html import Details, Div, Summary


def get_model_filter(models) -> Details:
    """
    Get model filter component.

    Parameters
    ----------
    models
        List of model names to include in filter options.

    Returns
    -------
    Details
        Model filter component.
    """
    model_options = [{"label": m, "value": m} for m in models]

    return Details(
        [
            Summary(
                "Visible models",
                style={
                    "cursor": "pointer",
                    "fontWeight": "600",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.07em",
                    "color": "#6c757d",
                    "padding": "5px",
                },
            ),
            Div(
                [
                    Dropdown(
                        id="model-filter-checklist",
                        options=model_options,
                        value=models,
                        multi=True,
                        placeholder="Select visible models",
                        closeOnSelect=False,
                        style={"fontSize": "13px"},
                    ),
                ],
                style={"padding": "8px 12px"},
            ),
        ],
        id="model-filter-details",
        open=True,
        style={"marginBottom": "8px", "fontSize": "13px"},
    )


def get_element_filter() -> Details:
    """
    Get element filter component.

    Returns
    -------
    Details
        Element filter component.
    """
    # Exclude placeholder symbol for index 0
    elements = chemical_symbols[1:]

    return Details(
        [
            Summary(
                "Filter by elements",
                style={
                    "cursor": "pointer",
                    "fontWeight": "600",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.07em",
                    "color": "#6c757d",
                    "padding": "5px",
                },
            ),
            Div(
                [
                    Dropdown(
                        id="element-filter",
                        options=elements,
                        value=None,
                        multi=True,
                        placeholder="Filter elements",
                        closeOnSelect=False,
                        style={"fontSize": "13px"},
                        debounce=True,
                    ),
                ],
                style={"padding": "8px 12px"},
            ),
        ],
        id="element-filter-details",
        open=True,
        style={"marginBottom": "8px", "fontSize": "13px"},
    )
