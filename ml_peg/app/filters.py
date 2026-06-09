"""Build data components."""

from __future__ import annotations

from dash import ALL, Input, Output, State, callback, ctx, no_update
from dash.dcc import Dropdown, Store
from dash.exceptions import PreventUpdate
from dash.html import Button, Details, Div, Span, Summary

from ml_peg.analysis.utils.periodic_table import (
    PERIODIC_TABLE_COLS,
    PERIODIC_TABLE_POSITIONS,
    PERIODIC_TABLE_ROWS,
    PERIODIC_TABLE_SYMBOLS,
)


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
                        maxHeight=600,
                        optionHeight=10,
                        placeholder="Select visible models",
                        closeOnSelect=False,
                        style={"fontSize": "12px"},
                    ),
                ],
                style={"padding": "8px 12px"},
            ),
        ],
        id="model-filter-details",
        open=True,
        style={"marginBottom": "8px", "fontSize": "13px"},
    )


_SUMMARY_STYLE = {
    "cursor": "pointer",
    "fontWeight": "600",
    "fontSize": "11px",
    "textTransform": "uppercase",
    "letterSpacing": "0.07em",
    "color": "#6c757d",
    "padding": "5px",
}

_CELL = 22  # px per element cell

_BTN_BASE = {
    "padding": "0",
    "width": f"{_CELL}px",
    "height": f"{_CELL}px",
    "fontSize": "9px",
    "fontWeight": "500",
    "border": "1px solid #ced4da",
    "borderRadius": "2px",
    "cursor": "pointer",
    "lineHeight": f"{_CELL}px",
    "textAlign": "center",
    "backgroundColor": "#e9ecef",
    "color": "#343a40",
    "overflow": "hidden",
}

_BTN_EXCLUDED = {
    **_BTN_BASE,
    "backgroundColor": "#f5c2c7",
    "border": "1px solid #dc3545",
    "color": "#842029",
    "fontWeight": "700",
}

_LEGEND_SWATCH_BASE = {
    "display": "inline-block",
    "width": "14px",
    "height": "14px",
    "borderRadius": "2px",
    "border": _BTN_BASE["border"],
    "backgroundColor": _BTN_BASE["backgroundColor"],
}

_LEGEND_SWATCH_EXCLUDED = {
    **_LEGEND_SWATCH_BASE,
    "border": _BTN_EXCLUDED["border"],
    "backgroundColor": _BTN_EXCLUDED["backgroundColor"],
}


def _btn_style(symbol: str, excluded: bool) -> dict:
    """
    Return grid-positioned button style for a periodic-table element.

    Parameters
    ----------
    symbol
        Chemical symbol of the element button.
    excluded
        Whether the element is selected for exclusion.

    Returns
    -------
    dict
        Dash style dictionary for the element button.
    """
    row, col = PERIODIC_TABLE_POSITIONS[symbol]
    base = _BTN_EXCLUDED if excluded else _BTN_BASE
    return {**base, "gridColumn": str(col + 1), "gridRow": str(row + 1)}


def get_element_filter() -> Div:
    """
    Get element filter component with clickable periodic table.

    Clicking an element marks it for exclusion (highlighted red). The filter
    is applied only when the "Apply" button is clicked. "Exclude all" marks
    every element for exclusion. "Clear" resets the selection. The committed
    selection is held in ``dcc.Store(id="element-filter")``.

    Returns
    -------
    Div
        Wrapper containing the periodic table filter UI and backing Stores.
    """
    buttons = [
        Button(
            sym,
            id={"type": "element-btn", "index": sym},
            n_clicks=0,
            style=_btn_style(sym, False),
        )
        for sym in PERIODIC_TABLE_SYMBOLS
    ]

    grid = Div(
        buttons,
        style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({PERIODIC_TABLE_COLS}, {_CELL}px)",
            "gridTemplateRows": f"repeat({PERIODIC_TABLE_ROWS}, {_CELL}px)",
            "gap": "1px",
        },
    )
    legend = Div(
        [
            Div(
                [
                    Span("Included:", style={"minWidth": "62px"}),
                    Span(style=_LEGEND_SWATCH_BASE),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "6px",
                },
            ),
            Div(
                [
                    Span("Excluded:", style={"minWidth": "62px"}),
                    Span(style=_LEGEND_SWATCH_EXCLUDED),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "6px",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            "fontSize": "12px",
            "color": "#343a40",
            "paddingTop": "2px",
            "whiteSpace": "nowrap",
        },
    )
    grid_with_legend = Div(
        [grid, legend],
        style={
            "display": "flex",
            "alignItems": "flex-start",
            "gap": "12px",
            "flexWrap": "wrap",
        },
    )

    actions = Div(
        [
            Button(
                "Apply",
                id="element-filter-apply",
                n_clicks=0,
                style={
                    "padding": "4px 12px",
                    "fontSize": "12px",
                    "backgroundColor": "#228be6",
                    "color": "#fff",
                    "border": "none",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                },
            ),
            Button(
                "Exclude all",
                id="element-filter-exclude-all",
                n_clicks=0,
                style={
                    "padding": "4px 12px",
                    "fontSize": "12px",
                    "backgroundColor": "#fff5f5",
                    "color": "#842029",
                    "border": "1px solid #f1aeb5",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                },
            ),
            Button(
                "Clear",
                id="element-filter-clear",
                n_clicks=0,
                style={
                    "padding": "4px 12px",
                    "fontSize": "12px",
                    "backgroundColor": "#e9ecef",
                    "color": "#343a40",
                    "border": "1px solid #ced4da",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                },
            ),
        ],
        style={
            "display": "flex",
            "gap": "8px",
            "marginTop": "8px",
            "alignItems": "center",
        },
    )
    filter_body = Div([grid_with_legend, actions])

    return Div(
        [
            Details(
                [
                    Summary("Exclude elements", style=_SUMMARY_STYLE),
                    Div(filter_body, style={"padding": "8px 12px"}),
                ],
                id="element-filter-details",
                open=False,
                style={"marginBottom": "8px", "fontSize": "13px"},
            ),
            Store(id="element-filter", data=None),
            Store(id="element-filter-pending", data=[]),
        ]
    )


def register_element_filter_callbacks() -> None:
    """Register callbacks for the periodic table element filter."""
    all_symbols = list(PERIODIC_TABLE_SYMBOLS)

    @callback(
        Output("element-filter-pending", "data"),
        Input({"type": "element-btn", "index": ALL}, "n_clicks"),
        State("element-filter-pending", "data"),
        prevent_initial_call=True,
    )
    def toggle_element(_n_clicks, pending):
        """
        Toggle the clicked element in the pending exclusion selection.

        Parameters
        ----------
        _n_clicks
            Click counts for all periodic-table element buttons.
        pending
            Currently pending element exclusion selection.

        Returns
        -------
        list
            Updated pending element exclusion selection.
        """
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict):
            raise PreventUpdate
        symbol = trigger["index"]
        pending = list(pending or [])
        if symbol in pending:
            pending.remove(symbol)
        else:
            pending.append(symbol)
        return pending

    @callback(
        Output({"type": "element-btn", "index": ALL}, "style"),
        Input("element-filter-pending", "data"),
        prevent_initial_call=True,
    )
    def sync_button_styles(pending):
        """
        Synchronise element button styles with the pending selection.

        Parameters
        ----------
        pending
            Currently pending element exclusion selection.

        Returns
        -------
        list
            Style dictionaries for all periodic-table element buttons.
        """
        excluded = set(pending or [])
        return [_btn_style(sym, sym in excluded) for sym in all_symbols]

    @callback(
        Output("element-filter", "data"),
        Output("element-filter-pending", "data", allow_duplicate=True),
        Input("element-filter-apply", "n_clicks"),
        Input("element-filter-exclude-all", "n_clicks"),
        Input("element-filter-clear", "n_clicks"),
        State("element-filter-pending", "data"),
        prevent_initial_call=True,
    )
    def apply_or_clear(_apply, _exclude_all, _clear, pending):
        """
        Apply, clear, or bulk-update the element exclusion selection.

        Parameters
        ----------
        _apply
            Click count for the Apply button.
        _exclude_all
            Click count for the Exclude all button.
        _clear
            Click count for the Clear button.
        pending
            Currently pending element exclusion selection.

        Returns
        -------
        tuple
            Committed element filter value and updated pending selection.
        """
        trigger = ctx.triggered_id
        if trigger == "element-filter-apply":
            return pending or [], no_update
        if trigger == "element-filter-exclude-all":
            return no_update, all_symbols
        if trigger == "element-filter-clear":
            return [], []
        raise PreventUpdate
