"""Build data components."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

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
from ml_peg.app import APP_ROOT


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

_PRESET_BUTTON_STYLE = {
    "padding": "4px 8px",
    "fontSize": "12px",
    "backgroundColor": "#f8f9fa",
    "color": "#343a40",
    "border": "1px solid #ced4da",
    "borderRadius": "4px",
    "cursor": "pointer",
}

_ELEMENT_COVERAGE_PATH = APP_ROOT / "data" / "element_coverage.json"


@lru_cache(maxsize=1)
def _load_element_coverage(path: Path = _ELEMENT_COVERAGE_PATH) -> dict:
    """
    Load element coverage JSON, returning an empty dict if unavailable.

    Parameters
    ----------
    path
        Path to the element coverage JSON file.

    Returns
    -------
    dict
        Parsed coverage payload, or an empty dictionary when unavailable.
    """
    try:
        with open(path, encoding="utf8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _dataset_presets() -> tuple[dict, ...]:
    """
    Build preset entries from element_coverage.json dataset keys.

    Returns
    -------
    tuple[dict, ...]
        One preset dict per dataset entry in the coverage file.
    """
    datasets = _load_element_coverage().get("datasets", {})
    return tuple(
        {
            "id": name.lower().replace("/", "-").replace(" ", "-"),
            "label": name,
            "include": tuple(
                s for s in PERIODIC_TABLE_SYMBOLS if s in set(data["supported"])
            ),
            "title": f"Keep elements covered by {name} training data",
        }
        for name, data in datasets.items()
        if data.get("supported")
    )


_ELEMENT_FILTER_PRESET_GROUPS = (
    (
        "Chemistry",
        (
            {
                "id": "chon",
                "label": "CHON",
                "include": ("C", "H", "N", "O"),
                "title": "Keep C, H, N, O",
            },
        ),
    ),
    (
        "Dataset coverage",
        _dataset_presets(),
    ),
)
_ELEMENT_FILTER_PRESETS = {
    preset["id"]: preset
    for _, presets in _ELEMENT_FILTER_PRESET_GROUPS
    for preset in presets
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


def _preset_excluded_symbols(preset_id: str) -> list[str]:
    """
    Return symbols excluded by an element-filter preset.

    Parameters
    ----------
    preset_id
        Identifier of the preset to apply.

    Returns
    -------
    list[str]
        Periodic-table symbols outside the preset's included element set.
    """
    preset = _ELEMENT_FILTER_PRESETS[preset_id]
    included = set(preset["include"])
    return [symbol for symbol in PERIODIC_TABLE_SYMBOLS if symbol not in included]


def _build_preset_section() -> Div:
    """
    Build preset buttons for common element-filter selections.

    Returns
    -------
    Div
        Preset button section.
    """
    groups = []
    for group_label, presets in _ELEMENT_FILTER_PRESET_GROUPS:
        groups.append(
            Div(
                [
                    Span(
                        group_label,
                        style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": "#6c757d",
                            "minWidth": "92px",
                        },
                    ),
                    Div(
                        [
                            Button(
                                preset["label"],
                                id={
                                    "type": "element-filter-preset",
                                    "index": preset["id"],
                                },
                                n_clicks=0,
                                title=preset["title"],
                                style=_PRESET_BUTTON_STYLE,
                            )
                            for preset in presets
                        ],
                        style={
                            "display": "flex",
                            "gap": "6px",
                            "flexWrap": "wrap",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "flexWrap": "wrap",
                },
            )
        )

    return Div(
        [
            Span(
                "Presets",
                style={
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.07em",
                    "color": "#6c757d",
                },
            ),
            *groups,
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            "marginTop": "8px",
            "paddingTop": "10px",
            "borderTop": "1px solid #dee2e6",
        },
    )


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
    filter_body = Div(
        [
            Div(
                [grid_with_legend, _build_preset_section()],
                style={"width": "fit-content"},
            ),
            actions,
        ]
    )

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
        Input({"type": "element-filter-preset", "index": ALL}, "n_clicks"),
        State("element-filter-pending", "data"),
        prevent_initial_call=True,
    )
    def apply_or_clear(_apply, _exclude_all, _clear, _presets, pending):
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
        _presets
            Click counts for preset element-set buttons.
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
        if isinstance(trigger, dict) and trigger.get("type") == "element-filter-preset":
            return no_update, _preset_excluded_symbols(trigger["index"])
        raise PreventUpdate
