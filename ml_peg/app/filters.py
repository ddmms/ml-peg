"""Build data components."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import TypedDict

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
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names, load_model_configs


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


class ElementPreset(TypedDict):
    """A named element-filter preset and the elements it keeps."""

    id: str
    label: str
    include: tuple[str, ...]
    title: str


def _dataset_presets() -> tuple[ElementPreset, ...]:
    """
    Build preset entries from element_coverage.json dataset keys.

    Datasets that share an identical supported-element set are merged into a
    single preset (label = their names joined with "/"), so families with the
    same coverage (e.g. MPtrj/Alexandria/OMAT) render as one button and future
    identical-coverage datasets fold in automatically.

    Returns
    -------
    tuple[dict, ...]
        One preset dict per group of identical-coverage datasets.
    """
    datasets = _load_element_coverage().get("datasets", {})
    groups: dict[frozenset[str], list[str]] = {}
    order: list[frozenset[str]] = []
    for name, data in datasets.items():
        supported = data.get("supported")
        if not supported:
            continue
        key = frozenset(supported)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(name)

    presets = []
    for key in order:
        label = "/".join(groups[key])
        presets.append(
            {
                "id": label.lower().replace("/", "-").replace(" ", "-"),
                "label": label,
                "include": tuple(s for s in PERIODIC_TABLE_SYMBOLS if s in key),
                "title": f"Keep elements covered by {label} training data",
            }
        )
    return tuple(presets)


@lru_cache(maxsize=1)
def _model_supported_elements() -> dict[str, frozenset[str]]:
    """
    Map each model to the elements covered by its training datasets.

    A model's elements are the union of the ``supported`` sets of every dataset
    listed under its ``datasets`` key in ``models.yml``. Models with no
    ``datasets`` tag, or whose datasets are absent from the coverage file, are
    omitted (treated as untagged).

    Returns
    -------
    dict[str, frozenset[str]]
        Supported elements keyed by model name, for tagged models only.
    """
    coverage = _load_element_coverage().get("datasets", {})
    models = get_model_names(current_models)
    configs, _ = load_model_configs(tuple(models))

    result: dict[str, frozenset[str]] = {}
    for model in models:
        elements: set[str] = set()
        for dataset in configs.get(model, {}).get("datasets") or []:
            supported = coverage.get(dataset, {}).get("supported")
            if supported:
                elements.update(supported)
        if elements:
            result[model] = frozenset(elements)
    return result


def _selected_models_excluded_symbols(
    selected_models: list[str] | None,
    mode: str,
) -> list[str] | None:
    """
    Return symbols to exclude to keep the union/intersection of model coverage.

    Parameters
    ----------
    selected_models
        Currently selected model names.
    mode
        Either ``"union"`` (elements supported by any selected model) or
        ``"intersection"`` (elements supported by every selected model).

    Returns
    -------
    list[str] | None
        Symbols outside the combined coverage, or ``None`` if no selected model
        is tagged with usable dataset coverage.
    """
    model_elements = _model_supported_elements()
    sets = [
        model_elements[model]
        for model in (selected_models or [])
        if model in model_elements
    ]
    if not sets:
        return None

    keep = (
        set(sets[0]).union(*sets)
        if mode == "union"
        else set(sets[0]).intersection(*sets)
    )
    return [symbol for symbol in PERIODIC_TABLE_SYMBOLS if symbol not in keep]


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

    # Row of buttons deriving the keep-set from the currently selected models'
    # dataset coverage, rather than a fixed preset.
    selected_models_row = Div(
        [
            Span(
                "Selected models",
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
                        "Union",
                        id="element-filter-union",
                        n_clicks=0,
                        title="Keep elements covered by any selected model",
                        style=_PRESET_BUTTON_STYLE,
                    ),
                    Button(
                        "Intersection",
                        id="element-filter-intersection",
                        n_clicks=0,
                        title="Keep elements covered by every selected model",
                        style=_PRESET_BUTTON_STYLE,
                    ),
                ],
                style={"display": "flex", "gap": "6px", "flexWrap": "wrap"},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "flexWrap": "wrap",
        },
    )

    return Div(
        [
            Div(
                [
                    Span(
                        "Presets",
                        style={
                            "fontWeight": "600",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.07em",
                        },
                    ),
                    Span(
                        " — keep only the chosen set",
                        style={"fontStyle": "italic"},
                    ),
                ],
                style={"fontSize": "11px", "color": "#6c757d"},
            ),
            *groups,
            selected_models_row,
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
    is applied only when the "Apply" button is clicked. "Exclude all" and
    "Clear" update the pending selection. The committed selection is held in
    ``dcc.Store(id="element-filter")``.

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
    grid_caption = Span(
        "Click on elements to exclude them, then Apply to update tables.",
        style={
            "display": "block",
            "fontSize": "12px",
            "color": "#6c757d",
            "marginBottom": "6px",
        },
    )

    filter_body = Div(
        [
            Div(
                [grid_caption, grid_with_legend, _build_preset_section()],
                style={"width": "fit-content"},
            ),
            actions,
        ]
    )

    return Div(
        [
            Details(
                [
                    Summary("Element filtering", style=_SUMMARY_STYLE),
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
        Input("element-filter-union", "n_clicks"),
        Input("element-filter-intersection", "n_clicks"),
        State("element-filter", "data"),
        State("element-filter-pending", "data"),
        State("selected-models-store", "data"),
        prevent_initial_call=True,
    )
    def apply_or_clear(
        _apply,
        _exclude_all,
        _clear,
        _presets,
        _union,
        _intersection,
        current,
        pending,
        selected_models,
    ):
        """
        Handle "Apply", clear, or bulk-update element exclusion actions.

        Parameters
        ----------
        _apply
            Click count for the "Apply" button.
        _exclude_all
            Click count for the Exclude all button.
        _clear
            Click count for the Clear button.
        _presets
            Click counts for preset element-set buttons.
        _union
            Click count for the selected-models Union button.
        _intersection
            Click count for the selected-models Intersection button.
        current
            Currently committed element exclusion selection.
        pending
            Currently pending element exclusion selection.
        selected_models
            Models currently selected in the global model filter.

        Returns
        -------
        tuple
            Committed element filter value and updated pending selection.
        """
        trigger = ctx.triggered_id
        current = sorted(current or [])
        pending = sorted(pending or [])
        if trigger == "element-filter-apply":
            if pending == current:
                raise PreventUpdate
            return pending, no_update
        if trigger == "element-filter-exclude-all":
            if sorted(all_symbols) == pending:
                raise PreventUpdate
            return no_update, all_symbols
        if trigger == "element-filter-clear":
            if not pending:
                raise PreventUpdate
            return no_update, []
        if isinstance(trigger, dict) and trigger.get("type") == "element-filter-preset":
            preset_excluded = _preset_excluded_symbols(trigger["index"])
            if sorted(preset_excluded) == pending:
                raise PreventUpdate
            return no_update, preset_excluded
        if trigger in ("element-filter-union", "element-filter-intersection"):
            mode = "union" if trigger == "element-filter-union" else "intersection"
            excluded = _selected_models_excluded_symbols(selected_models, mode)
            if excluded is None or sorted(excluded) == pending:
                raise PreventUpdate
            return no_update, excluded
        raise PreventUpdate
