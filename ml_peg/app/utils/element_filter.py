"""Shared helpers for periodic-table element filtering."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from dash import ALL, Input, Output, State, callback, ctx, html
from dash.exceptions import PreventUpdate

from ml_peg.analysis.utils.decorators import (
    PERIODIC_TABLE_COLS,
    PERIODIC_TABLE_POSITIONS,
    PERIODIC_TABLE_ROWS,
)
from ml_peg.analysis.utils.utils import mae

ELEMENT_FILTER_STORE_ID = "element-filter-store"
ELEMENT_BUTTON_TYPE = "element-filter-button"
ELEMENT_SELECT_ALL_ID = "element-filter-select-all"
ELEMENT_CLEAR_ID = "element-filter-clear"
ALL_ELEMENTS = sorted(PERIODIC_TABLE_POSITIONS.keys())
FILTER_CALLBACKS_REGISTERED = False


def build_element_filter_panel(
    title: str = "Element Filter",
    description: str | None = None,
    available_elements: set[str] | None = None,
) -> html.Div:
    """
    Build a periodic-table control panel for element filtering.

    Parameters
    ----------
    title
        Panel title displayed above the grid.
    description
        Optional helper text describing the control behaviour.
    available_elements
        Optional subset of elements that may be filtered. Elements not present are
        rendered disabled.

    Returns
    -------
    html.Div
        Component tree containing the filter panel.
    """
    available = available_elements or set(ALL_ELEMENTS)
    buttons: list[html.Button] = []
    for element, (row, col) in PERIODIC_TABLE_POSITIONS.items():
        is_enabled = element in available
        style = {
            "gridRow": row + 1,
            "gridColumn": col + 1,
            "width": "100%",
            "height": "100%",
            "border": "1px solid #ced4da",
            "borderRadius": "4px",
            "fontSize": "11px",
            "padding": "0",
            "cursor": "pointer" if is_enabled else "not-allowed",
            "opacity": 1.0 if is_enabled else 0.3,
            "backgroundColor": "#0d6efd" if is_enabled else "#f8f9fa",
            "color": "#fff" if is_enabled else "#495057",
        }
        buttons.append(
            html.Button(
                element,
                id={"type": ELEMENT_BUTTON_TYPE, "element": element},
                n_clicks=0,
                disabled=not is_enabled,
                style=style,
            )
        )

    grid = html.Div(
        buttons,
        style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({PERIODIC_TABLE_COLS}, 32px)",
            "gridTemplateRows": f"repeat({PERIODIC_TABLE_ROWS}, 32px)",
            "gap": "4px",
            "justifyContent": "flex-start",
            "alignItems": "flex-start",
            "maxWidth": "fit-content",
        },
    )

    description_text = (
        description
        or "Toggle individual elements to include/exclude benchmarks containing "
        "those species. Select all to reset the default."
    )

    return html.Div(
        [
            html.H4(title),
            html.P(description_text),
            grid,
            html.Div(
                [
                    html.Button(
                        "Select all",
                        id=ELEMENT_SELECT_ALL_ID,
                        className="btn btn-outline-primary btn-sm",
                        style={"marginRight": "8px"},
                    ),
                    html.Button(
                        "Clear selection",
                        id=ELEMENT_CLEAR_ID,
                        className="btn btn-outline-secondary btn-sm",
                    ),
                ],
                style={"marginTop": "12px"},
            ),
        ],
        style={
            "border": "1px solid #dee2e6",
            "borderRadius": "6px",
            "padding": "12px",
            "backgroundColor": "#f8f9fa",
            "marginBottom": "16px",
        },
    )


def register_element_filter_callbacks(
    default_selection: list[str] | None = None,
) -> None:
    """
    Register callbacks to sync periodic-table buttons with a shared store.

    Parameters
    ----------
    default_selection
        Elements selected when "Select all" is clicked. Defaults to all elements.
    """
    global FILTER_CALLBACKS_REGISTERED
    if FILTER_CALLBACKS_REGISTERED:
        return
    FILTER_CALLBACKS_REGISTERED = True

    default_selection = default_selection or ALL_ELEMENTS

    @callback(
        Output(ELEMENT_FILTER_STORE_ID, "data", allow_duplicate=True),
        Input({"type": ELEMENT_BUTTON_TYPE, "element": ALL}, "n_clicks"),
        Input(ELEMENT_SELECT_ALL_ID, "n_clicks"),
        Input(ELEMENT_CLEAR_ID, "n_clicks"),
        State({"type": ELEMENT_BUTTON_TYPE, "element": ALL}, "id"),
        State(ELEMENT_FILTER_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def update_selection(
        _button_clicks,
        _select_all_clicks,
        _clear_clicks,
        button_ids,
        current_selection,
    ) -> list[str]:
        """
        Update the element selection whenever a grid button or control is used.

        Parameters
        ----------
        _button_clicks
            Click counts for every periodic-table cell (unused, required by Dash).
        _select_all_clicks
            Click count for the select-all button.
        _clear_clicks
            Click count for the clear-selection button.
        button_ids
            List of element identifiers corresponding to the button Inputs.
        current_selection
            Previously stored list of selected elements.

        Returns
        -------
        list[str]
            Sorted list of currently selected elements.
        """
        trigger = ctx.triggered_id
        current = set(current_selection or default_selection)
        if trigger == ELEMENT_SELECT_ALL_ID:
            return default_selection
        if trigger == ELEMENT_CLEAR_ID:
            return []
        if isinstance(trigger, dict):
            element = trigger.get("element")
            if element is None:
                raise PreventUpdate
            if element in current:
                current.remove(element)
            else:
                current.add(element)
            return sorted(current)
        raise PreventUpdate

    @callback(
        Output({"type": ELEMENT_BUTTON_TYPE, "element": ALL}, "style"),
        Input(ELEMENT_FILTER_STORE_ID, "data"),
        State({"type": ELEMENT_BUTTON_TYPE, "element": ALL}, "id"),
        prevent_initial_call=False,
    )
    def sync_button_styles(selected_elements, button_ids):
        """
        Synchronise button styles with the current selection.

        Parameters
        ----------
        selected_elements
            Elements currently stored in the shared selection Store.
        button_ids
            ID payload for each periodic-table button.

        Returns
        -------
        list[dict[str, Any]]
            Inline style dictionaries applied to each button.
        """
        selection = set(selected_elements or default_selection)
        styles: list[dict[str, Any]] = []
        for button_id in button_ids:
            element = button_id.get("element")
            row, col = PERIODIC_TABLE_POSITIONS[element]
            is_selected = element in selection
            styles.append(
                {
                    "gridRow": row + 1,
                    "gridColumn": col + 1,
                    "width": "100%",
                    "height": "100%",
                    "border": "1px solid #ced4da",
                    "borderRadius": "4px",
                    "fontSize": "11px",
                    "padding": "0",
                    "cursor": "pointer",
                    "opacity": 1.0,
                    "backgroundColor": "#0d6efd" if is_selected else "#f8f9fa",
                    "color": "#fff" if is_selected else "#495057",
                }
            )
        return styles


def load_filter_elements(payload_path: Path) -> list[str]:
    """
    Load the set of elements present in an analysis payload.

    Parameters
    ----------
    payload_path
        Path to JSON payload produced during analysis.

    Returns
    -------
    list[str]
        Sorted list of elements present in the payload (fallbacks to all elements).
    """
    try:
        with payload_path.open() as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return ALL_ELEMENTS

    systems = payload.get("systems", [])
    elements = {symbol for entry in systems for symbol in entry.get("elements", [])}
    return sorted(elements or ALL_ELEMENTS)


def build_x23_filter_handler(
    payload_path: Path,
    alias_map: dict[str, str] | None = None,
) -> Callable[[list[dict], list[str] | None], list[dict]]:
    """
    Build a handler that filters X23 metrics based on element selection.

    Parameters
    ----------
    payload_path
        Path to ``x23_filter_payload.json`` generated during analysis.
    alias_map
        Optional mapping from display names to canonical model identifiers so the
        payload entries can be aligned with the current table rows.

    Returns
    -------
    Callable
        Function that takes base rows and selected elements, returning filtered rows.
    """

    def _load_payload() -> dict[str, Any]:
        """
        Load the cached X23 payload from disk if present.

        Returns
        -------
        dict[str, Any]
            Payload contents including systems and structure model (or empty fallback).
        """
        try:
            with payload_path.open() as handle:
                return json.load(handle)
        except FileNotFoundError:
            return {"systems": []}

    payload_cache: dict[str, Any] | None = None

    alias_map = alias_map or {}

    def handler(rows: list[dict], selected: list[str] | None) -> list[dict]:
        """
        Filter a copy of the table rows based on the selected elements.

        Parameters
        ----------
        rows
            Benchmark rows to mutate with recalculated MAEs.
        selected
            Elements that remain active in the global filter.

        Returns
        -------
        list[dict]
            Updated table rows with filtered MAE values.
        """
        nonlocal payload_cache
        if payload_cache is None:
            payload_cache = _load_payload()

        systems = payload_cache.get("systems", [])
        if not systems:
            return rows

        allowed = set(selected or ALL_ELEMENTS)
        if not allowed:
            for row in rows:
                for key in row:
                    if key not in {"MLIP", "id"}:
                        row[key] = None
            return rows

        filtered = [
            entry for entry in systems if set(entry.get("elements", [])) <= allowed
        ]
        refs = [entry["ref"] for entry in filtered]

        for row in rows:
            canonical = row.get("id") or row.get("MLIP")
            if isinstance(canonical, str):
                canonical = alias_map.get(canonical, canonical)
            preds = [entry.get("models", {}).get(canonical) for entry in filtered]
            pairs = [
                (ref, pred)
                for ref, pred in zip(refs, preds, strict=True)
                if pred is not None
            ]
            if pairs:
                ref_vals, pred_vals = zip(*pairs, strict=True)
                row["MAE"] = mae(list(ref_vals), list(pred_vals))
            else:
                row["MAE"] = None
        return rows

    return handler
