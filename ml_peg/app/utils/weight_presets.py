"""Summary-table category-weight presets: sidebar control and its callbacks."""

from __future__ import annotations

import random

from dash import ALL, Input, Output, State, callback, ctx
from dash.dash_table import DataTable
from dash.dcc import Store
from dash.exceptions import PreventUpdate
from dash.html import Button, Details, Div, Summary
from yaml import safe_load

from ml_peg.app import APP_ROOT

# Each preset lists the category titles that stay weighted (1.0); every other
# category is set to 0.0. Titles must match the category `title:` values exactly.
# The "Random" preset is generated in code.
WEIGHT_PRESETS = safe_load((APP_ROOT / "utils" / "weight_presets.yml").read_text())

_RESERVED_COLUMNS = {"MLIP", "Score", "id", "link"}


def _preset_btn_style(selected: bool) -> dict[str, str]:
    """
    Return the style for a weight-preset toggle button.

    Parameters
    ----------
    selected
        Whether the preset is currently selected.

    Returns
    -------
    dict[str, str]
        Style dictionary for the button.
    """
    return {
        "padding": "4px 10px",
        "fontSize": "12px",
        "borderRadius": "4px",
        "cursor": "pointer",
        "border": "1px solid #228be6" if selected else "1px solid #ced4da",
        "backgroundColor": "#228be6" if selected else "#fff",
        "color": "#fff" if selected else "#495057",
    }


def _weight_columns(summary_table: DataTable) -> list[str]:
    """
    Return the configurable (category) weight columns of the summary table.

    Parameters
    ----------
    summary_table
        Overall summary table.

    Returns
    -------
    list[str]
        Column ids weightable by a preset (i.e. not reserved columns).
    """
    return [
        column["id"]
        for column in summary_table.columns
        if column["id"] not in _RESERVED_COLUMNS
    ]


def build_weight_preset_selector(label_style: dict[str, str]) -> Details:
    """
    Build the weight-preset sidebar control (toggle buttons + Apply/Reset).

    Parameters
    ----------
    label_style
        Style shared by the sidebar section summary labels.

    Returns
    -------
    Details
        Collapsible control containing the preset buttons and pending store.
    """
    preset_names = [*WEIGHT_PRESETS, "Random"]
    return Details(
        [
            Summary("Weight presets", style=label_style),
            Div(
                [
                    Div(
                        [
                            Button(
                                name,
                                id={"type": "weight-preset-btn", "index": name},
                                n_clicks=0,
                                style=_preset_btn_style(False),
                            )
                            for name in preset_names
                        ],
                        style={"display": "flex", "flexWrap": "wrap", "gap": "6px"},
                    ),
                    Div(
                        [
                            Button(
                                "Apply",
                                id="weight-preset-apply",
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
                                "Reset",
                                id="weight-preset-reset",
                                n_clicks=0,
                                style={
                                    "padding": "4px 12px",
                                    "fontSize": "12px",
                                    "backgroundColor": "#fff",
                                    "color": "#495057",
                                    "border": "1px solid #ced4da",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={"display": "flex", "gap": "6px", "marginTop": "8px"},
                    ),
                    Store(id="weight-preset-pending", storage_type="memory", data=[]),
                ],
                style={"padding": "8px 12px"},
            ),
        ],
        style={"marginBottom": "8px", "fontSize": "13px"},
    )


def register_weight_preset_callbacks(
    summary_table: DataTable, default_weights: dict[str, float]
) -> None:
    """
    Register the weight-preset toggle/apply/reset callbacks.

    Applying writes the union of the selected presets to the summary weight store,
    which the normal score-propagation chain picks up. This does not touch table
    data directly, so model-filter and colormap handling are inherited.

    Parameters
    ----------
    summary_table
        Overall summary table, used to enumerate the weightable category columns.
    default_weights
        Default summary weights restored by the reset button.
    """

    @callback(
        Output("weight-preset-pending", "data", allow_duplicate=True),
        Input({"type": "weight-preset-btn", "index": ALL}, "n_clicks"),
        State("weight-preset-pending", "data"),
        prevent_initial_call=True,
    )
    def toggle_weight_preset(_n_clicks: list[int], pending: list[str]) -> list[str]:
        """
        Toggle the clicked preset in the pending selection.

        Parameters
        ----------
        _n_clicks
            Click counts for all preset buttons.
        pending
            Currently pending preset selection.

        Returns
        -------
        list[str]
            Updated pending preset selection.
        """
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict):
            raise PreventUpdate
        name = trigger["index"]
        pending = list(pending or [])
        if name in pending:
            pending.remove(name)
        else:
            pending.append(name)
        return pending

    @callback(
        Output({"type": "weight-preset-btn", "index": ALL}, "style"),
        Input("weight-preset-pending", "data"),
        State({"type": "weight-preset-btn", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def sync_weight_preset_styles(
        pending: list[str], ids: list[dict]
    ) -> list[dict[str, str]]:
        """
        Highlight preset buttons that are currently selected.

        Parameters
        ----------
        pending
            Currently pending preset selection.
        ids
            Ids of the preset buttons, in layout order.

        Returns
        -------
        list[dict[str, str]]
            Style dictionaries for each preset button.
        """
        selected = set(pending or [])
        return [_preset_btn_style(button["index"] in selected) for button in ids]

    @callback(
        Output("summary-table-weight-store", "data", allow_duplicate=True),
        Input("weight-preset-apply", "n_clicks"),
        State("weight-preset-pending", "data"),
        prevent_initial_call=True,
    )
    def apply_weight_presets(n_clicks: int, pending: list[str]) -> dict[str, float]:
        """
        Apply the union of the selected presets to the summary weights.

        A category is weighted 1.0 if it appears in any selected preset, else 0.0
        (the union of the selected presets). "Random" instead gives every category
        an independent random weight in [0, 1].

        Parameters
        ----------
        n_clicks
            Apply-button click count.
        pending
            Currently selected presets.

        Returns
        -------
        dict[str, float]
            Category-column weights written to the summary weight store.
        """
        if not n_clicks or not pending:
            raise PreventUpdate

        weight_columns = _weight_columns(summary_table)
        weights = dict.fromkeys(weight_columns, 0.0)
        for name in pending:
            if name == "Random":
                contribution = {
                    column: round(random.random(), 2) for column in weight_columns
                }
            else:
                relevant = {f"{title} Score" for title in WEIGHT_PRESETS[name]}
                contribution = {
                    column: (1.0 if column in relevant else 0.0)
                    for column in weight_columns
                }
            weights = {
                column: max(weights[column], contribution[column])
                for column in weight_columns
            }
        return weights

    @callback(
        Output("summary-table-weight-store", "data", allow_duplicate=True),
        Output("weight-preset-pending", "data", allow_duplicate=True),
        Input("weight-preset-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_weight_presets(n_clicks: int) -> tuple[dict[str, float], list[str]]:
        """
        Restore the category yml default weights and clear the selection.

        Parameters
        ----------
        n_clicks
            Reset-button click count.

        Returns
        -------
        tuple[dict[str, float], list[str]]
            Default category weights and an empty pending selection.
        """
        if not n_clicks:
            raise PreventUpdate
        return default_weights, []
