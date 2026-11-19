"""Summary-table category-weight presets: sidebar control and its callbacks."""

from __future__ import annotations

import json

from dash import ALL, Input, Output, State, callback, clientside_callback
from dash.dash_table import DataTable
from dash.dcc import Store
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
        "border": (
            "1px solid var(--mlpeg-accent)"
            if selected
            else "1px solid var(--mlpeg-border-strong)"
        ),
        "backgroundColor": (
            "var(--mlpeg-accent)" if selected else "var(--mlpeg-surface)"
        ),
        "color": "#ffffff" if selected else "var(--mlpeg-ink-2)",
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
                                    "backgroundColor": "var(--mlpeg-accent)",
                                    "color": "#ffffff",
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
                                    "backgroundColor": "var(--mlpeg-surface)",
                                    "color": "var(--mlpeg-ink-2)",
                                    "border": "1px solid var(--mlpeg-border-strong)",
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
    clientside_callback(
        """function(clicks, pending) {
            const trigger = window.dash_clientside.callback_context.triggered[0];
            if (!trigger) return window.dash_clientside.no_update;
            const id = JSON.parse(trigger.prop_id.slice(
                0, trigger.prop_id.lastIndexOf('.')));
            const selected = [...(pending || [])];
            return selected.includes(id.index)
                ? selected.filter(name => name !== id.index) : [...selected, id.index];
        }""",
        Output("weight-preset-pending", "data", allow_duplicate=True),
        Input({"type": "weight-preset-btn", "index": ALL}, "n_clicks"),
        State("weight-preset-pending", "data"),
        prevent_initial_call=True,
    )

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

    config = {
        "columns": _weight_columns(summary_table),
        "presets": WEIGHT_PRESETS,
        "defaults": default_weights,
    }
    clientside_callback(
        f"""function(applyClicks, resetClicks, pending) {{
            return window.dash_clientside.settingsEditors.presets(
                {json.dumps(config)}, applyClicks, resetClicks, pending);
        }}""",
        Output("summary-table-weight-store", "data", allow_duplicate=True),
        Output("weight-preset-pending", "data", allow_duplicate=True),
        Input("weight-preset-apply", "n_clicks"),
        Input("weight-preset-reset", "n_clicks"),
        State("weight-preset-pending", "data"),
        prevent_initial_call=True,
    )
