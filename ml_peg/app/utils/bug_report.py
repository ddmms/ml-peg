"""Controls for preparing live-site bug reports in the browser."""

from __future__ import annotations

import json

from dash import Input, Output, State, clientside_callback
from dash.dcc import Store
from dash.html import Button, Div

from ml_peg import __version__


def build_bug_report_button() -> Div:
    """
    Build the global bug-report button and callback output.

    Returns
    -------
    Div
        Header control for opening the browser's report dialog.
    """
    return Div(
        [
            Button(
                "Report a bug",
                id="bug-report-button",
                className="mlpeg-bug-report-button",
                n_clicks=0,
                title="Report a bug",
                **{"aria-haspopup": "dialog", "aria-label": "Report a bug"},
            ),
            Div(id="bug-report-dummy", style={"display": "none"}),
        ]
    )


def register_bug_report_callbacks(stores: list[Store]) -> None:
    """
    Capture report context only when the report button is clicked.

    Parameters
    ----------
    stores
        Globally mounted app stores. Only weights, thresholds and appearance
        preferences are included; computed results and unrelated storage are omitted.
    """
    preferences = {"theme-store", "zoom-store", "font-store", "cmap-store"}
    report_stores = [
        store
        for store in stores
        if isinstance(store.id, str)
        and (
            store.id.endswith(("-weight-store", "-thresholds-store"))
            or store.id in preferences
        )
    ]
    config = {
        "version": __version__,
        "store_ids": [store.id for store in report_stores],
        "defaults": {store.id: getattr(store, "data", None) for store in report_stores},
    }
    clientside_callback(
        f"""
        function(n_clicks, models, excluded, ...values) {{
            if (n_clicks) {{
                window.mlpegBugReport.open(
                    {json.dumps(config)}, models, excluded, values
                );
            }}
            return "";
        }}
        """,
        Output("bug-report-dummy", "children"),
        Input("bug-report-button", "n_clicks"),
        State("selected-models-store", "data"),
        State("element-filter", "data"),
        *[State(store.id, "data") for store in report_stores],
        prevent_initial_call=True,
    )
