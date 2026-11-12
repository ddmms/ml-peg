"""Controls for preparing live-site bug reports in the browser."""

from __future__ import annotations

from dash import ClientsideFunction, Input, Output, State, clientside_callback
from dash.dcc import Store
from dash.html import Button, Div

from ml_peg import __version__


def _report_stores(stores: list[Store]) -> list[Store]:
    """
    Select the stores whose values are worth including in a report.

    Parameters
    ----------
    stores
        Globally mounted app stores.

    Returns
    -------
    list[Store]
        Weight, threshold and appearance stores. Computed results and unrelated
        storage are omitted.
    """
    preferences = {"theme-store", "zoom-store", "font-store", "cmap-store"}
    return [
        store
        for store in stores
        if isinstance(store.id, str)
        and (
            store.id.endswith(("-weight-store", "-thresholds-store"))
            or store.id in preferences
        )
    ]


def build_bug_report_button(stores: list[Store]) -> Div:
    """
    Build the global bug-report button, its config store and callback output.

    The defaults travel to the browser as store *data* rather than as part of the
    clientside callback's source. Dash names an inline clientside callback by
    hashing its source, so baking the defaults in renamed the function whenever
    benchmark data changed, breaking any page still holding the previous name.

    Parameters
    ----------
    stores
        Globally mounted app stores, used to record the defaults a report is
        compared against.

    Returns
    -------
    Div
        Header control for opening the browser's report dialog.
    """
    report_stores = _report_stores(stores)
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
            Store(
                id="bug-report-config",
                data={
                    "version": __version__,
                    "store_ids": [store.id for store in report_stores],
                    "defaults": {
                        store.id: getattr(store, "data", None)
                        for store in report_stores
                    },
                },
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
    clientside_callback(
        ClientsideFunction(namespace="bug_report", function_name="open"),
        Output("bug-report-dummy", "children"),
        Input("bug-report-button", "n_clicks"),
        State("bug-report-config", "data"),
        State("selected-models-store", "data"),
        State("element-filter", "data"),
        *[State(store.id, "data") for store in _report_stores(stores)],
        prevent_initial_call=True,
    )
