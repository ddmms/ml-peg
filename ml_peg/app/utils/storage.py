"""
Buttons and callbacks for clearing the app data saved in the browser.

Kept in its own file so ``build_app`` stays short. ``build_header_controls``
makes the buttons shown in the top-right corner; ``register_storage_callbacks``
makes them work: clearing the saved data when the button is clicked, and
automatically after a new version is released.
"""

from __future__ import annotations

from dash import Input, Output, clientside_callback
from dash.html import Button, Div

from ml_peg import __version__
from ml_peg.app.utils.onboarding import build_tutorial_button

_CLEAR_BUTTON_STYLE = {
    "padding": "8px 16px",
    "borderRadius": "6px",
    "border": "1px solid #cbd5e1",
    "background": "white",
    "color": "#475569",
    "cursor": "pointer",
    "fontWeight": 600,
    "fontSize": "14px",
    "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.1)",
    "transition": "all 0.2s ease",
}


def build_header_controls() -> Div:
    """
    Build the buttons shown in the top-right corner of the app.

    Holds the "Clear cache" button next to the "Tutorial" button. The two hidden
    Divs are not shown, they just give the callbacks somewhere to write to.

    Returns
    -------
    Div
        Container holding the top-right buttons.
    """
    return Div(
        [
            Button(
                "Clear cache",
                id="clear-storage-button",
                n_clicks=0,
                title=(
                    "Clear browser-stored app state (weights, thresholds, "
                    "tutorial progress) and reload. Use after an update if "
                    "the app shows stale data."
                ),
                style=_CLEAR_BUTTON_STYLE,
            ),
            build_tutorial_button(),
            Div(id="clear-storage-dummy", style={"display": "none"}),
            Div(id="storage-version-dummy", style={"display": "none"}),
        ],
        style={
            "position": "fixed",
            "top": "20px",
            "right": "20px",
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "zIndex": "1600",  # Above loading overlays (1200/1400).
        },
    )


def register_storage_callbacks() -> None:
    """Register the clear-cache and version-bump auto-clear clientside callbacks."""
    # Clear all browser-persisted dcc.Store data (session + local) and reload, so
    # stale cached state after an update can be wiped from the header button.
    clientside_callback(
        """
        function (n_clicks) {
            if (n_clicks && window.confirm(
                "Clear cached app data and reload? Saved weights and thresholds"
                + " will be reset."
            )) {
                window.localStorage.clear();
                window.sessionStorage.clear();
                window.location.reload();
            }
            return "";
        }
        """,
        Output("clear-storage-dummy", "children"),
        Input("clear-storage-button", "n_clicks"),
        prevent_initial_call=True,
    )

    # Auto-clear browser-persisted stores when the released version changes, so a
    # new release drops stale cached state automatically. The version is recorded
    # in localStorage.
    clientside_callback(
        f"""
        function (pathname) {{
            const current = "{__version__}";
            const stored = window.localStorage.getItem("ml-peg-store-version");
            if (stored !== current) {{
                window.localStorage.clear();
                window.sessionStorage.clear();
                window.localStorage.setItem("ml-peg-store-version", current);
                if (stored !== null) {{
                    window.location.reload();
                }}
            }}
            return "";
        }}
        """,
        Output("storage-version-dummy", "children"),
        Input("app-location", "pathname"),
        prevent_initial_call=False,
    )
