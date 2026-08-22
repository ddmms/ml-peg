"""
Buttons and callbacks for clearing the app data saved in the browser.

Kept in its own file so ``build_app`` stays short. ``build_header_controls``
makes the controls shown in the top-right corner; ``register_storage_callbacks``
makes the Hard Reset button work: clearing the saved data when it is clicked,
and automatically after a new version is released.
"""

from __future__ import annotations

from dash import Input, Output, clientside_callback
from dash.html import Div

from ml_peg import __version__
from ml_peg.app.utils.onboarding import build_tutorial_button
from ml_peg.app.utils.settings import build_settings_panel

# Both clear paths wipe localStorage wholesale but keep the pure UI preferences
# (theme, table zoom, colour scheme, font, expand-all): losing dark mode or an
# accessibility zoom on
# a version bump (or an explicit cache clear, which only promises to reset
# weights/thresholds/tutorial progress) would read as a bug. Everything else —
# weights, thresholds, tutorial state — is wiped by design.
_CLEAR_STORAGE_JS = """
    const preserved = [
        "theme-store", "zoom-store", "font-store", "bench-expand-store",
        "cmap-store", "ml-peg-store-version"
    ].map(
        (key) => [key, window.localStorage.getItem(key)]
    );
    window.localStorage.clear();
    window.sessionStorage.clear();
    for (const [key, value] of preserved) {
        if (value !== null) {
            window.localStorage.setItem(key, value);
        }
    }
"""


def build_header_controls() -> Div:
    """
    Build the controls shown in the top-right corner of the app.

    Holds the settings popover (theme, colour scheme, expand preference, clear
    cache) next to the "Tutorial" button. The hidden Divs are not shown, they
    just give the callbacks somewhere to write to.

    Returns
    -------
    Div
        Container holding the top-right controls.
    """
    return Div(
        [
            build_settings_panel(),
            build_tutorial_button(),
            Div(id="clear-storage-dummy", style={"display": "none"}),
            Div(id="storage-version-dummy", style={"display": "none"}),
            Div(id="theme-apply-dummy", style={"display": "none"}),
            Div(id="zoom-apply-dummy", style={"display": "none"}),
            Div(id="font-apply-dummy", style={"display": "none"}),
        ],
        className="mlpeg-header-actions",
    )


def register_storage_callbacks() -> None:
    """Register the Hard Reset and version-bump auto-clear clientside callbacks."""
    # Clear all browser-persisted dcc.Store data (session + local) and reload, so
    # stale cached state after an update can be wiped from the header button.
    clientside_callback(
        f"""
        function (n_clicks) {{
            if (n_clicks && window.confirm(
                "Clear cached app data and reload? Saved weights and thresholds"
                + " will be reset."
            )) {{
                {_CLEAR_STORAGE_JS}
                window.location.reload();
            }}
            return "";
        }}
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
                {_CLEAR_STORAGE_JS}
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
