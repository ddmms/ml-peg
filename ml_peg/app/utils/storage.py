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


def build_version_check_script() -> str:
    """
    Build the pre-hydration script that drops cached state after a version bump.

    Runs from ``<head>`` rather than as a callback, because a ``dcc.Store``
    reads ``localStorage`` while it mounts and writes it back whenever its data
    changes. A callback fires after all of that, so clearing there races the
    stores it is clearing: any write still in flight re-persists the stale value
    the bump was meant to drop, and it takes a reload to recover. Nothing has
    read ``localStorage`` yet at this point, so no reload is needed either.

    Returns
    -------
    str
        A ``<script>`` element, ready to be injected ahead of the parsed head.
    """
    return (
        "    <script>(function(){try{"
        f'var current="{__version__}";'
        'var stored=window.localStorage.getItem("ml-peg-store-version");'
        "if(stored!==current){"
        # A first visit has nothing stale to drop; only record the version.
        "if(stored!==null){"
        f"{_CLEAR_STORAGE_JS}"
        "}"
        'window.localStorage.setItem("ml-peg-store-version",current);'
        "}}catch(e){}})();</script>"
    )


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

    # The version-bump auto-clear is NOT a callback: see
    # build_version_check_script.
