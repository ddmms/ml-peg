"""Run main application."""

from __future__ import annotations

import os
from pathlib import Path

from dash import Dash

from ml_peg.app.build_app import build_full_app
from ml_peg.app.utils.head_scripts import build_analytics_head, build_no_flash_script

DATA_PATH = Path(__file__).parent / "data"
ANALYTICS_ID = os.environ.get("ML_PEG_ANALYTICS_ID")


def _build_full_app(app: Dash, category: str, test: str):
    """
    Build full app layout and callbacks.

    Parameters
    ----------
    app
        Dash application.
    category
        Category to build application for.
    test
        Test to build application for.
    """
    build_full_app(app, category, test)


# Load the async gtag script in <head> only when an analytics ID is configured
_analytics_scripts = (
    [
        {
            "src": f"https://www.googletagmanager.com/gtag/js?id={ANALYTICS_ID}",
            "async": True,
        }
    ]
    if ANALYTICS_ID
    else []
)

# Make server accessible for gunicorn
app = Dash(
    __name__,
    assets_folder=DATA_PATH,
    title="ML-PEG",  # set browser tab title
    update_title=None,  # prevent the tab changing to Updating... during callbacks
    external_scripts=_analytics_scripts,
    # Benchmark cards mount their bodies lazily, so their control ids are absent
    # from the initial layout; allow callbacks to reference not-yet-present ids.
    suppress_callback_exceptions=True,
    meta_tags=[
        # Render at real device width so the responsive layout + media queries
        # apply on phones (without this, mobile browsers assume a ~980px page).
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

# Apply the saved preferences before first paint so a reload doesn't flash the
# wrong theme, zoom or font while Dash hydrates (see utils/head_scripts.py).
app.index_string = app.index_string.replace(
    "{%metas%}", "{%metas%}\n" + build_no_flash_script()
)

# Inject the inline gtag init into the parsed <head> so the browser executes it
if ANALYTICS_ID:
    app.index_string = app.index_string.replace(
        "{%metas%}", "{%metas%}\n" + build_analytics_head(ANALYTICS_ID)
    )

# Only build app when in production, otherwise run_app's layout is missing
if bool(os.environ.get("ML_PEG_PROD", False)):
    _build_full_app(app, "*", "*")

server = app.server


def run_app(
    category: str = "*",
    test: str = "*",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Set port and run Dash application.

    Parameters
    ----------
    category
        Category to build app for. Default is `*`, corresponding to all categories.
    test
        Test to build app for. Default is `*`, corresponding to all tests.
    port
        Port to run application on. Default is 8050.
    debug
        Whether to run with Dash debugging. Default is `True`.
    """
    _build_full_app(app, category=category, test=test)

    print(f"Starting Dash app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    run_app(port=port, debug=True)
