"""Run main application."""

from __future__ import annotations

import os
from pathlib import Path

from dash import Dash

from ml_peg.app.build_app import build_full_app

DATA_PATH = Path(__file__).parent / "data"


def run_app(
    category: str = "*",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Set port and run Dash application.

    Parameters
    ----------
    category
        Category to build app for. Default is `*`, corresponding to all categories.
    port
        Port to run application on. Default is 8050.
    debug
        Whether to run with Dash debugging. Default is `True`.
    """
    full_app = Dash(__name__, assets_folder=DATA_PATH)
    build_full_app(full_app, category=category)

    print(f"Starting Dash app on port {port}...")
    full_app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    run_app(port=port, debug=True)
