"""Fixtures for browser-driven ML-PEG app tests.

A small subset of the app is built once per test session and served on an
ephemeral port via a background Werkzeug thread. This exercises the real Dash app
(layout + callbacks) while avoiding the full ~13s cold start of every benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
import threading

import pytest
from werkzeug.serving import make_server

# Quieten the per-request werkzeug access log so test output stays readable.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# One representative benchmark: it exercises a metrics table, an embedded plot,
# weight + threshold controls, the element filter and download controls, plus the
# category and overall summary tables that wrap it.
TEST_CATEGORY = "non_covalent_interactions"
TEST_BENCHMARK = "IONPI19"

# Generous ceiling for first hydration of the app under CI load; the test files
# mirror this value in their own TIMEOUT constants.
READY_TIMEOUT = 60_000


@pytest.fixture(scope="session")
def app_url() -> Iterator[str]:
    """
    Build the ML-PEG app for a subset and serve it for the test session.

    Yields
    ------
    str
        Base URL of the running app.
    """
    from ml_peg.app import run_app as run_app_module
    from ml_peg.app.build_app import build_full_app

    build_full_app(run_app_module.app, category=TEST_CATEGORY, test=TEST_BENCHMARK)

    server = make_server("127.0.0.1", 0, run_app_module.app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def ready_page(page, app_url):  # noqa: ANN001
    """
    Return a Playwright page with the app loaded and onboarding pre-dismissed.

    Parameters
    ----------
    page
        Playwright page fixture (function scoped).
    app_url
        Base URL of the running app.

    Returns
    -------
    Page
        Loaded, hydrated page ready for interaction.
    """
    # Mark the tutorial complete before load so its modal never overlays the page
    # (the modal is gated on the locally-persisted ``onboarding-state-store``).
    page.add_init_script(
        "try { window.localStorage.setItem('onboarding-state-store', "
        "JSON.stringify({completed: true})); } catch (e) {}"
    )
    page.goto(app_url)
    page.wait_for_selector("#startup-mask", state="hidden", timeout=READY_TIMEOUT)
    return page
