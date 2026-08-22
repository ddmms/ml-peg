"""Fixtures for browser-driven ML-PEG app tests.

A small subset of the app is built once per test session and served on an
ephemeral port via a background Werkzeug thread. This exercises the real Dash app
(layout + callbacks) while avoiding the full ~13s cold start of every benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path
import shutil
import threading
import warnings

import pytest
from werkzeug.serving import make_server

from ml_peg.app import APP_ROOT

# Quieten the per-request werkzeug access log so test output stays readable.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Committed stand-in for the published data tarball (~2.4 MB against ~2 GB
# compressed / ~11 GB unpacked). The suite never read more than a handful of
# benchmarks, and downloading meant testing whatever was last published rather
# than a fixed input.
FIXTURE_DATA = Path(__file__).parents[1] / "data" / "app"

# Dropped into each directory this fixture copies, so a run interrupted before
# teardown can tell its own leftovers from a developer's real data.
FIXTURE_MARKER = ".ml-peg-test-fixture"

# Benchmarks the fixture supplies, chosen for coverage per KB rather than size:
#   IONPI19          one metric, a parity plot and a structure viewer; the
#                    benchmark every interaction test targets by id.
#   extensivity      physicality category (weight 1, so the overall score is a
#                    real weighted mean) and the only mace-multihead benchmark
#                    here, so /framework/mace-multihead exists.
#   oxidation_states three metrics with experimental levels of theory, and the
#                    per-cell plot dispatch rather than IONPI19's per-column one.
#   iron_properties  twelve metric columns and dropdown-driven figures, i.e. the
#                    widest weight/threshold grid and a custom callback shape.
# Together they span three categories, which is what puts more than one slice in
# the overall summary table and the Explorer.
FIXTURE_BENCHMARKS = (
    ("non_covalent_interactions", "IONPI19"),
    ("physicality", "extensivity"),
    ("physicality", "oxidation_states"),
    ("bulk_crystal", "iron_properties"),
)

# The benchmark the interaction tests drive by id.
TEST_CATEGORY = "non_covalent_interactions"
TEST_BENCHMARK = "IONPI19"

# Generous ceiling for first hydration of the app under CI load; the test files
# mirror this value in their own TIMEOUT constants.
READY_TIMEOUT = 60_000


@pytest.fixture(scope="session", autouse=True)
def benchmark_data() -> Iterator[None]:
    """
    Make the fixture benchmarks' data available under the app's assets folder.

    The app resolves data through ``APP_ROOT / "data"`` and serves the same tree
    as Dash assets, so the fixture has to be visible there rather than read from
    ``tests/data``. A developer who already has the real data keeps it: this only
    fills in the gaps, and only removes the directories it created.

    Yields
    ------
    None
        Control returns to the test session once the data is in place.
    """
    created: list[Path] = []
    for category, benchmark in FIXTURE_BENCHMARKS:
        target = APP_ROOT / "data" / category / benchmark
        # A previous run killed mid-session leaves its copy behind. The marker
        # says the directory is ours, so re-adopt it rather than mistaking it for
        # real data and never cleaning it up again.
        if target.exists() and not (target / FIXTURE_MARKER).exists():
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(FIXTURE_DATA / category / benchmark, target)
        (target / FIXTURE_MARKER).touch()
        created.append(target)

    try:
        yield
    finally:
        for target in created:
            shutil.rmtree(target, ignore_errors=True)
            # Leave no empty category directory behind on a clean checkout.
            parent = target.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()


@pytest.fixture(scope="session")
def app_url(benchmark_data: None) -> Iterator[str]:
    """
    Build the ML-PEG app for the fixture benchmarks and serve it for the session.

    ``get_all_tests`` globs the *code* tree and skips any benchmark whose data is
    missing (warning rather than raising), so building with ``"*"`` builds
    exactly what the fixture supplies. That keeps the fixture directory as the
    single place coverage is declared, but it also means a fixture that failed to
    load would silently shrink coverage instead of failing, hence the check on
    the skip warnings below.

    Parameters
    ----------
    benchmark_data
        Ensures the benchmarks' data is in place before the app is built.

    Yields
    ------
    str
        Base URL of the running app.
    """
    from ml_peg.app import run_app as run_app_module
    from ml_peg.app.build_app import build_full_app

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_full_app(run_app_module.app, category="*", test="*")

    expected = {benchmark for _, benchmark in FIXTURE_BENCHMARKS}
    messages = [str(warning.message) for warning in caught]
    skipped = sorted(
        benchmark
        for benchmark in expected
        if any(f" {benchmark} in " in message for message in messages)
    )
    assert not skipped, (
        f"fixture benchmarks failed to build: {skipped}. The data under "
        f"{FIXTURE_DATA} is incomplete for them."
    )

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
