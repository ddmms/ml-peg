"""Fixtures for browser-driven ML-PEG app tests.

A small subset of the app is built once per test session and served on an
ephemeral port via a background Werkzeug thread. This exercises the real Dash app
(layout + callbacks) while avoiding the full ~13s cold start of every benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import logging
from pathlib import Path
import shutil
import threading
import warnings
import zipfile

from botocore.exceptions import BotoCoreError, ClientError
import pytest
from werkzeug.serving import make_server

from ml_peg.app import APP_ROOT
from ml_peg.calcs.utils.utils import BENCHMARK_DATA_DIR
from ml_peg.data.data import download

# Quieten the per-request werkzeug access log so test output stays readable.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Stand-in for the published data tarball (~800 KB against ~9 GB): a handful of
# benchmarks, published once to the ml-peg-data bucket and never overwritten. A
# changed fixture is a new -v2 object plus a bump here. The bucket has no
# versioning, so the digest is what actually enforces that: a swapped object
# fails loudly instead of quietly changing what the suite tests. (The object as
# uploaded also carries macOS "__MACOSX/" resource-fork entries, which
# extraction skips.)
FIXTURE_VERSION = "v1"
FIXTURE_KEY = f"tests/ui-fixture-{FIXTURE_VERSION}.zip"
FIXTURE_SHA256 = "b0fab46961813f329702b6d464171efc02edf73503c7c4c5069db7f91935564f"
FIXTURE_ARCHIVE = BENCHMARK_DATA_DIR / f"ui-fixture-{FIXTURE_VERSION}.zip"
# The archive's single top-level directory, which the benchmark tree sits under.
FIXTURE_ROOT = BENCHMARK_DATA_DIR / f"ui-fixture-{FIXTURE_VERSION}"

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


def _sha256(path: Path) -> str:
    """
    Return the SHA-256 hex digest of a file.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def fixture_data() -> Path:
    """
    Fetch, verify and unpack the published test fixture, returning its root.

    ``download_s3_data`` is deliberately not used: it extracts before returning,
    so there is no point at which the archive can be checked before it is
    trusted, and it unpacks every member rather than just the fixture directory.
    The cached archive is re-hashed on every session, so a truncated or stale
    copy fails the same way a swapped object does. Behaviour is identical locally
    and in CI: no network means a failed session, not a silently skipped one.

    Returns
    -------
    Path
        Directory holding the fixture's ``<category>/<benchmark>`` tree.
    """
    if not FIXTURE_ARCHIVE.exists():
        try:
            download(key=FIXTURE_KEY, filename=str(FIXTURE_ARCHIVE))
        except (BotoCoreError, ClientError) as err:
            raise RuntimeError(
                f"Could not fetch the test fixture {FIXTURE_KEY} from the "
                "ml-peg-data bucket. Check the connection, or fetch it by hand: "
                f"ml_peg download --key {FIXTURE_KEY} --filename {FIXTURE_ARCHIVE}"
            ) from err

    digest = _sha256(FIXTURE_ARCHIVE)
    if digest != FIXTURE_SHA256:
        FIXTURE_ARCHIVE.unlink()
        raise RuntimeError(
            f"Test fixture {FIXTURE_ARCHIVE} has SHA-256 {digest}, expected "
            f"{FIXTURE_SHA256}. The archive has been deleted; rerun to fetch it "
            "again. If the published object itself changed, it should have been "
            "a new version rather than an overwrite."
        )

    # Unpack afresh so a stale extract from an earlier version can't mix in.
    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)
    prefix = f"{FIXTURE_ROOT.name}/"
    try:
        with zipfile.ZipFile(FIXTURE_ARCHIVE) as archive:
            members = [name for name in archive.namelist() if name.startswith(prefix)]
            archive.extractall(BENCHMARK_DATA_DIR, members=members)
    except zipfile.BadZipFile as err:
        raise RuntimeError(f"Test fixture {FIXTURE_ARCHIVE} is not a zip file") from err
    return FIXTURE_ROOT


@pytest.fixture(scope="session", autouse=True)
def benchmark_data(fixture_data: Path) -> Iterator[None]:
    """
    Make the fixture benchmarks' data available under the app's assets folder.

    The app resolves data through ``APP_ROOT / "data"`` and serves the same tree
    as Dash assets, so the fixture has to be visible there rather than read from
    the cache. A developer who already has the real data keeps it: this only
    fills in the gaps, and only removes the directories it created.

    Parameters
    ----------
    fixture_data
        Root of the unpacked fixture.

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
        shutil.copytree(fixture_data / category / benchmark, target)
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
        f"fixture benchmarks failed to build: {skipped}. The fixture "
        f"{FIXTURE_KEY} is incomplete for them."
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
