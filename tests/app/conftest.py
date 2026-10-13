"""Fixtures for browser-driven ML-PEG app tests.

A small subset of the app is built once per test session and served on an
ephemeral port via a background Werkzeug thread. This exercises the real Dash app
(layout + callbacks) while avoiding the full ~13s cold start of every benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
from importlib import import_module
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


def wait_for_app_ready(page, timeout: int = READY_TIMEOUT) -> None:  # noqa: ANN001
    """
    Block until the app has hydrated and hidden its start-up mask.

    Waiting on ``#startup-mask`` being ``state="hidden"`` is not enough: the
    mask is part of the layout Dash fetches after ``load``, so it is briefly
    absent — and Playwright counts "detached" as hidden, making such a wait
    return immediately against a blank page. Hence exists *and* hidden.

    Parameters
    ----------
    page
        Playwright page to wait on.
    timeout
        Milliseconds to wait before failing.
    """
    page.wait_for_function(
        """() => {
          const mask = document.getElementById('startup-mask');
          return mask !== null && mask.style.display === 'none';
        }""",
        timeout=timeout,
    )


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
def fixture_data(tmp_path_factory) -> Path:
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

    extract_root = tmp_path_factory.mktemp("ui-fixture")
    prefix = f"ui-fixture-{FIXTURE_VERSION}/"
    try:
        with zipfile.ZipFile(FIXTURE_ARCHIVE) as archive:
            members = [name for name in archive.namelist() if name.startswith(prefix)]
            archive.extractall(extract_root, members=members)
    except zipfile.BadZipFile as err:
        raise RuntimeError(f"Test fixture {FIXTURE_ARCHIVE} is not a zip file") from err
    return extract_root / prefix


@pytest.fixture(scope="session")
def benchmark_data(fixture_data: Path, tmp_path_factory) -> Iterator[Path]:
    """
    Make the fixture benchmarks' data available under the app's assets folder.

    Redirect benchmark module paths and Dash assets to an isolated temporary
    tree. Local analysis outputs cannot change the dataset or be overwritten.

    Parameters
    ----------
    fixture_data
        Root of the unpacked fixture.
    tmp_path_factory
        Session temporary-directory factory.

    Yields
    ------
    Path
        Isolated assets directory.
    """
    data_root = tmp_path_factory.mktemp("ui-assets")
    shutil.copytree(APP_ROOT / "data" / "ui", data_root / "ui")
    patches = pytest.MonkeyPatch()
    for category, benchmark in FIXTURE_BENCHMARKS:
        target = data_root / category / benchmark
        shutil.copytree(fixture_data / category / benchmark, target)
        module = import_module(f"ml_peg.app.{category}.{benchmark}.app_{benchmark}")
        # Redirect all module-level data paths, including derived info paths.
        for name, value in vars(module).copy().items():
            if isinstance(value, Path) and value.is_relative_to(APP_ROOT / "data"):
                patches.setattr(
                    module, name, data_root / value.relative_to(APP_ROOT / "data")
                )
    try:
        yield data_root
    finally:
        patches.undo()


@pytest.fixture(scope="session")
def app_url(benchmark_data: Path) -> Iterator[str]:
    """
    Build the ML-PEG app for the fixture benchmarks and serve it for the session.

    Discover only the explicitly selected benchmarks and fail if any are
    missing. Unrelated local benchmarks must not affect browser tests.

    Parameters
    ----------
    benchmark_data
        Ensures the benchmarks' data is in place before the app is built.

    Yields
    ------
    str
        Base URL of the running app.
    """
    from dash import Dash

    from ml_peg.app import build_app
    from ml_peg.app.utils.head_scripts import inject_head_scripts

    app = Dash(
        __name__,
        assets_folder=str(benchmark_data),
        title="ML-PEG",
        update_title=None,
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
    )
    inject_head_scripts(app)

    discover = build_app.get_all_tests

    def fixture_tests(**kwargs):
        result = ({}, {}, {}, {})
        for category, benchmark in FIXTURE_BENCHMARKS:
            selected = discover(category=category, test=benchmark)
            result[0].update(selected[0])
            for output, incoming in zip(result[1:], selected[1:], strict=True):
                for key, value in incoming.items():
                    output.setdefault(key, {}).update(value)
        assert set(result[0]) == {name for _, name in FIXTURE_BENCHMARKS}
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(build_app, "get_all_tests", fixture_tests)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_app.build_full_app(app, category="*", test="*")

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

    server = make_server("127.0.0.1", 0, app.server, threaded=True)
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
    wait_for_app_ready(page)
    return page
