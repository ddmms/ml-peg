"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_peg import models
from ml_peg.calcs import CALCS_ROOT


def pytest_addoption(parser):
    """Add flag to run tests for extra MLIPs."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow benchmarks",
    )
    parser.addoption(
        "--run-very-slow",
        action="store_true",
        default=False,
        help="Run very slow benchmarks",
    )
    parser.addoption(
        "--models",
        action="store",
        default=None,
        help="MLIPs, in comma-separated list. Default is all models",
    )
    parser.addoption(
        "--models-file",
        action="store",
        default=None,
        help="Filepath to model definitions. Default models.yml in models directory.",
    )
    parser.addoption(
        "--framework",
        action="store",
        default=None,
        help=(
            "Run only tests belonging to these MLIP framework(s), as a "
            "comma-separated list of framework ids. Default is all tests."
        ),
    )


def _is_calculation(path: Path) -> bool:
    """Return whether a path is an ML-PEG calculation benchmark script."""
    return (
        path.is_relative_to(CALCS_ROOT)
        and len(path.relative_to(CALCS_ROOT).parts) == 3
        and path.name.startswith("calc_")
        and path.suffix == ".py"
    )


class CitationReporter:
    """Report citations for the calculation benchmarks that actually ran."""

    def __init__(self, config) -> None:
        self.rootpath = Path(config.rootpath)
        self.script_paths: set[Path] = set()
        self.framework_ids: dict[Path, set[str]] = {}

    def pytest_collection_modifyitems(self, items) -> None:
        """Record the source frameworks attached to calculation tests."""
        for item in items:
            path = self.rootpath / Path(item.fspath)
            if not _is_calculation(path):
                continue
            ids = {
                framework_id
                for marker in item.iter_markers(name="framework")
                for framework_id in marker.args
            }
            if ids:
                self.framework_ids.setdefault(path, set()).update(ids)

    def pytest_runtest_logreport(self, report) -> None:
        """Record calculation tests that reached their call phase."""
        if report.when != "call" or report.skipped:
            return
        path = self.rootpath / Path(report.fspath)
        if _is_calculation(path):
            self.script_paths.add(path)

    def pytest_terminal_summary(self, terminalreporter) -> None:
        """Print citation guidance for the calculations that ran."""
        if not self.script_paths:
            return

        from ml_peg.citations import build_run_citations

        framework_ids = {
            framework_id
            for path in self.script_paths
            for framework_id in self.framework_ids.get(path, ())
        }
        summary = build_run_citations(self.script_paths, framework_ids=framework_ids)
        terminalreporter.write_line("")
        terminalreporter.write_line(summary)


def pytest_configure(config):
    """Configure pytest to custom markers and CLI inputs."""
    # Create custom marker for slow tests
    config.addinivalue_line("markers", "slow: mark test as slow calculations")
    config.addinivalue_line("markers", "very_slow: mark test as very slow calculations")
    config.addinivalue_line(
        "markers",
        "framework(*ids): mark test as belonging to MLIP framework(s)",
    )

    # Set current models from CLI input
    models.current_models = config.getoption("--models")
    model_file = config.getoption("--models-file")
    if model_file:
        models.models_file = model_file

    config.pluginmanager.register(CitationReporter(config))


def pytest_collection_modifyitems(config, items):
    """Skip slow tests and deselect tests outside the requested framework(s)."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_very_slow = pytest.mark.skip(reason="need --run-very-slow option to run")
    for item in items:
        if "very_slow" in item.keywords and not config.getoption("--run-very-slow"):
            item.add_marker(skip_very_slow)
        elif "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)

    # Keep only tests tagged with one of the requested frameworks
    framework = config.getoption("--framework")
    if not framework:
        return
    requested = {name.strip() for name in framework.split(",") if name.strip()}
    selected = []
    deselected = []
    for item in items:
        item_frameworks = {
            fw for marker in item.iter_markers(name="framework") for fw in marker.args
        }
        (selected if item_frameworks & requested else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
