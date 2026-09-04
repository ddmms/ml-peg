"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ml_peg import models
from ml_peg.analysis import ANALYSIS_ROOT
from ml_peg.calcs import CALCS_ROOT

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.nodes import Item
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter


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


class CitationReporter:
    """
    Report what to cite for the benchmarks a pytest session actually ran.

    Records the benchmark scripts that execute, then prints the citations and
    implementation credits at the end of the session. Tests outside the calculation
    and analysis trees are ignored, so running the package's own test suite produces
    no citation output.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialise an empty record of executed benchmarks.

        Parameters
        ----------
        config
            Pytest configuration object.
        """
        self.rootpath = Path(config.rootpath)
        self.models = config.getoption("--models")
        self.models_file = config.getoption("--models-file")
        self.mock_only = config.getoption("--mock-only", default=False)
        self.script_paths: set[Path] = set()
        self.framework_ids: dict[Path, set[str]] = {}

    def pytest_collection_modifyitems(self, items: list[Item]) -> None:
        """
        Record the source frameworks each benchmark script is tagged with.

        Parameters
        ----------
        items
            Collected test items.
        """
        for item in items:
            ids = {
                framework_id
                for marker in item.iter_markers(name="framework")
                for framework_id in marker.args
            }
            if ids:
                path = self.rootpath / Path(item.fspath)
                self.framework_ids.setdefault(path, set()).update(ids)

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """
        Record the benchmark script of any test that was not skipped or deselected.

        Parameters
        ----------
        report
            Test report emitted by pytest for one test phase.
        """
        if report.when != "call" or report.skipped:
            return
        # fspath is relative to the pytest rootdir
        path = self.rootpath / Path(report.fspath)
        if not path.name.startswith(("calc_", "analyse_")):
            return
        for root in (CALCS_ROOT, ANALYSIS_ROOT):
            # Benchmark scripts live at <root>/<category>/<benchmark>/<script>.py
            if path.is_relative_to(root) and len(path.relative_to(root).parts) == 3:
                self.script_paths.add(path)
                return

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter) -> None:
        """
        Print citation guidance for the benchmarks that ran.

        Parameters
        ----------
        terminalreporter
            Pytest terminal reporter used to write the summary.
        """
        if not self.script_paths:
            return

        from ml_peg.citations import build_run_citations
        from ml_peg.models.get_models import get_model_names

        model_names = (
            () if self.mock_only else get_model_names(self.models, self.models_file)
        )
        framework_ids = {
            framework_id
            for path in self.script_paths
            for framework_id in self.framework_ids.get(path, ())
        }
        summary = build_run_citations(
            self.script_paths, model_names, self.models_file, framework_ids
        )
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
