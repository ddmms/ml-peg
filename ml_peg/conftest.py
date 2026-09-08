"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pytest import Config, Item, Parser

from ml_peg import models
from ml_peg.analysis import ANALYSIS_ROOT
from ml_peg.calcs import CALCS_ROOT

if TYPE_CHECKING:
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter


def pytest_addoption(parser: Parser) -> None:
    """
    Add flag to run tests for extra MLIPs.

    Parameters
    ----------
    parser
        Pytest parser object.
    """
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


def _is_benchmark(path: Path) -> bool:
    """
    Identify calculation and analysis scripts belonging to this checkout.

    Parameters
    ----------
    path
        Absolute path to a collected test module.

    Returns
    -------
    bool
        Whether the path follows the ML-PEG benchmark layout.
    """
    return any(
        path.is_relative_to(root)
        and len(path.relative_to(root).parts) == 3
        and path.name.startswith(prefix)
        and path.suffix == ".py"
        for root, prefix in ((CALCS_ROOT, "calc_"), (ANALYSIS_ROOT, "analyse_"))
    )


def _is_calculation(path: Path) -> bool:
    """
    Identify calculation scripts belonging to this checkout.

    Parameters
    ----------
    path
        Absolute path to a collected test module.

    Returns
    -------
    bool
        Whether the path is an ML-PEG calculation benchmark script.
    """
    return (
        path.is_relative_to(CALCS_ROOT)
        and len(path.relative_to(CALCS_ROOT).parts) == 3
        and path.name.startswith("calc_")
        and path.suffix == ".py"
    )


class CitationReporter:
    """
    Report what to cite for the calculations a pytest session actually ran.

    Tests outside the calculation tree are ignored, so analysis and package tests
    produce no citation output.

    Parameters
    ----------
    config
        Pytest configuration object.
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
        self.script_paths: set[Path] = set()
        self.framework_ids: dict[Path, set[str]] = {}

    def pytest_collection_modifyitems(self, items: list[Item]) -> None:
        """
        Record the source frameworks each calculation script is tagged with.

        Parameters
        ----------
        items
            Collected test items.
        """
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

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """
        Record a calculation script when its test was not skipped or deselected.

        Parameters
        ----------
        report
            Test report emitted by pytest for one test phase.
        """
        if report.when != "call" or report.skipped:
            return
        path = self.rootpath / Path(report.fspath)
        if _is_calculation(path):
            self.script_paths.add(path)

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter) -> None:
        """
        Print citation guidance for the calculations that ran.

        Parameters
        ----------
        terminalreporter
            Pytest terminal reporter used to write the summary.
        """
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


def pytest_configure(config: Config) -> None:
    """
    Configure pytest to custom markers and CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    # Create custom markers
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


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """
    Deselect tests outside the requested framework(s).

    Parameters
    ----------
    config
        Pytest configuration object.
    items
        Collected test items, modified in place to remove deselected tests.
    """
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
