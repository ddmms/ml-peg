"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pytest import Config, Item, Parser

from ml_peg import models
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


def _is_calculation(path: Path) -> bool:
    """
    Return whether a path is an ML-PEG calculation benchmark script.

    Parameters
    ----------
    path
        Absolute path to a test file.

    Returns
    -------
    bool
        Whether `path` is a ``calc_*.py`` script at
        ``<CALCS_ROOT>/<category>/<benchmark>/``.
    """
    return (
        path.is_relative_to(CALCS_ROOT)
        and len(path.relative_to(CALCS_ROOT).parts) == 3
        and path.name.startswith("calc_")
        and path.suffix == ".py"
    )


class CitationReporter:
    """
    Report citations for the calculation benchmarks that actually ran.

    Parameters
    ----------
    config
        Pytest configuration object.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialise an empty record of executed calculations.

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
        Record the source frameworks attached to calculation tests.

        Parameters
        ----------
        items
            Collected test items.
        """
        for item in items:
            if not _is_calculation(item.path):
                continue
            ids = {
                framework_id
                for marker in item.iter_markers(name="framework")
                for framework_id in marker.args
            }
            if ids:
                self.framework_ids.setdefault(item.path, set()).update(ids)

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """
        Record calculation tests that reached their call phase.

        Parameters
        ----------
        report
            Test report emitted by pytest for one test phase.
        """
        if report.when != "call" or report.skipped:
            return
        # Reports have no absolute path, and fspath is relative to the rootdir
        path = self.rootpath / report.fspath
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

        # Imported here so collection does not pay for loading the citation module
        from ml_peg.utils.citations import build_run_citations

        summary = build_run_citations(
            self.script_paths,
            framework_ids_by_script=self.framework_ids,
        )
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
