"""Configure pytest for calculations."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import catch_warnings, simplefilter

from pytest import Config, Item, Parser, hookimpl, mark

from ml_peg import models
from ml_peg.calcs.utils.run_log import (
    Status,
    benchmark_id,
    end_run,
    format_error,
    format_warning,
    start_run,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any
    from warnings import WarningMessage

    from _pytest.nodes import Node
    from _pytest.reports import TestReport
    from _pytest.runner import CallInfo

#: Parameter names that calculations use for the model under test.
MODEL_PARAMS = ("mlip", "model_name")


def pytest_addoption(parser: Parser) -> None:
    """
    Add custom CLI inputs to pytest.

    Parameters
    ----------
    parser
        Pytest parser object.
    """
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
        "--run-multi-day",
        action="store_true",
        default=False,
        help="Run multi-day benchmarks",
    )
    parser.addoption(
        "--run-mock",
        action="store_true",
        default=False,
        help="Include mock model in tests",
    )
    parser.addoption(
        "--mock-only",
        action="store_true",
        default=False,
        help="Only run mock model, ignoring other models",
    )
    parser.addoption(
        "--no-run-log",
        action="store_true",
        default=False,
        help="Do not record run logs in benchmark output directories",
    )


class RunLogger:
    """
    Record when each benchmark ran for each model, and whether it completed.

    Calculations are logged without any benchmark needing to opt in, by reading the
    model from the test's parameters and the output directory from its module.
    """

    def __init__(self) -> None:
        """Initialise an empty record of in-progress runs."""
        self.log_paths: dict[str, Path] = {}
        self.warnings: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        self.errors: dict[str, dict[str, str]] = {}
        self.statuses: dict[str, Status] = {}

    @staticmethod
    def _model_name(params: dict[str, Any]) -> str | None:
        """
        Get the model a test is parametrised over.

        Parameters
        ----------
        params
            Parameters of the test, from its callspec.

        Returns
        -------
        str | None
            Name of the model, or `None` if the test is not run per model.
        """
        for name in MODEL_PARAMS:
            if name not in params:
                continue
            value = params[name]
            # Tests take either (model_name, model) or just the name
            return value[0] if isinstance(value, tuple) else value
        return None

    def _target(self, item: Item) -> tuple[Path, str, str, dict[str, Any]] | None:
        """
        Get what a test would be logged as, if it is a calculation.

        Parameters
        ----------
        item
            Test to describe.

        Returns
        -------
        tuple[Path, str, str, dict[str, Any]] | None
            Output directory, model, benchmark and remaining parameters, or `None`
            if the test is not a calculation run for a single model.
        """
        path = Path(item.path)
        benchmark = benchmark_id(path)
        if benchmark is None or not path.name.startswith("calc_"):
            return None

        params = dict(item.callspec.params) if getattr(item, "callspec", None) else {}
        model_name = self._model_name(params)
        if model_name is None:
            return None

        out_path = getattr(item.module, "OUT_PATH", path.parent / "outputs")
        extra = {key: value for key, value in params.items() if key not in MODEL_PARAMS}
        return out_path, model_name, benchmark, extra

    def pytest_runtest_setup(self, item: Item) -> None:
        """
        Log the start of a calculation.

        Skipped tests raise before this runs, so are not logged at all, leaving a
        missing log to mean the calculation was never attempted.

        Parameters
        ----------
        item
            Test about to be run.
        """
        target = self._target(item)
        if target is None:
            return
        out_path, model_name, benchmark, params = target
        self.log_paths[item.nodeid] = start_run(
            out_path, model_name, benchmark, params or None
        )

    def pytest_warning_recorded(
        self, warning_message: WarningMessage, nodeid: str
    ) -> None:
        """
        Count a warning raised by a test.

        Benchmarks warn rather than raise when an individual structure fails, so
        these counts are the only record of a run that completed only in part.

        Parameters
        ----------
        warning_message
            Warning raised, as recorded by pytest.
        nodeid
            Identifier of the test that raised it.
        """
        self.warnings[nodeid][
            format_warning(
                warning_message.category.__name__, str(warning_message.message)
            )
        ] += 1

    def pytest_exception_interact(
        self, node: Node, call: CallInfo, report: TestReport
    ) -> None:
        """
        Record the exception that ended a calculation.

        Parameters
        ----------
        node
            Test that raised the exception.
        call
            Information about the phase that raised it.
        report
            Report for that phase.
        """
        if call.excinfo is None or node.nodeid not in self.log_paths:
            return
        self.errors[node.nodeid] = format_error(
            call.excinfo.typename, str(call.excinfo.value), report.longreprtext
        )

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """
        Record the outcome of one phase of a calculation.

        Parameters
        ----------
        report
            Report for one phase of a test.
        """
        if report.nodeid not in self.log_paths:
            return

        if report.failed:
            # A failure in any phase, including teardown, fails the whole run
            self.statuses[report.nodeid] = "failed"
        elif report.skipped:
            self.statuses.setdefault(report.nodeid, "skipped")
        elif report.when == "call":
            self.statuses.setdefault(report.nodeid, "completed")

    @hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item: Item) -> Generator[None, object, object]:
        """
        Log a calculation once pytest has finished with it.

        Warnings are flushed, and exceptions reported, only after the final phase of
        a test, so the log is written here rather than from a report. Wrapping
        first puts this outside pytest's own warnings capture, which would
        otherwise flush after the log had been written.

        Parameters
        ----------
        item
            Test being run.

        Returns
        -------
        object
            Result of running the test protocol.

        Yields
        ------
        None
            Yielded while the test runs.
        """
        if self._target(item) is None:
            return (yield)

        try:
            # Benchmarks warn once per failed structure, which the default filter
            # would collapse to a single record, losing how many actually failed
            with catch_warnings():
                simplefilter("always")
                return (yield)
        finally:
            self._finalise(item.nodeid)

    def _finalise(self, nodeid: str) -> None:
        """
        Write the outcome of a finished calculation to its log.

        Parameters
        ----------
        nodeid
            Identifier of the test that ran.
        """
        path = self.log_paths.pop(nodeid, None)
        warning_counts = self.warnings.pop(nodeid, Counter())
        error = self.errors.pop(nodeid, None)
        status = self.statuses.pop(nodeid, "failed")
        if path is not None:
            end_run(path, status, error, warning_counts)


def pytest_configure(config: Config) -> None:
    """
    Configure pytest to custom CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    # Create custom markers for speed tiers
    config.addinivalue_line("markers", "fast: seconds to minutes on GPU")
    config.addinivalue_line("markers", "medium: tens of minutes on GPU")
    config.addinivalue_line("markers", "slow: mark test as slow calculations")
    config.addinivalue_line("markers", "very_slow: mark test as very slow calculations")
    config.addinivalue_line(
        "markers", "multi_day: mark test as requiring multiple GPU days"
    )

    # Set mock options from CLI input
    models.run_mock = config.getoption("--run-mock")
    models.mock_only = config.getoption("--mock-only")

    if not config.getoption("--no-run-log"):
        config.pluginmanager.register(RunLogger())


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """
    Skip slow tests.

    Parameters
    ----------
    config
        Pytest configuration object.
    items
        Collected test items, marked in place to skip slow tests.
    """
    skip_slow = mark.skip(reason="need --run-slow option to run")
    skip_very_slow = mark.skip(reason="need --run-very-slow option to run")
    skip_multi_day = mark.skip(reason="need --run-multi-day option to run")
    for item in items:
        if "multi_day" in item.keywords and not config.getoption("--run-multi-day"):
            item.add_marker(skip_multi_day)
        elif "very_slow" in item.keywords and not config.getoption("--run-very-slow"):
            item.add_marker(skip_very_slow)
        elif "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
