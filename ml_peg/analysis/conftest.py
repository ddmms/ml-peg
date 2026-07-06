"""Configure pytest for analysis."""

from __future__ import annotations

import pytest
from pytest import CallInfo, Item, Parser

from ml_peg.calcs.utils import completion


def pytest_addoption(parser: Parser) -> None:
    """
    Add custom CLI inputs to pytest.

    Parameters
    ----------
    parser
        Pytest parser object.
    """
    parser.addoption(
        "--force-analysis",
        action="store_true",
        default=False,
        help="Run analysis even if it previously completed",
    )


def pytest_runtest_setup(item: Item) -> None:
    """
    Skip analysis that previously completed with identical inputs.

    Unlike calculations, analysis aggregates all models into shared outputs,
    so completion is tracked per test rather than per model. Modules missing
    any of MODELS, CALC_PATH or OUT_PATH always run and keep no marker.

    Parameters
    ----------
    item
        Pytest test item.
    """
    module = getattr(item, "module", None)
    models = getattr(module, "MODELS", None)
    calc_path = getattr(module, "CALC_PATH", None)
    out_path = getattr(module, "OUT_PATH", None)
    if models is None or calc_path is None or out_path is None:
        return

    completion.clear_data_files()
    item._analysis_fingerprint = completion.analysis_fingerprint(
        item.path.parent, models, calc_path
    )
    item._analysis_out_path = out_path

    # Empty model name: analysis markers live directly in OUT_PATH, not per model
    if not item.config.getoption("--force-analysis") and completion.is_complete(
        out_path, "", item.name, item._analysis_fingerprint
    ):
        pytest.skip("Analysis previously completed. Use --force-analysis to re-run.")

    # The analysis will run: drop any previous marker entry so a failed or
    # interrupted run cannot leave outputs masked as complete
    completion.unmark_complete(out_path, "", item.name)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo):
    """
    Record whether the test call phase passed.

    Needed as pytest does not expose the test outcome to
    pytest_runtest_teardown, which must only mark analysis as completed if
    the test passed.

    Parameters
    ----------
    item
        Pytest test item.
    call
        Result of the test phase that just ran.

    Yields
    ------
    Result
        Hook wrapper result holding the test report.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed:
        item._analysis_passed = True


def pytest_runtest_teardown(item: Item) -> None:
    """
    Mark completed analysis.

    Parameters
    ----------
    item
        Pytest test item.
    """
    fingerprint = getattr(item, "_analysis_fingerprint", None)
    if fingerprint is None or not getattr(item, "_analysis_passed", False):
        return

    completion.mark_complete(
        item._analysis_out_path,
        "",
        item.name,
        fingerprint,
        completion.used_data_files(),
    )
