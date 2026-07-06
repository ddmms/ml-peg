"""Configure pytest for calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import CallInfo, Config, Item, Parser, TerminalReporter

from ml_peg import models
from ml_peg.calcs.utils import completion

DRY_RUN_STATUS = pytest.StashKey[list]()


def pytest_addoption(parser: Parser) -> None:
    """
    Add custom CLI inputs to pytest.

    Parameters
    ----------
    parser
        Pytest parser object.
    """
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
        "--force-calcs",
        action="store_true",
        default=False,
        help="Run calculations even if they previously completed",
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report which calculations would run, without running any",
    )


def pytest_configure(config: Config) -> None:
    """
    Configure pytest to custom CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    # Set current models from CLI input
    models.run_mock = config.getoption("--run-mock")
    models.mock_only = config.getoption("--mock-only")


def _item_mlip(item: Item) -> tuple[str, Any] | None:
    """
    Get the model a test item is parametrized with, if any.

    Parameters
    ----------
    item
        Pytest test item.

    Returns
    -------
    tuple[str, Any] | None
        The item's (model_name, model) "mlip" parameter, or None if the test
        is not parametrized over models.
    """
    callspec = getattr(item, "callspec", None)
    mlip = callspec.params.get("mlip") if callspec is not None else None
    if isinstance(mlip, tuple) and len(mlip) == 2 and isinstance(mlip[0], str):
        return mlip
    return None


def _completion_test_name(item: Item) -> str:
    """
    Get the test name to key completion markers on for a model ("mlip") test.

    Uses the unparametrized test name, extended with any parameters other
    than the model itself (e.g. a case index), so each parametrized case is
    tracked separately per model.

    Parameters
    ----------
    item
        Pytest test item.

    Returns
    -------
    str
        Test name to key completion markers on.
    """
    callspec = getattr(item, "callspec", None)
    extra = {
        name: value
        for name, value in (callspec.params.items() if callspec is not None else ())
        if name != "mlip"
    }
    if not extra:
        return item.originalname
    suffix = "-".join(f"{name}={value}" for name, value in sorted(extra.items()))
    return f"{item.originalname}[{suffix}]"


def _module_models(item: Item) -> dict[str, Any] | None:
    """
    Get the module-level MODELS dict for a test item, if defined.

    Parameters
    ----------
    item
        Pytest test item.

    Returns
    -------
    dict[str, Any] | None
        The test module's MODELS dict, or None if not defined.
    """
    module_models = getattr(getattr(item, "module", None), "MODELS", None)
    return module_models if isinstance(module_models, dict) else None


def _model_statuses(item: Item) -> dict[str, bool] | None:
    """
    Get completion status for each model a test item would run.

    Parameters
    ----------
    item
        Pytest test item.

    Returns
    -------
    dict[str, bool] | None
        Whether each model's calculation previously completed with identical
        inputs, or None if the test does not run models.
    """
    mlip = _item_mlip(item)
    if mlip is not None:
        names = [mlip[0]]
        test_name = _completion_test_name(item)
    else:
        module_models = _module_models(item)
        if module_models is None:
            return None
        names = list(module_models)
        test_name = item.name

    calc_dir = item.path.parent
    return {
        name: completion.is_complete(
            calc_dir / "outputs",
            name,
            test_name,
            completion.calc_fingerprint(calc_dir, name),
        )
        for name in names
    }


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """
    Report which calculations would run in dry run or collect-only mode.

    In dry run mode, all calculations are also skipped.

    Parameters
    ----------
    config
        Pytest configuration object.
    items
        Collected test items.
    """
    dry_run = config.getoption("--dry-run", default=False)
    if not dry_run and not config.getoption("collectonly", default=False):
        return

    calcs_dir = Path(__file__).parent
    status_lines = config.stash.setdefault(DRY_RUN_STATUS, [])
    skip_dry_run = pytest.mark.skip(reason="dry run")
    for item in items:
        if not item.path.is_relative_to(calcs_dir):
            continue
        for name, done in (_model_statuses(item) or {}).items():
            status = "up to date" if done else "would run"
            status_lines.append(f"{status}: {item.nodeid} - {name}")
        if dry_run:
            item.add_marker(skip_dry_run)


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: int, config: Config
) -> None:
    """
    Print the calculation statuses gathered in dry run mode.

    Parameters
    ----------
    terminalreporter
        Pytest terminal reporter.
    exitstatus
        Exit status that will be reported to the operating system.
    config
        Pytest configuration object.
    """
    status_lines = config.stash.get(DRY_RUN_STATUS, None)
    if not status_lines:
        return

    terminalreporter.section("calculations dry run")
    for line in sorted(status_lines):
        terminalreporter.write_line(line)

    pending = sum(line.startswith("would run") for line in status_lines)
    terminalreporter.write_line(
        f"{pending} calculation(s) to run, {len(status_lines) - pending} up to date"
    )


def pytest_runtest_setup(item: Item) -> None:
    """
    Skip calculations that previously completed with identical inputs.

    Tests parametrized over models ("mlip") are skipped per model. For tests
    that loop over the module's MODELS dict instead, completed models are
    removed from the dict, so the loop only runs the remaining models, and
    the test is skipped entirely if none remain. Pruned models are restored
    in pytest_runtest_teardown.

    Parameters
    ----------
    item
        Pytest test item.
    """
    mlip = _item_mlip(item)
    module_models = _module_models(item)
    if mlip is None and module_models is None:
        return

    completion.clear_data_files()
    if mlip is None:
        item._pruned_models = {}
    if item.config.getoption("--force-calcs"):
        return

    statuses = _model_statuses(item)

    if mlip is not None:
        if statuses[mlip[0]]:
            pytest.skip(
                f"'{mlip[0]}' previously completed. Use --force-calcs to re-run."
            )
        return

    for name, done in statuses.items():
        if done:
            print(f"[skip] {item.name}: '{name}' previously completed")
            item._pruned_models[name] = module_models.pop(name)

    if item._pruned_models and not module_models:
        pytest.skip("All models previously completed. Use --force-calcs to re-run.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo):
    """
    Record whether the test call phase passed.

    Needed as pytest does not expose the test outcome to
    pytest_runtest_teardown, which must only mark calculations as completed
    if the test passed.

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
        item._calcs_passed = True


def pytest_runtest_teardown(item: Item) -> None:
    """
    Mark completed calculations and restore pruned models.

    Parameters
    ----------
    item
        Pytest test item.
    """
    mlip = _item_mlip(item)
    pruned = getattr(item, "_pruned_models", None)
    module_models = _module_models(item)

    if getattr(item, "_calcs_passed", False):
        if mlip is not None:
            test_name, names = _completion_test_name(item), [mlip[0]]
        elif pruned is not None:
            test_name, names = item.name, list(module_models)
        else:
            test_name, names = item.name, []

        calc_dir = item.path.parent
        data_files = completion.used_data_files()
        for name in names:
            completion.mark_complete(
                calc_dir / "outputs",
                name,
                test_name,
                completion.calc_fingerprint(calc_dir, name),
                data_files,
            )

    if pruned is not None and module_models is not None:
        module_models.update(pruned)
