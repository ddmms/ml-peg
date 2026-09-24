"""Test custom pytest options passed by the CLI and handled by conftest.py."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from ml_peg.cli.cli import app

REPO_ROOT = Path(__file__).parents[1]

runner = CliRunner()


@pytest.fixture
def pytest_args(monkeypatch):
    """
    Record the options the CLI would pass to pytest, without running pytest.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.

    Returns
    -------
    list[str]
        Options passed to `pytest.main`, populated when the CLI is invoked.
    """
    recorded = []

    def fake_main(options):
        """
        Record options instead of running pytest.

        Parameters
        ----------
        options
            Options the CLI passes to `pytest.main`.

        Returns
        -------
        int
            Exit code indicating success.
        """
        recorded.extend(str(option) for option in options)
        return 0

    monkeypatch.setattr(pytest, "main", fake_main)
    return recorded


def run_calc_cli(*args: str) -> None:
    """
    Invoke `ml_peg calc`, checking it exits successfully.

    Parameters
    ----------
    *args
        Additional command line arguments.
    """
    result = runner.invoke(app, ["calc", *args])
    assert result.exit_code == 0, result.output


def run_analyse_cli(*args: str) -> None:
    """
    Invoke `ml_peg analyse`, checking it exits successfully.

    Parameters
    ----------
    *args
        Additional command line arguments.
    """
    result = runner.invoke(app, ["analyse", *args])
    assert result.exit_code == 0, result.output


def run_pytest(tmp_path: Path, body: str, *options: str) -> subprocess.CompletedProcess:
    """
    Run pytest on a temporary test file, with the ml_peg conftests as plugins.

    Parameters
    ----------
    tmp_path
        Directory to write the temporary test file to.
    body
        Contents of the temporary test file.
    *options
        Custom options to pass to pytest.

    Returns
    -------
    subprocess.CompletedProcess
        Completed pytest run.
    """
    (tmp_path / "test_options.py").write_text(body, encoding="utf8")

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            # Load the conftests being tested, which are not in the temporary directory
            "-p",
            "ml_peg.conftest",
            "-p",
            "ml_peg.calcs.conftest",
            "-p",
            "no:cacheprovider",
            "-rs",
            "-v",
            *options,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


SLOW_TESTS = """
import pytest


def test_fast():
    pass


@pytest.mark.slow
def test_slow():
    pass


@pytest.mark.very_slow
def test_very_slow():
    pass
"""


FRAMEWORK_TESTS = """
import pytest


def test_unmarked():
    pass


@pytest.mark.framework("mace-polar-1")
def test_polar():
    pass


@pytest.mark.framework("mlip_arena")
def test_arena():
    pass
"""


def state_test(**expected: str) -> str:
    """
    Build a test file asserting the state set in `ml_peg.models` by the conftests.

    Parameters
    ----------
    **expected
        Attributes of `ml_peg.models` mapped to their expected values, as source
        code to be compared against.

    Returns
    -------
    str
        Contents of the test file.
    """
    checks = "\n".join(
        f"    assert models.{attr} == {value}, models.{attr}"
        for attr, value in expected.items()
    )
    return f"""
from pathlib import Path

from ml_peg import models


def test_state():
{checks}
"""


def outcomes(result: subprocess.CompletedProcess) -> dict[str, str]:
    """
    Extract the outcome of each test from verbose pytest output.

    Parameters
    ----------
    result
        Completed pytest run.

    Returns
    -------
    dict[str, str]
        Names of selected tests mapped to their outcome, either "PASSED" or
        "SKIPPED". Deselected tests are not included.
    """
    results = {}
    for line in result.stdout.splitlines():
        if "::" in line and (" PASSED" in line or " SKIPPED" in line):
            name = line.split("::")[1].split()[0]
            results[name] = "PASSED" if " PASSED" in line else "SKIPPED"

    return results


@pytest.mark.parametrize(
    "options, expected",
    (
        (
            (),
            {
                "test_fast": "PASSED",
                "test_slow": "SKIPPED",
                "test_very_slow": "SKIPPED",
            },
        ),
        (
            ("--run-slow",),
            {"test_fast": "PASSED", "test_slow": "PASSED", "test_very_slow": "SKIPPED"},
        ),
        (
            ("--run-very-slow",),
            {"test_fast": "PASSED", "test_slow": "SKIPPED", "test_very_slow": "PASSED"},
        ),
        (
            ("--run-slow", "--run-very-slow"),
            {"test_fast": "PASSED", "test_slow": "PASSED", "test_very_slow": "PASSED"},
        ),
    ),
)
def test_slow_options(tmp_path, options, expected):
    """
    Test slow tests are only run when the corresponding option is passed.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    options
        Options to pass to pytest.
    expected
        Expected outcome of each test.
    """
    result = run_pytest(tmp_path, SLOW_TESTS, *options)
    assert outcomes(result) == expected


@pytest.mark.parametrize(
    "options, run_mock, mock_only",
    (
        ((), False, False),
        (("--run-mock",), True, False),
        (("--mock-only",), False, True),
        (("--run-mock", "--mock-only"), True, True),
    ),
)
def test_mock_options(tmp_path, options, run_mock, mock_only):
    """
    Test mock options set the corresponding state in `ml_peg.models`.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    options
        Options to pass to pytest.
    run_mock
        Expected value of `models.run_mock`.
    mock_only
        Expected value of `models.mock_only`.
    """
    result = run_pytest(
        tmp_path,
        state_test(run_mock=run_mock, mock_only=mock_only),
        *options,
    )
    assert result.returncode == 0, result.stdout


def test_models_option(tmp_path):
    """
    Test --models sets the current models.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    result = run_pytest(
        tmp_path,
        state_test(current_models="'mace-mp-0a,mace-mpa-0'"),
        "--models",
        "mace-mp-0a,mace-mpa-0",
    )
    assert result.returncode == 0, result.stdout


def test_models_default(tmp_path):
    """
    Test all models are used if --models is not passed.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    result = run_pytest(tmp_path, state_test(current_models=None))
    assert result.returncode == 0, result.stdout


def test_models_file_option(tmp_path):
    """
    Test --models-file sets the models file used.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    models_file = tmp_path / "my_models.yml"
    models_file.write_text("", encoding="utf8")

    result = run_pytest(
        tmp_path,
        state_test(models_file=f"'{models_file}'"),
        "--models-file",
        str(models_file),
    )
    assert result.returncode == 0, result.stdout


def test_models_file_default(tmp_path):
    """
    Test the default models file is used if --models-file is not passed.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    result = run_pytest(
        tmp_path, state_test(models_file="models.MODELS_ROOT / 'models.yml'")
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "options, selected",
    (
        ((), ("test_unmarked", "test_polar", "test_arena")),
        (("--framework", "mace-polar-1"), ("test_polar",)),
        (("--framework", "mace-polar-1,mlip_arena"), ("test_polar", "test_arena")),
        # Whitespace and empty entries in the list are ignored
        (("--framework", " mlip_arena, "), ("test_arena",)),
    ),
)
def test_framework_option(tmp_path, options, selected):
    """
    Test --framework only runs tests marked with the requested framework(s).

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    options
        Options to pass to pytest.
    selected
        Tests expected to be selected. All other tests must be deselected.
    """
    result = run_pytest(tmp_path, FRAMEWORK_TESTS, *options)

    assert outcomes(result) == dict.fromkeys(selected, "PASSED")
    deselected = 3 - len(selected)
    if deselected:
        assert f"{deselected} deselected" in result.stdout


@pytest.mark.parametrize(
    "cli_options, expected, unexpected",
    (
        ((), ("--run-slow", "--run-mock"), ("--run-very-slow", "--mock-only")),
        (("--no-run-slow",), (), ("--run-slow",)),
        (("--run-very-slow",), ("--run-very-slow",), ()),
        (("--no-run-mock",), (), ("--run-mock",)),
        (("--mock-only",), ("--run-mock", "--mock-only"), ()),
    ),
)
def test_calc_cli_mock_and_slow(pytest_args, cli_options, expected, unexpected):
    """
    Test `ml_peg calc` passes the expected slow and mock options to pytest.

    Parameters
    ----------
    pytest_args
        Fixture recording the options passed to `pytest.main`.
    cli_options
        Options to pass to `ml_peg calc`.
    expected
        Options expected to be passed to pytest.
    unexpected
        Options expected not to be passed to pytest.
    """
    run_calc_cli(*cli_options)

    for option in expected:
        assert option in pytest_args
    for option in unexpected:
        assert option not in pytest_args


@pytest.mark.parametrize("command", ("calc", "analyse"))
def test_cli_model_options(pytest_args, command, tmp_path):
    """
    Test `ml_peg calc` and `ml_peg analyse` pass model options to pytest.

    Parameters
    ----------
    pytest_args
        Fixture recording the options passed to `pytest.main`.
    command
        CLI command to run.
    tmp_path
        Pytest temporary directory fixture.
    """
    models_file = tmp_path / "my_models.yml"

    result = runner.invoke(
        app,
        [
            command,
            "--models",
            "mace-mp-0a,mace-mpa-0",
            "--models-file",
            str(models_file),
            "--framework",
            "mace-polar-1",
        ],
    )
    assert result.exit_code == 0, result.output

    assert pytest_args[pytest_args.index("--models") + 1] == "mace-mp-0a,mace-mpa-0"
    assert pytest_args[pytest_args.index("--models-file") + 1] == str(models_file)
    assert pytest_args[pytest_args.index("--framework") + 1] == "mace-polar-1"


@pytest.mark.parametrize("command", ("calc", "analyse"))
def test_cli_model_option_defaults(pytest_args, command):
    """
    Test model options are not passed to pytest if they are not set.

    Parameters
    ----------
    pytest_args
        Fixture recording the options passed to `pytest.main`.
    command
        CLI command to run.
    """
    result = runner.invoke(app, [command])
    assert result.exit_code == 0, result.output

    for option in ("--models", "--models-file", "--framework"):
        assert option not in pytest_args


def test_analyse_cli_no_calc_options(pytest_args):
    """
    Test `ml_peg analyse` does not pass calculation-only options to pytest.

    Parameters
    ----------
    pytest_args
        Fixture recording the options passed to `pytest.main`.
    """
    run_analyse_cli()

    for option in ("--run-slow", "--run-very-slow", "--run-mock", "--mock-only"):
        assert option not in pytest_args
