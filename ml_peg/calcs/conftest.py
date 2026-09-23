"""Configure pytest for calculations."""

from __future__ import annotations

from pytest import Config, Item, Parser, mark

from ml_peg import models


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


def pytest_configure(config: Config) -> None:
    """
    Configure pytest to custom CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    # Create custom markers for slow tests
    config.addinivalue_line("markers", "slow: mark test as slow calculations")
    config.addinivalue_line("markers", "very_slow: mark test as very slow calculations")

    # Set mock options from CLI input
    models.run_mock = config.getoption("--run-mock")
    models.mock_only = config.getoption("--mock-only")


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
    for item in items:
        if "very_slow" in item.keywords and not config.getoption("--run-very-slow"):
            item.add_marker(skip_very_slow)
        elif "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
