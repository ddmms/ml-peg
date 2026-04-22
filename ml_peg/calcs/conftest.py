"""Configure pytest for calculations."""

from __future__ import annotations

from pytest import Config, Parser

from ml_peg.models import get_models


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


def pytest_configure(config: Config) -> None:
    """
    Configure pytest to custom CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    # Set current models from CLI input
    get_models.include_mock = config.getoption("--run-mock")
