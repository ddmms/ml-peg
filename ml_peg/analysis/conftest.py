"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

import pytest

from ml_peg import analysis


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add custom flags.

    Parameters
    ----------
    parser
        Parser to add command line options to.
    """
    parser.addoption(
        "--update",
        action="store_true",
        default=False,
        help=(
            "Update saved tables and plots in place, preserving results for models "
            "not included in --models, rather than overwriting them."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest to custom markers and CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration, with command line options set.
    """
    # Set whether saved results are updated, rather than overwritten
    analysis.update_results = config.getoption("--update")
