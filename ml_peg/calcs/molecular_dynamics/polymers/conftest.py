"""Pytest CLI options for the polymers benchmark."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register the ``--poly-id`` and ``--time-prefactor`` CLI options.

    Parameters
    ----------
    parser
        Pytest argument parser injected by the framework.
    """
    parser.addoption("--poly-id", action="store", required=True, type=str)
    parser.addoption("--time-prefactor", action="store", default=1.0, type=float)


@pytest.fixture
def poly_id(request: pytest.FixtureRequest) -> str:
    """Return the ``--poly-id`` value (a polymer identifier from data.csv)."""
    return str(request.config.getoption("--poly-id"))


@pytest.fixture
def time_prefactor(request: pytest.FixtureRequest) -> float:
    """Return the ``--time-prefactor`` value (multiplier on all stage durations)."""
    return float(request.config.getoption("--time-prefactor"))
