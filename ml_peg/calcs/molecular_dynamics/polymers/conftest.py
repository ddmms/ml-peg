"""Pytest CLI options for the polymers benchmark."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register the polymer-benchmark CLI options.

    Note that ``--models`` (used to restrict to a subset of MLIPs from
    ``models.yml``) is registered by the root ``conftest.py``, not here.

    Parameters
    ----------
    parser
        Pytest argument parser injected by the framework.
    """
    parser.addoption("--poly-id", action="store", required=True, type=str)
    parser.addoption("--time-prefactor", action="store", default=1.0, type=float)


@pytest.fixture
def poly_id(request: pytest.FixtureRequest) -> str:
    """
    Return the ``--poly-id`` value (a polymer identifier from data.csv).

    Parameters
    ----------
    request
        Pytest fixture request injected by the framework.

    Returns
    -------
    str
        The value passed to ``--poly-id`` on the command line.
    """
    return str(request.config.getoption("--poly-id"))


@pytest.fixture
def time_prefactor(request: pytest.FixtureRequest) -> float:
    """
    Return the ``--time-prefactor`` value (multiplier on all stage durations).

    Parameters
    ----------
    request
        Pytest fixture request injected by the framework.

    Returns
    -------
    float
        The value passed to ``--time-prefactor`` on the command line.
    """
    return float(request.config.getoption("--time-prefactor"))
