"""Command line options for the binary mixtures benchmark."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """
    Add pytest options.

    Parameters
    ----------
    parser
        Parser to use.
    """
    parser.addoption("--box-id", action="store", default=0, type=int)
    # None: the protocol length set in calc_binary_mixtures.py
    parser.addoption("--run-ps", action="store", default=None, type=float)


@pytest.fixture
def box_id(request):
    """
    Get the index of the box to simulate.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    int
        Index into the sorted list of benchmark boxes.
    """
    return request.config.getoption("--box-id")


@pytest.fixture
def run_ps(request):
    """
    Get the total NPT length (equilibration plus production) in ps.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    float | None
        Run length in ps, or ``None`` for the benchmark protocol.
    """
    return request.config.getoption("--run-ps")
