"""Obtain the composition and run_id input arguments."""

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
    parser.addoption("--composition", action="store", default="", type=str)
    parser.addoption("--run-id", action="store", default=-1, type=int)


@pytest.fixture
def composition(request):
    """
    Get composition argument.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--composition")


@pytest.fixture
def run_id(request):
    """
    Get run_id argument.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--run-id")
