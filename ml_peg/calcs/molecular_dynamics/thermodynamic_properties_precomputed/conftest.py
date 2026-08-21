"""Obtain the cas input argument."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """
    Add pytest option.

    Parameters
    ----------
    parser
        Parser to use.
    """
    parser.addoption("--cas", action="store", default=None, type=str)
    parser.addoption("--list-cas", action="store_true", default=False)


@pytest.fixture
def cas(request):
    """
    Get cas argument.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option

        Requested command line argument.
    """
    return request.config.getoption("--cas")


@pytest.fixture
def list_cas(request) -> bool:
    """
    Get the list-cas flag.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--list-cas")
