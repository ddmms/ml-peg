"""Obtain the density_index input argument."""

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
    parser.addoption("--density-index", action="store", default=0, type=int)


@pytest.fixture
def density_index(request):
    """
    Get density_index argument.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--density-index")
