"""Obtain the system_id input argument."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """
    Add the crystal-selection option.

    Parameters
    ----------
    parser
        Pytest argument parser.
    """
    try:
        parser.addoption("--system", action="store", default=0, type=int)
    except ValueError:
        pass


@pytest.fixture
def system(request):
    """
    Return the selected crystal name.

    Parameters
    ----------
    request
        Pytest fixture request.

    Returns
    -------
    int
        Requested crystal name.
    """
    return request.config.getoption("--system")
