"""Obtain the system_id input argument."""

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
    parser.addoption("--system-id", action="store", default=0, type=int)


@pytest.fixture
def system_id(request):
    """
    Get system_id argument.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--system-id")
