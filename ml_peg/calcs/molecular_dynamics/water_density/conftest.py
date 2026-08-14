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
    parser.addoption("--temperature-idx", action="store", default=0, type=int)


@pytest.fixture
def temperature_idx(request):
    """
    Get temperature index argument.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--temperature-idx")
