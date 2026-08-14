"""Obtain the run_id input argument."""

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
    parser.addoption("--run-id", action="store", default=-1, type=int)


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
