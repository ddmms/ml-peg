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
    parser.addoption("--cas", action="store", default="67-64-1", type=str)
    parser.addoption("--total-md-steps", action="store", default=2_000_000, type=int)
    parser.addoption("--traj-interval", action="store", default=1000, type=int)
    parser.addoption("--log-interval", action="store", default=1, type=int)


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
def total_md_steps(request):
    """
    Get number of md steps.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--total-md-steps")


@pytest.fixture
def traj_interval(request):
    """
    Get trajectory dumping interval.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--traj-interval")


@pytest.fixture
def log_interval(request):
    """
    Get log dumping interval.

    Parameters
    ----------
    request
        Request.

    Returns
    -------
    option
        Requested command line argument.
    """
    return request.config.getoption("--log-interval")
