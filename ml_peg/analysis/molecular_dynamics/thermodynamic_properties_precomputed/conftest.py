"""Configure thermodynamic properties analysis tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """
    Add command-line options for thermodynamic properties analysis.

    Parameters
    ----------
    parser
        Pytest command-line option parser.
    """
    parser.addoption("--block-size", action="store", default=100, type=int)
    parser.addoption("--skip-time-ps", action="store", default=0.0, type=float)
    parser.addoption("--detailed-results", action="store_true", default=False)


@pytest.fixture
def block_size(request) -> int:
    """
    Return the block size used for statistical analysis.

    Parameters
    ----------
    request
        The request.

    Returns
    -------
    int
        The block size.
    """
    return request.config.getoption("--block-size")


@pytest.fixture
def skip_time_ps(request) -> float:
    """
    Return the skipped time in ps.

    Parameters
    ----------
    request
        The request.

    Returns
    -------
    float
        The equilibration time in ps.
    """
    return request.config.getoption("--skip-time-ps")


@pytest.fixture
def detailed_results(request) -> bool:
    """
    Return the detailed_results flag.

    Parameters
    ----------
    request
        The request.

    Returns
    -------
    float
        The equilibration time in ps.
    """
    return request.config.getoption("--detailed-results")
