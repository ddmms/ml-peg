"""Obtain the input structure index from the --system option."""

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
    Return the zero-based index into the sorted input structure files.

    Parameters
    ----------
    request
        Pytest fixture request.

    Returns
    -------
    int
        Requested index into the alphabetically sorted XYZ filenames.
    """
    return request.config.getoption("--system")
