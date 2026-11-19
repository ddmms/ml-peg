"""Pytest configuration and fixtures for the HPHT CH4/H2O benchmark."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """
    Add custom pytest command-line options.

    Parameters
    ----------
    parser : pytest.Parser
        Pytest parser used to register command-line options.
    """
    try:
        parser.addoption(
            "--structure",
            action="store",
            default=None,
        )
    except ValueError:
        pass


@pytest.fixture
def structure(request):
    """
    Retrieve the structure specified via the pytest command line.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request providing access to command-line options.

    Returns
    -------
    str or None
        Name of the selected structure, or None if not specified.
    """
    return request.config.getoption("--structure")
