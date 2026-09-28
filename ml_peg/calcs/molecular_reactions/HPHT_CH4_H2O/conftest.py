from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """Add pytest options."""
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
    """Get structure command line argument."""
    return request.config.getoption("--structure")
