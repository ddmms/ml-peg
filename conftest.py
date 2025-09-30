"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """Add flag to run tests for extra MLIPs."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow benchmarks",
    )
    parser.addoption(
        "--models",
        action="store",
        default="all",
        help="MLIPs, in comma-separated list. Default: 'all'.",
    )


def pytest_configure(config):
    """Configure pytest to include marker for extra MLIPs."""
    config.addinivalue_line("markers", "slow: mark test as slow calculations")


def pytest_collection_modifyitems(config, items):
    """Skip tests if marker applied to unit tests."""
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def models(request):
    """Get models option."""
    return request.config.getoption("--models")
