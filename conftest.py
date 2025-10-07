"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

import pytest

from ml_peg.models import models


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
        default=None,
        help="MLIPs, in comma-separated list. Default: None.",
    )


def pytest_configure(config):
    """Configure pytest to custom markers and CLI inputs."""
    # Create custom marker for slow tests
    config.addinivalue_line("markers", "slow: mark test as slow calculations")

    # Set current models from CLI input
    models.current_models = config.getoption("--models")


def pytest_collection_modifyitems(config, items):
    """Skip tests if marker applied to unit tests."""
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
