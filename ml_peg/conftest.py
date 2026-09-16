"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

from pytest import Config, Item, Parser

from ml_peg import models


def pytest_addoption(parser: Parser) -> None:
    """
    Add flag to run tests for extra MLIPs.

    Parameters
    ----------
    parser
        Pytest parser object.
    """
    parser.addoption(
        "--models",
        action="store",
        default=None,
        help="MLIPs, in comma-separated list. Default is all models",
    )
    parser.addoption(
        "--models-file",
        action="store",
        default=None,
        help="Filepath to model definitions. Default models.yml in models directory.",
    )
    parser.addoption(
        "--framework",
        action="store",
        default=None,
        help=(
            "Run only tests belonging to these MLIP framework(s), as a "
            "comma-separated list of framework ids. Default is all tests."
        ),
    )


def pytest_configure(config: Config) -> None:
    """
    Configure pytest to custom markers and CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    # Create custom markers
    config.addinivalue_line(
        "markers",
        "framework(*ids): mark test as belonging to MLIP framework(s)",
    )

    # Set current models from CLI input
    models.current_models = config.getoption("--models")
    model_file = config.getoption("--models-file")
    if model_file:
        models.models_file = model_file


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """
    Deselect tests outside the requested framework(s).

    Parameters
    ----------
    config
        Pytest configuration object.
    items
        Collected test items, modified in place to remove deselected tests.
    """
    # Keep only tests tagged with one of the requested frameworks
    framework = config.getoption("--framework")
    if not framework:
        return
    requested = {name.strip() for name in framework.split(",") if name.strip()}
    selected = []
    deselected = []
    for item in items:
        item_frameworks = {
            fw for marker in item.iter_markers(name="framework") for fw in marker.args
        }
        (selected if item_frameworks & requested else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
