"""Tests for model options passed to the CLI."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from ml_peg import models as models_package
from ml_peg.cli.cli import run_dash_app
from ml_peg.models.get_models import get_model_names

MOCK_MODELS_FILE = Path(__file__).parents[1] / "ml_peg" / "models" / "mock.yml"


@pytest.fixture
def fake_run_app(monkeypatch) -> dict:
    """
    Replace the Dash app with a stub, and restore the configured models.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.

    Returns
    -------
    dict
        Keyword arguments the app was run with.
    """
    monkeypatch.setattr(models_package, "current_models", None, raising=False)
    monkeypatch.setattr(models_package, "models_file", models_package.models_file)

    called: dict = {}
    run_app_module = ModuleType("ml_peg.app.run_app")
    run_app_module.run_app = lambda **kwargs: called.update(kwargs)
    monkeypatch.setitem(sys.modules, "ml_peg.app.run_app", run_app_module)

    return called


def test_app_uses_requested_models_file(fake_run_app):
    """Models and models file given to the app are used to load models."""
    run_dash_app(
        models="mock",
        models_file=MOCK_MODELS_FILE,
        category="*",
        test="*",
        port=8050,
        debug=False,
    )

    assert models_package.current_models == "mock"
    assert models_package.models_file == MOCK_MODELS_FILE
    # Loaded through the module-level default, as the app does.
    assert get_model_names(models_package.current_models) == ["mock"]
    assert fake_run_app == {"category": "*", "test": "*", "port": 8050, "debug": False}


def test_app_keeps_default_models_file(fake_run_app):
    """The default models file is kept when the app is run without one."""
    default_file = models_package.models_file

    run_dash_app(
        models=None, models_file=None, category="*", test="*", port=8050, debug=False
    )

    assert models_package.models_file == default_file
    assert "mace-mp-0a" in get_model_names()
