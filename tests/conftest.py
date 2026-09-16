"""Configure tests."""

from __future__ import annotations

import pytest

from ml_peg import analysis, models
from ml_peg.analysis.utils import decorators


@pytest.fixture
def fake_models(monkeypatch):
    """
    Replace the model registry with two fake models.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    """

    def get_model_names(models=None, filepath=None):
        """
        Get fake model names.

        Parameters
        ----------
        models
            Models to select, as a comma-separated list. Default is `None`,
            corresponding to all models.
        filepath
            Unused path to model definitions.

        Returns
        -------
        list[str]
            Selected model names.
        """
        return ["model_1", "model_2"] if models is None else models.split(",")

    monkeypatch.setattr(decorators, "get_model_names", get_model_names)
    monkeypatch.setattr(
        decorators,
        "load_model_configs",
        lambda mlips, filepath=None: (
            {mlip: {} for mlip in mlips},
            dict.fromkeys(mlips),
        ),
    )
    monkeypatch.setattr(models, "current_models", None)
    monkeypatch.setattr(analysis, "update_results", False)


@pytest.fixture
def update_model_2(monkeypatch, fake_models):
    """
    Set up an update run, analysing only `model_2`.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    fake_models
        Fixture replacing the model registry with two fake models.
    """
    monkeypatch.setattr(models, "current_models", "model_2")
    monkeypatch.setattr(analysis, "update_results", True)
