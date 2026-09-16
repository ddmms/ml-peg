"""Test building and updating tables."""

from __future__ import annotations

import json

import pytest

from ml_peg import models
from ml_peg.analysis.utils.decorators import build_table

pytestmark = pytest.mark.usefixtures("fake_models")

THRESHOLDS = {"MAE": {"good": 0.0, "bad": 10.0, "unit": "eV"}}


def build(filename, results, mlip_name_map=None):
    """
    Build a table from pre-computed results.

    Parameters
    ----------
    filename
        Filename to save table.
    results
        Metric values for each model.
    mlip_name_map
        Optional mapping of model identifier to display name.
    """

    @build_table(thresholds=THRESHOLDS, filename=filename, mlip_name_map=mlip_name_map)
    def metrics():
        """
        Get metrics.

        Returns
        -------
        dict
            Metric values for each model.
        """
        return results

    metrics()


def get_values(filename):
    """
    Get saved metric values, keyed by model.

    Parameters
    ----------
    filename
        Filename of the saved table.

    Returns
    -------
    dict[str, float | None]
        Saved MAE for each model.
    """
    with open(filename) as fp:
        return {row["id"]: row["MAE"] for row in json.load(fp)["data"]}


def test_build_table(tmp_path):
    """
    Test all models are written to a new table.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved table.
    """
    filename = tmp_path / "table.json"
    build(filename, {"MAE": {"model_1": 1.0, "model_2": 2.0}})

    assert get_values(filename) == {"model_1": 1.0, "model_2": 2.0}


def test_overwrite_table(tmp_path, monkeypatch):
    """
    Test rerunning a subset of models without `--update` clears other models.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved table.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    filename = tmp_path / "table.json"
    build(filename, {"MAE": {"model_1": 1.0, "model_2": 2.0}})

    monkeypatch.setattr(models, "current_models", "model_2")
    build(filename, {"MAE": {"model_2": 5.0}})

    assert get_values(filename) == {"model_1": None, "model_2": 5.0}


def test_update_table(tmp_path, update_model_2):
    """
    Test updating a table preserves values for models not being analysed.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved table.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "table.json"
    build(filename, {"MAE": {"model_1": 1.0, "model_2": 2.0}})

    assert get_values(filename) == {"model_1": 1.0, "model_2": 2.0}

    build(filename, {"MAE": {"model_2": 5.0}})

    assert get_values(filename) == {"model_1": 1.0, "model_2": 5.0}


def test_update_table_display_names(tmp_path, update_model_2):
    """
    Test display names of preserved models are set by the current run.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved table.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "table.json"
    name_map = {"model_1": "model_1-D3", "model_2": "model_2-D3"}

    build(filename, {"MAE": {"model_1": 1.0, "model_2": 2.0}}, name_map)
    build(filename, {"MAE": {"model_2": 5.0}}, name_map)

    with open(filename) as fp:
        rows = json.load(fp)["data"]

    # Display names must be applied to all models, not only those being analysed
    assert {row["id"]: row["MLIP"] for row in rows} == name_map
    assert get_values(filename) == {"model_1": 1.0, "model_2": 5.0}


def test_update_new_table(tmp_path, update_model_2):
    """
    Test updating a table that has not previously been saved.

    Parameters
    ----------
    tmp_path
        Temporary directory for the saved table.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    filename = tmp_path / "table.json"

    build(filename, {"MAE": {"model_2": 5.0}})

    assert get_values(filename) == {"model_1": None, "model_2": 5.0}
