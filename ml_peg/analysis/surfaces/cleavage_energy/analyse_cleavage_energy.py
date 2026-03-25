"""Analyse cleavage energy benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    load_metrics_config,
    mae,
    write_density_trajectories,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "surfaces" / "cleavage_energy" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "cleavage_energy"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_MEV = 1000.0


def compute_cleavage_energy(
    slab_energy: float,
    bulk_energy: float,
    thickness_ratio: float,
    area_slab: float,
) -> float:
    """
    Compute cleavage energy from slab and bulk energies.

    Parameters
    ----------
    slab_energy
        Total energy of the slab.
    bulk_energy
        Total energy of the lattice-matched bulk unit cell.
    thickness_ratio
        Number of bulk unit cells in the slab thickness.
    area_slab
        Surface area of the slab in Angstrom^2.

    Returns
    -------
    float
        Cleavage energy in eV/Angstrom^2.
    """
    return (slab_energy - thickness_ratio * bulk_energy) / (2 * area_slab)


@pytest.fixture
def cleavage_energies() -> dict[str, dict[str, list]]:
    """
    Get cleavage energies for all systems in meV/A^2.

    Also saves per-system errors for the distribution plot.

    Returns
    -------
    dict[str, dict[str, list]]
        Dictionary of model names to ``{"ref": [...], "pred": [...]}`` in meV/A^2.
    """
    results = {mlip: {"ref": [], "pred": []} for mlip in MODELS}
    ref_stored = False
    stored_ref = []

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        if canonical_ids is None:
            canonical_ids = sorted(data.keys())
            for uid in canonical_ids:
                entry = data[uid]
                results["ref"].append(entry["ref_cleavage_energy"] * EV_TO_MEV)

        for uid in canonical_ids:
            entry = data[uid]
            pred_ce = (
                compute_cleavage_energy(
                    entry["slab_energy"],
                    entry["bulk_energy"],
                    entry["thickness_ratio"],
                    entry["area_slab"],
                )
                * EV_TO_MEV
            )
            results[model_name].append(pred_ce)

    return results


@pytest.fixture
def cleavage_mae(cleavage_energies: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for cleavage energies.

    Parameters
    ----------
    cleavage_energies
        Dictionary of reference and predicted cleavage energies.

    Returns
    -------
    dict[str, float]
        MAE for each model.
    """
    results = {}
    for model_name in MODELS:
        if cleavage_energies[model_name]:
            results[model_name] = mae(
                cleavage_energies["ref"], cleavage_energies[model_name]
            )
        else:
            results[model_name] = None
    return results


@pytest.fixture
def cleavage_rmse(cleavage_energies: dict[str, list]) -> dict[str, float]:
    """
    Get root mean squared error for cleavage energies.

    Parameters
    ----------
    cleavage_energies
        Dictionary of reference and predicted cleavage energies.

    Returns
    -------
    dict[str, float]
        RMSE for each model.
    """
    results = {}
    for model_name in MODELS:
        if cleavage_energies[model_name]:
            ref = np.array(cleavage_energies["ref"])
            pred = np.array(cleavage_energies[model_name])
            results[model_name] = float(np.sqrt(np.mean((pred - ref) ** 2)))
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "cleavage_energy_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    cleavage_mae: dict[str, float],
    cleavage_rmse: dict[str, float],
) -> dict[str, dict]:
    """
    Get all cleavage energy metrics.

    Parameters
    ----------
    cleavage_mae
        Mean absolute errors for all models.
    cleavage_rmse
        Root mean squared errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": cleavage_mae,
        "RMSE": cleavage_rmse,
    }


def test_cleavage_energy(metrics: dict[str, dict]) -> None:
    """
    Run cleavage energy analysis.

    Parameters
    ----------
    metrics
        All cleavage energy metrics.
    cleavage_density
        Density-scatter inputs for all models (drives saved plots).
    """
    return
