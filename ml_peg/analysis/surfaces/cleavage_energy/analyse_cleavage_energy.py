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

        model_pred = []
        model_ref = []

        for xyz_file in sorted(model_dir.glob("*.xyz"), key=lambda p: int(p.stem)):
            slab = read(xyz_file)

            pred_ce = (
                compute_cleavage_energy(
                    slab.info["slab_energy"],
                    slab.info["bulk_energy"],
                    slab.info["thickness_ratio"],
                    slab.info["area_slab"],
                )
                * EV_TO_MEV
            )
            model_pred.append(pred_ce)

            if not ref_stored:
                model_ref.append(slab.info["ref_cleavage_energy"] * EV_TO_MEV)

        if model_pred:
            results[model_name]["pred"] = model_pred
            if not ref_stored:
                stored_ref = model_ref
            results[model_name]["ref"] = stored_ref
            ref_stored = True

    return results

@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_cleavage_energies.json",
    title="Cleavage Energies",
    x_label="Predicted cleavage energy / meV/Å²",
    y_label="Reference cleavage energy / meV/Å²",
)
def cleavage_density(
    cleavage_energies: dict[str, dict[str, list]],
) -> dict[str, dict]:
    """
    Build density scatter inputs and write density trajectories.

    Parameters
    ----------
    cleavage_energies
        Reference and predicted cleavage energies per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model names to density-plot payloads.
    """
    label_list = [
        f.stem
        for f in sorted(
            (CALC_PATH / MODELS[0]).glob("*.xyz"), key=lambda p: int(p.stem)
        )
    ]

    density_inputs: dict[str, dict] = {}
    for model_name in MODELS:
        preds = cleavage_energies[model_name]["pred"]
        refs = cleavage_energies[model_name]["ref"]
        density_inputs[model_name] = {
            "ref": refs,
            "pred": preds,
            "meta": {"system_count": len([v for v in preds if v is not None])},
        }
        if preds:
            write_density_trajectories(
                labels_list=label_list,
                ref_vals=refs,
                pred_vals=preds,
                struct_dir=CALC_PATH / model_name,
                traj_dir=OUT_PATH / model_name / "density_traj",
                struct_filename_builder=lambda label: f"{label}.xyz",
            )
    return density_inputs
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
        ref_vals = cleavage_energies[model_name]["ref"]
        pred_vals = cleavage_energies[model_name]["pred"]
        if pred_vals:
            results[model_name] = mae(ref_vals, pred_vals)
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "cleavage_energy_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
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
