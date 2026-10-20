"""Analyse the battery electrolyte densities benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase import units
from ase.io import Trajectory, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.molecular_dynamics.battery_electrolyte_densities.protocol import (
    EQUILIB_TIME_PS,
    EXPECTED_FRAMES,
    LOG_INTERVAL_PS,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

BENCHMARK = "battery_electrolyte_densities"
CALC_PATH = CALCS_ROOT / "molecular_dynamics" / BENCHMARK / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / BENCHMARK

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.traj",
    index=0,
    info_keys=["exp_density", "concentration_M", "category", "salt", "solvent"],
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
    include_filenames=True,
)


def compute_density(traj_path: Path) -> float:
    """
    Compute the average density over the production part of an NPT trajectory.

    Densities are recomputed from the stored cells rather than parsed out of the
    text log, so the result does not depend on the log format and a truncated
    log cannot silently change the answer.

    Parameters
    ----------
    traj_path
        Path to the ASE trajectory file.

    Returns
    -------
    float
        Average density in g/cm^3, or NaN for incomplete, failed, missing or
        unreadable trajectories, or non-finite saved positions, energies or forces.
    """
    au_to_g_cm3 = 1e24 / units.mol
    skip_frames = round(EQUILIB_TIME_PS / LOG_INTERVAL_PS)
    status_path = traj_path.with_suffix(".status.json")
    try:
        if status_path.exists() and not json.loads(status_path.read_text())["complete"]:
            return np.nan
        with Trajectory(traj_path) as traj:
            if len(traj) != EXPECTED_FRAMES:
                return np.nan
            densities = []
            for index, atoms in enumerate(traj):
                volume = atoms.get_volume()
                if (
                    not np.isfinite(volume)
                    or volume <= 0
                    or not np.isfinite(atoms.positions).all()
                    or not np.isfinite(atoms.get_potential_energy())
                    or not np.isfinite(atoms.get_forces()).all()
                ):
                    return np.nan
                if index > skip_frames:
                    densities.append(au_to_g_cm3 * atoms.get_masses().sum() / volume)
        return float(np.mean(densities))
    except Exception as exc:
        warn(f"Cannot score {traj_path}: {exc}", stacklevel=2)
        return np.nan


def mape(ref: list, prediction: list) -> float:
    """
    Get mean absolute percentage error.

    ML-PEG's analysis utilities provide MAE and RMSE but not MAPE, so it is
    defined here. NaN handling matches those helpers: any NaN prediction makes
    the whole metric NaN, so a failed system cannot be hidden by averaging.

    Parameters
    ----------
    ref
        Reference data.
    prediction
        Predicted data.

    Returns
    -------
    float
        Mean absolute percentage error, in percent.
    """
    if np.isnan(np.sum(prediction)):
        return np.nan

    ref_arr = np.asarray(ref, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    return float(100 * np.mean(np.abs((ref_arr - pred_arr) / ref_arr)))


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_battery_electrolyte_densities.json",
    title="Battery electrolyte densities",
    x_label="Predicted density / g cm<sup>-3</sup>",
    y_label="Reference density / g cm<sup>-3</sup>",
    hoverdata={
        "Labels": INFO["filenames"],
        "Salt": INFO["salt"],
        "Solvent": INFO["solvent"],
        "Starting concentration / M": INFO["concentration_M"],
    },
    symbol_by=INFO["category"],
)
def densities() -> dict[str, list]:
    """
    Get battery electrolyte densities for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted densities.
    """
    results = {"ref": list(INFO["exp_density"])} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        for label in INFO["filenames"]:
            traj_path = CALC_PATH / model_name / f"{label}.traj"
            density = compute_density(traj_path)
            if np.isfinite(density):
                try:
                    with Trajectory(traj_path) as traj:
                        atoms = traj[-1]
                except Exception as exc:
                    warn(f"Cannot read final frame of {traj_path}: {exc}", stacklevel=2)
                    results[model_name].append(np.nan)
                    continue
                structs_dir = OUT_PATH / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / f"{label}.xyz", atoms)
            results[model_name].append(density)

    return results


@pytest.fixture
def get_mae(densities: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for densities.

    Parameters
    ----------
    densities
        Dictionary of reference and predicted densities.

    Returns
    -------
    dict[str, float]
        Dictionary of density errors for all models.
    """
    return {
        model_name: mae(densities["ref"], densities[model_name])
        for model_name in MODELS
    }


@pytest.fixture
def get_rmse(densities: dict[str, list]) -> dict[str, float]:
    """
    Get root mean squared error for densities.

    Parameters
    ----------
    densities
        Dictionary of reference and predicted densities.

    Returns
    -------
    dict[str, float]
        Dictionary of density errors for all models.
    """
    return {
        model_name: rmse(densities["ref"], densities[model_name])
        for model_name in MODELS
    }


@pytest.fixture
def get_mape(densities: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute percentage error for densities.

    Parameters
    ----------
    densities
        Dictionary of reference and predicted densities.

    Returns
    -------
    dict[str, float]
        Dictionary of density errors for all models.
    """
    return {
        model_name: mape(densities["ref"], densities[model_name])
        for model_name in MODELS
    }


@plot_scatter(
    filename=OUT_PATH / "figure_completion.json",
    title="Complete electrolyte runs",
    x_label="System",
    y_label="Completed / %",
    hovertemplate="System: %{x}<br>Completed: %{y}%<extra>%{fullData.name}</extra>",
)
def completion_plot(densities: dict[str, list]) -> dict[str, list]:
    """
    Plot complete (100) and incomplete or invalid (0) systems for each model.

    Parameters
    ----------
    densities
        Reference and predicted densities, aligned to the expected systems.

    Returns
    -------
    dict[str, list]
        System labels and binary completion percentages for models with output.
    """
    return {
        model_name: [INFO["filenames"], (100 * np.isfinite(values)).tolist()]
        for model_name, values in densities.items()
        if model_name != "ref"
        and any(
            (CALC_PATH / model_name / f"{label}.traj").exists()
            for label in INFO["filenames"]
        )
    }


@pytest.fixture
def stable_runs(densities: dict[str, list]) -> dict[str, float]:
    """
    Return the percentage of expected systems with complete, finite trajectories.

    Parameters
    ----------
    densities
        Reference and predicted densities, with invalid runs represented by NaN.

    Returns
    -------
    dict[str, float]
        Completion percentages, or NaN for models without trajectory output.
    """
    completion_plot(densities)
    return {
        model_name: (
            100.0 * np.isfinite(densities[model_name]).sum() / len(INFO["filenames"])
            if any(
                (CALC_PATH / model_name / f"{label}.traj").exists()
                for label in INFO["filenames"]
            )
            else np.nan
        )
        for model_name in MODELS
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "battery_electrolyte_densities_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    get_mae: dict[str, float],
    get_rmse: dict[str, float],
    get_mape: dict[str, float],
    stable_runs: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.
    get_rmse
        Root mean squared errors for all models.
    get_mape
        Mean absolute percentage errors for all models.
    stable_runs
        Percentage of complete, finite trajectories for each model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
        "RMSE": get_rmse,
        "MAPE": get_mape,
        "Stable runs": stable_runs,
    }


@pytest.mark.framework("omol25-electrolytes")
def test_battery_electrolyte_densities(metrics: dict[str, dict]) -> None:
    """
    Run battery electrolyte densities test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
