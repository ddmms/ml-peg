"""Analyse the battery electrolyte densities benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import Trajectory, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

# Must match LOG_INTERVAL * TIMESTEP in calc_battery_electrolyte_densities.py.
LOG_INTERVAL_PS = 0.1

# Portion of each trajectory discarded before averaging, leaving 100 ps of
# production. The inputs are already NPT-equilibrated, so this only needs to
# cover relaxation into the new potential energy surface. Protocol from
# ddmms/ml-peg#358.
EQUILIB_TIME_PS = 50

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
        Average density in g/cm^3, or NaN if the trajectory is shorter than the
        discarded equilibration period.
    """
    au_to_g_cm3 = 1e24 / units.mol
    skip_frames = int(EQUILIB_TIME_PS / LOG_INTERVAL_PS)

    traj = Trajectory(traj_path)
    if len(traj) <= skip_frames:
        return np.nan

    densities = [
        au_to_g_cm3 * atoms.get_masses().sum() / atoms.get_volume()
        for atoms in traj[skip_frames:]
    ]
    return float(np.mean(densities))


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
    x_label="Reference density / g cm<sup>-3</sup>",
    y_label="Predicted density / g cm<sup>-3</sup>",
    hoverdata={
        "Labels": INFO["filenames"],
        "Salt": INFO["salt"],
        "Solvent": INFO["solvent"],
        "Concentration / M": INFO["concentration_M"],
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
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for label in INFO["filenames"]:
            traj_path = CALC_PATH / model_name / f"{label}.traj"
            results[model_name].append(compute_density(traj_path))

            atoms = Trajectory(traj_path)[-1]
            if not ref_stored:
                results["ref"].append(atoms.info["exp_density"])

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)
        ref_stored = True

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


@pytest.fixture
@build_table(
    filename=OUT_PATH / "battery_electrolyte_densities_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    get_mae: dict[str, float],
    get_rmse: dict[str, float],
    get_mape: dict[str, float],
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

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
        "RMSE": get_rmse,
        "MAPE": get_mape,
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
