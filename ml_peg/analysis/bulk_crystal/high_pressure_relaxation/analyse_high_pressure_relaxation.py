"""Analyse high-pressure crystal relaxation benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from ase.io import write as ase_write
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    build_density_inputs,
    load_metrics_config,
    mae,
    write_density_trajectories,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "high_pressure_relaxation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "high_pressure_relaxation"

# Pressure conditions
PRESSURES = [0, 25, 50, 75, 100, 125, 150]
PRESSURE_LABELS = ["P000", "P025", "P050", "P075", "P100", "P125", "P150"]

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Energy outlier thresholds (eV/atom).  Converged structures whose predicted
# energy per atom falls outside this range are treated as unconverged for both
# the parity metrics and the convergence rate.
ENERGY_OUTLIER_MIN = -25
ENERGY_OUTLIER_MAX = 25


def get_converged_data_for_pressure(
    model_name: str, pressure_label: str
) -> dict[str, list]:
    """
    Get converged volume and energy data for a model at a specific pressure.

    Reads individual per-structure xyz files and copies them to the app data
    directory. Filters energy outliers.

    Parameters
    ----------
    model_name
        Name of the model.
    pressure_label
        Pressure label (e.g., "P000").

    Returns
    -------
    dict[str, list]
        Labels (mat_ids) and lists of ref/pred volumes and energies.
    """
    xyz_dir = CALC_PATH / model_name / pressure_label
    if not xyz_dir.exists():
        return {
            "labels": [],
            "ref_vol": [],
            "pred_vol": [],
            "ref_energy": [],
            "pred_energy": [],
        }

    app_dir = OUT_PATH / model_name / pressure_label
    app_dir.mkdir(parents=True, exist_ok=True)

    labels, ref_vol, pred_vol, ref_energy, pred_energy = [], [], [], [], []
    for xyz_file in sorted(xyz_dir.glob("*.xyz")):
        atoms = ase_read(xyz_file)
        pred_e = atoms.info.get("pred_energy_per_atom")
        if (
            pred_e is None
            or pred_e <= ENERGY_OUTLIER_MIN
            or pred_e >= ENERGY_OUTLIER_MAX
        ):
            continue
        labels.append(xyz_file.stem)
        ref_vol.append(atoms.info["ref_volume_per_atom"])
        pred_vol.append(atoms.info["pred_volume_per_atom"])
        ref_energy.append(atoms.info["ref_energy_per_atom"])
        pred_energy.append(pred_e)
        # Copy structure to app data directory
        ase_write(app_dir / xyz_file.name, atoms)

    return {
        "labels": labels,
        "ref_vol": ref_vol,
        "pred_vol": pred_vol,
        "ref_energy": ref_energy,
        "pred_energy": pred_energy,
    }


@pytest.fixture
def high_pressure_stats() -> dict[str, dict[str, Any]]:
    """
    Load and aggregate converged volume/energy data across all pressures per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model dicts with pooled "labels", "volume" and "energy" ref/pred lists.
    """
    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        labels, ref_vol, pred_vol, ref_energy, pred_energy = [], [], [], [], []
        for pressure_label in PRESSURE_LABELS:
            data = get_converged_data_for_pressure(model_name, pressure_label)
            # Use "{pressure_label}/{mat_id}" as compound label so struct paths resolve
            labels.extend(f"{pressure_label}/{mat_id}" for mat_id in data["labels"])
            ref_vol.extend(data["ref_vol"])
            pred_vol.extend(data["pred_vol"])
            ref_energy.extend(data["ref_energy"])
            pred_energy.extend(data["pred_energy"])
        stats[model_name] = {
            "labels": labels,
            "volume": {"ref": ref_vol, "pred": pred_vol},
            "energy": {"ref": ref_energy, "pred": pred_energy},
        }
    return stats


def get_convergence_rate_for_pressure(
    model_name: str, pressure_label: str
) -> float | None:
    """
    Get convergence rate for a model at a specific pressure.

    Structures that converged but have predicted energies outside the
    outlier range (``ENERGY_OUTLIER_MIN`` to ``ENERGY_OUTLIER_MAX`` eV/atom)
    are counted as unconverged.

    Parameters
    ----------
    model_name
        Name of the model.
    pressure_label
        Pressure label (e.g., "P000").

    Returns
    -------
    float | None
        Convergence rate (%) or None if no data.
    """
    csv_path = CALC_PATH / model_name / f"results_{pressure_label}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    n_total = len(df)
    n_converged = int(df["converged"].sum())

    # Subtract converged structures whose energy is an outlier
    xyz_dir = CALC_PATH / model_name / pressure_label
    if xyz_dir.is_dir():
        for xyz_file in xyz_dir.glob("*.xyz"):
            atoms = ase_read(xyz_file)
            pred_e = atoms.info.get("pred_energy_per_atom")
            if pred_e is not None and (
                pred_e <= ENERGY_OUTLIER_MIN or pred_e >= ENERGY_OUTLIER_MAX
            ):
                n_converged -= 1

    return (n_converged / n_total) * 100


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_volume_density.json",
    title="Volume per atom (all pressures)",
    x_label="Reference volume / Å³/atom",
    y_label="Predicted volume / Å³/atom",
)
def volume_density(
    high_pressure_stats: dict[str, dict[str, Any]],
) -> dict[str, dict]:
    """
    Density scatter inputs for volume per atom (pooled across all pressures).

    Parameters
    ----------
    high_pressure_stats
        Aggregated volume/energy data per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    for model_name in MODELS:
        write_density_trajectories(
            labels_list=high_pressure_stats[model_name]["labels"],
            ref_vals=high_pressure_stats[model_name]["volume"]["ref"],
            pred_vals=high_pressure_stats[model_name]["volume"]["pred"],
            struct_dir=OUT_PATH / model_name,
            traj_dir=OUT_PATH / model_name / "density_traj_volume",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return build_density_inputs(MODELS, high_pressure_stats, "volume", metric_fn=mae)


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_energy_density.json",
    title="Energy per atom (all pressures)",
    x_label="Reference energy / eV/atom",
    y_label="Predicted energy / eV/atom",
)
def energy_density(
    high_pressure_stats: dict[str, dict[str, Any]],
) -> dict[str, dict]:
    """
    Density scatter inputs for energy per atom (pooled across all pressures).

    Parameters
    ----------
    high_pressure_stats
        Aggregated volume/energy data per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    for model_name in MODELS:
        write_density_trajectories(
            labels_list=high_pressure_stats[model_name]["labels"],
            ref_vals=high_pressure_stats[model_name]["energy"]["ref"],
            pred_vals=high_pressure_stats[model_name]["energy"]["pred"],
            struct_dir=OUT_PATH / model_name,
            traj_dir=OUT_PATH / model_name / "density_traj_energy",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return build_density_inputs(MODELS, high_pressure_stats, "energy", metric_fn=mae)


@pytest.fixture
def volume_mae_per_pressure() -> dict[str, dict[str, float]]:
    """
    Calculate MAE for volume predictions at each pressure.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {pressure_gpa: {model_name: mae_value}}.
    """
    results = {}
    for pressure, pressure_label in zip(PRESSURES, PRESSURE_LABELS, strict=False):
        pressure_key = f"Volume MAE ({pressure} GPa)"
        results[pressure_key] = {}
        for model_name in MODELS:
            data = get_converged_data_for_pressure(model_name, pressure_label)
            if data["ref_vol"] and data["pred_vol"]:
                results[pressure_key][model_name] = mae(
                    data["ref_vol"], data["pred_vol"]
                )
    return results


@pytest.fixture
def energy_mae_per_pressure() -> dict[str, dict[str, float]]:
    """
    Calculate MAE for energy predictions at each pressure.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {pressure_gpa: {model_name: mae_value}}.
    """
    results = {}
    for pressure, pressure_label in zip(PRESSURES, PRESSURE_LABELS, strict=False):
        pressure_key = f"Energy MAE ({pressure} GPa)"
        results[pressure_key] = {}
        for model_name in MODELS:
            data = get_converged_data_for_pressure(model_name, pressure_label)
            if data["ref_energy"] and data["pred_energy"]:
                results[pressure_key][model_name] = mae(
                    data["ref_energy"], data["pred_energy"]
                )
    return results


@pytest.fixture
def convergence_per_pressure() -> dict[str, dict[str, float]]:
    """
    Calculate convergence rate at each pressure.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {pressure_gpa: {model_name: convergence_rate}}.
    """
    results = {}
    for pressure, pressure_label in zip(PRESSURES, PRESSURE_LABELS, strict=False):
        pressure_key = f"Convergence ({pressure} GPa)"
        results[pressure_key] = {}
        for model_name in MODELS:
            conv_rate = get_convergence_rate_for_pressure(model_name, pressure_label)
            if conv_rate is not None:
                results[pressure_key][model_name] = conv_rate
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "high_pressure_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    volume_mae_per_pressure: dict[str, dict[str, float]],
    energy_mae_per_pressure: dict[str, dict[str, float]],
    convergence_per_pressure: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    Get all high-pressure relaxation metrics separated by pressure.

    Parameters
    ----------
    volume_mae_per_pressure
        Volume MAE for all models at each pressure.
    energy_mae_per_pressure
        Energy MAE for all models at each pressure.
    convergence_per_pressure
        Convergence rate for all models at each pressure.

    Returns
    -------
    dict[str, dict]
        All metrics for all models.
    """
    all_metrics = {}
    all_metrics.update(volume_mae_per_pressure)
    all_metrics.update(energy_mae_per_pressure)
    all_metrics.update(convergence_per_pressure)
    return all_metrics


def test_high_pressure_relaxation(
    metrics: dict[str, dict],
    volume_density: dict[str, dict],
    energy_density: dict[str, dict],
) -> None:
    """
    Run high-pressure relaxation analysis test.

    Parameters
    ----------
    metrics
        All high-pressure relaxation metrics.
    volume_density
        Triggers volume density plot generation.
    energy_density
        Triggers energy density plot generation.
    """
    return
