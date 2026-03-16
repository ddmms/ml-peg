"""Plutonium Dioxide benchmark against DFT+U."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import io, units
import numpy as np
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
CALC_PATH = CALCS_ROOT / "actinides" / "plutonium_dioxide" / "outputs"
OUT_PATH = APP_ROOT / "data" / "actinides" / "plutonium_dioxide"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KJ_PER_MOL = units.mol / units.kJ


@pytest.fixture
def puo2_stats() -> dict[str, dict[str, Any]]:
    """
    Load and cache statistics per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Processed information per model (energy, force, stress, labels).
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, Any]] = {}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        struct_dir = OUT_PATH / model_name
        struct_dir.mkdir(parents=True, exist_ok=True)

        energies_ref, energies_pred = [], []
        forces_ref, forces_pred = [], []
        stress_ref, stress_pred = [], []
        energy_labels: list[str] = []
        force_labels: list[str] = []
        stress_labels: list[str] = []
        excluded = 0
        frame_idx = 0

        for xyz_file in sorted(model_dir.glob("*.xyz")):
            frames = io.read(xyz_file, ":")
            for atoms in frames:
                label = str(frame_idx)
                natoms = atoms.get_number_of_atoms()
                e_ref = atoms.info.get("energy_xtb")
                f_ref = atoms.arrays.get("forces_xtb")
                s_ref = atoms.info.get("REF_stress")

                io.write(struct_dir / f"{label}.xyz", atoms)

                if e_ref is not None:
                    energies_ref.append(e_ref / natoms)
                    energies_pred.append(atoms.get_total_energy() / natoms)
                    energy_labels.append(label)

                if f_ref is not None:
                    forces_ref.append(f_ref.ravel())
                    forces_pred.append(atoms.get_forces().ravel())
                    force_labels.extend([label] * (natoms * 3))

                if s_ref is not None:
                    s_ref_flat = np.asarray(s_ref).ravel().tolist()
                    stress_ref.extend(s_ref_flat)
                    stress_pred.extend(atoms.get_stress(voigt=False).ravel())
                    stress_labels.extend([label] * len(s_ref_flat))

                frame_idx += 1

        stats[model_name] = {
            "energies": {
                "ref": energies_ref,
                "pred": energies_pred,
            },
            "forces": {
                "ref": np.concatenate(forces_ref).tolist() if forces_ref else [],
                "pred": np.concatenate(forces_pred).tolist() if forces_pred else [],
            },
            "stress": {
                "ref": stress_ref,
                "pred": stress_pred,
            },
            "excluded": excluded,
            "energy_labels": energy_labels,
            "force_labels": force_labels,
            "stress_labels": stress_labels,
        }
    return stats


@pytest.fixture
def energy_mae(puo2_stats: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """
    Mean absolute error for energy predictions.

    Parameters
    ----------
    puo2_stats
        Aggregrated energy/force/stress per model.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model (``None`` if no data).
    """
    results: dict[str, float | None] = {}
    for model_name, props in puo2_stats.items():
        energies = props.get("energies", {})
        results[model_name] = mae(energies.get("ref", []), energies.get("pred", []))
    return results


@pytest.fixture
def forces_mae(puo2_stats: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """
    Mean absolute error for force predictions.

    Parameters
    ----------
    puo2_stats
        Aggregrated energy/force/stress per model.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model (``None`` if no data).
    """
    results: dict[str, float | None] = {}
    for model_name, props in puo2_stats.items():
        forces = props.get("forces", {})
        results[model_name] = mae(forces.get("ref", []), forces.get("pred", []))
    return results


@pytest.fixture
def stress_mae(puo2_stats: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """
    Mean absolute error for stress predictions.

    Parameters
    ----------
    puo2_stats
        Aggregrated energy/force/stress per model.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model (``None`` if no data).
    """
    results: dict[str, float | None] = {}
    for model_name, props in puo2_stats.items():
        stress = props.get("stress", {})
        results[model_name] = mae(stress.get("ref", []), stress.get("pred", []))
    return results


# Density plots for each metric.


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_energy_density.json",
    title="Relative Energy Plutonium Dioxide",
    x_label="PBE+U Reference Energy / eV / Atom",
    y_label="Predicted Energy / eV / Atom",
    annotation_metadata={"excluded": "Excluded"},
)
def energy_density(puo2_stats: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Density scatter input for energy.

    Parameters
    ----------
    puo2_stats
        Aggregrated energy/force/stress per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    result = build_density_inputs(
        list(puo2_stats.keys()),
        puo2_stats,
        "energies",
        metric_fn=mae,
    )
    for model_name, model_stats in puo2_stats.items():
        write_density_trajectories(
            labels_list=model_stats["energy_labels"],
            ref_vals=model_stats["energies"]["ref"],
            pred_vals=model_stats["energies"]["pred"],
            struct_dir=OUT_PATH / model_name,
            traj_dir=OUT_PATH / model_name / "density_traj_energy",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return result


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_force_density.json",
    title="Forces Plutonium Dioxide",
    x_label="PBE+U Reference Forces / eV / Å",
    y_label="Predicted Forces / eV / Å",
    annotation_metadata={"excluded": "Excluded"},
)
def force_density(puo2_stats: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Density scatter input for force.

    Parameters
    ----------
    puo2_stats
        Aggregrated energy/force/stress per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    result = build_density_inputs(
        list(puo2_stats.keys()),
        puo2_stats,
        "forces",
        metric_fn=mae,
    )
    for model_name, model_stats in puo2_stats.items():
        write_density_trajectories(
            labels_list=model_stats["force_labels"],
            ref_vals=model_stats["forces"]["ref"],
            pred_vals=model_stats["forces"]["pred"],
            struct_dir=OUT_PATH / model_name,
            traj_dir=OUT_PATH / model_name / "density_traj_force",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return result


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_stress_density.json",
    title="Stress Plutonium Dioxide",
    x_label="PBE+U Reference Stress / eV / Å³",
    y_label="Predicted Stress / eV / Å³",
    annotation_metadata={"excluded": "Excluded"},
)
def stress_density(puo2_stats: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Density scatter input for stress.

    Parameters
    ----------
    puo2_stats
        Aggregrated energy/force/stress per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    result = build_density_inputs(
        list(puo2_stats.keys()),
        puo2_stats,
        "stress",
        metric_fn=mae,
    )
    for model_name, model_stats in puo2_stats.items():
        write_density_trajectories(
            labels_list=model_stats["stress_labels"],
            ref_vals=model_stats["stress"]["ref"],
            pred_vals=model_stats["stress"]["pred"],
            struct_dir=OUT_PATH / model_name,
            traj_dir=OUT_PATH / model_name / "density_traj_stress",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return result


@pytest.fixture
@build_table(
    filename=OUT_PATH / "puo2_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    energy_mae: dict[str, float | None],
    forces_mae: dict[str, float | None],
    stress_mae: dict[str, float | None],
) -> dict[str, dict]:
    """
    Metric table.

    Parameters
    ----------
    energy_mae
        Energy MAE per model.
    forces_mae
        Force MAE per model.
    stress_mae
        Stress MAE per model.

    Returns
    -------
    dict[str, dict]
        Mapping of metric name to model-value dictionaries.
    """
    return {
        "Energy MAE": energy_mae,
        "Force MAE": forces_mae,
        "Stress MAE": stress_mae,
    }


def test_puo2(
    metrics: dict[str, dict],
    energy_density: dict[str, dict],
    force_density: dict[str, dict],
    stress_density: dict[str, dict],
) -> None:
    """
    Run puo2 analysis.

    Parameters
    ----------
    metrics
        Benchmark metric values.
    energy_density
        Density scatter inputs for energy.
    force_density
        Density scatter inputs for forces.
    stress_density
        Density scatter inputs for stress.
    """
    assert metrics is not None
    assert energy_density is not None
    assert force_density is not None
    assert stress_density is not None

    return
