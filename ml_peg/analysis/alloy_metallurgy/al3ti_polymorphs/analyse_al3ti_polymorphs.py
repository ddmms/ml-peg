"""Analyse the Al3Ti polymorph benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "alloy_metallurgy" / "al3ti_polymorphs" / "outputs"
OUT_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "al3ti_polymorphs"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

REQUIRED_ELEMENTS = ("Al", "Ti")
AL_REFERENCE = "Al_fcc"
TI_REFERENCE = "Ti_hcp"
POLYMORPHS = ("D022", "D023", "L12")


def write_benchmark_info() -> None:
    """Write benchmark-wide element requirements for application filtering."""
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(
        json.dumps({"elements": REQUIRED_ELEMENTS}, indent=1),
        encoding="utf8",
    )


def load_references() -> dict[str, Any]:
    """
    Load references for all polymorph computations.

    Returns
    -------
    dict[str, Any]
        Reference data keyed by polymorph name.
    """
    ref_path = CALC_PATH / "references" / "DFT.json"
    try:
        with ref_path.open(encoding="utf8") as file:
            references = json.load(file)
        if not isinstance(references, dict):
            raise TypeError("expected a JSON object")
        return references
    except Exception as exc:
        warn(
            f"Unable to load reference data from {ref_path}: {exc}",
            stacklevel=2,
        )
        return {}


@pytest.fixture
def reference_data() -> dict[str, Any]:
    """
    Load DFT references once for the analysis fixtures.

    Returns
    -------
    dict[str, Any]
        Reference data keyed by polymorph name.
    """
    return load_references()


def load_structure(structure: str, data_path: Path) -> Atoms | None:
    """
    Load a structure from a xyz file.

    Parameters
    ----------
    structure
        Name of the structure to be loaded.
    data_path
        Path to the root of the data directory.

    Returns
    -------
    Atoms
        Loaded structure.
    None
        If loading failed.
    """
    structure_path = data_path / f"{structure}.xyz"
    try:
        return read(structure_path)
    except Exception as exc:
        warn(f"Error loading structure {structure}: {exc}", stacklevel=2)
        return None


def energy_per_atom(atoms: Atoms, model_name: str, structure: str) -> float:
    """
    Get energy per atom, returning NaN if evaluation fails.

    Parameters
    ----------
    atoms
        Structure with a stored calculator result.
    model_name
        Model that produced the structure.
    structure
        Name of the structure.

    Returns
    -------
    float
        Potential energy per atom, or NaN on failure.
    """
    try:
        energy = atoms.get_potential_energy() / len(atoms)
        if not np.isfinite(energy):
            raise ValueError("non-finite energy")
        return float(energy)
    except Exception as exc:
        warn(
            f"Error getting energy for model {model_name}, "
            f"structure {structure}: {exc}",
            stacklevel=2,
        )
        return np.nan


@pytest.fixture
def relaxed_polymorphs() -> dict[str, list[Atoms | None]]:
    """
    Load each model's relaxed polymorphs in ``POLYMORPHS`` order.

    Returns
    -------
    dict[str, list[Atoms | None]]
        Relaxed structures by model, with missing structures represented by None.
    """
    structures = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        structures[model_name] = (
            [load_structure(name, model_dir) for name in POLYMORPHS]
            if model_dir.exists()
            else [None] * len(POLYMORPHS)
        )
    return structures


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_formation_energies.json",
    title="Al3Ti Formation Energies",
    x_label="Predicted formation energy / eV/atom",
    y_label="Reference formation energy / eV/atom",
)
def formation_energies(
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, list[float]]:
    """
    Get formation energies for all Al3Ti polymorphs.

    Parameters
    ----------
    reference_data
        DFT reference values for the three polymorphs.
    relaxed_polymorphs
        Relaxed structures by model, in polymorph order.

    Returns
    -------
    dict[str, list[float]]
        Dictionary of reference and predicted formation energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        # Initialize nan results
        results[model_name] = [np.nan] * len(POLYMORPHS)

        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        al_struct = load_structure(AL_REFERENCE, model_dir)
        ti_struct = load_structure(TI_REFERENCE, model_dir)
        if al_struct is None or ti_struct is None:
            # Formation energies need both elemental references.
            continue
        al_energy = energy_per_atom(al_struct, model_name, AL_REFERENCE)
        ti_energy = energy_per_atom(ti_struct, model_name, TI_REFERENCE)
        if not np.isfinite(al_energy) or not np.isfinite(ti_energy):
            continue

        predictions = []
        for name, poly_struct in zip(
            POLYMORPHS, relaxed_polymorphs[model_name], strict=True
        ):
            if poly_struct is None:
                predictions.append(np.nan)
                continue
            poly_energy = energy_per_atom(poly_struct, model_name, name)
            formation_energy = poly_energy - 0.75 * al_energy - 0.25 * ti_energy
            predictions.append(formation_energy)
        results[model_name] = predictions

    # Store reference energies
    missing = [name for name in POLYMORPHS if name not in reference_data]
    if missing:
        raise ValueError(f"Missing DFT references for: {', '.join(missing)}")
    ref = [
        float(reference_data[name]["formation_energy_0k_ev_atom"]["value"])
        for name in POLYMORPHS
    ]
    results["ref"] = ref

    return results


def pairwise_order_accuracy(reference: list[float], prediction: list[float]) -> float:
    """
    Return the percentage of polymorph pairs ordered as in the reference.

    Parameters
    ----------
    reference
        Reference formation energies, in the order given by ``POLYMORPHS``.
    prediction
        Predicted formation energies, in the same order.

    Returns
    -------
    float
        Percentage of pairwise comparisons with matching energy order.
    """
    reference_values = np.asarray(reference, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if (
        reference_values.ndim != 1
        or prediction_values.ndim != 1
        or reference_values.size < 2
        or reference_values.size != prediction_values.size
        or not np.isfinite(reference_values).all()
        or not np.isfinite(prediction_values).all()
    ):
        return np.nan

    correct_pairs = 0
    total_pairs = 0
    for i in range(reference_values.size):
        for j in range(i + 1, reference_values.size):
            reference_order = np.sign(reference_values[i] - reference_values[j])
            prediction_order = np.sign(prediction_values[i] - prediction_values[j])
            correct_pairs += int(reference_order == prediction_order)
            total_pairs += 1

    return float(100.0 * correct_pairs / total_pairs)


@pytest.fixture
def formation_energy_errors(formation_energies) -> dict[str, float]:
    """
    Get mean absolute error for formation energies.

    Parameters
    ----------
    formation_energies
        Dictionary of reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted formation energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if formation_energies.get(model_name):
            results[model_name] = mae(
                formation_energies["ref"], formation_energies[model_name]
            )
        else:
            results[model_name] = np.nan
    return results


@pytest.fixture
def polymorph_order_accuracies(formation_energies) -> dict[str, float]:
    """
    Calculate pairwise stability-order accuracy for each model.

    Parameters
    ----------
    formation_energies
        Dictionary of reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted order accuracies for all models.
    """
    reference = formation_energies["ref"]
    return {
        model_name: pairwise_order_accuracy(
            reference, formation_energies.get(model_name, [])
        )
        for model_name in MODELS
    }


@pytest.fixture
def volumes_per_atom(
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, list[float]]:
    """
    Get DFT and relaxed volumes per atom in ``POLYMORPHS`` order.

    Parameters
    ----------
    reference_data
        DFT reference values for the three polymorphs.
    relaxed_polymorphs
        Relaxed structures by model, in polymorph order.

    Returns
    -------
    dict[str, list[float]]
        Dictionary of predicted and reference volumes per atom.
    """
    volumes = {
        "ref": [
            float(reference_data[name]["volume_per_atom_angstrom3"]["value"])
            for name in POLYMORPHS
        ]
    }
    for model_name, structures in relaxed_polymorphs.items():
        values = []
        for atoms in structures:
            if atoms is None or not len(atoms) or atoms.cell.rank != 3:
                values.append(np.nan)
                continue
            volume = atoms.get_volume() / len(atoms)
            values.append(volume if np.isfinite(volume) and volume > 0 else np.nan)
        volumes[model_name] = values
    return volumes


@pytest.fixture
def volume_errors(volumes_per_atom: dict[str, list[float]]) -> dict[str, float]:
    """
    Get the volume-per-atom mean absolute error for each model.

    Parameters
    ----------
    volumes_per_atom
        Reference and model volumes per atom in polymorph order.

    Returns
    -------
    dict[str, float]
        Volume-per-atom MAE in Å³/atom for each model.
    """
    return {
        model_name: mae(volumes_per_atom["ref"], values)
        for model_name, values in volumes_per_atom.items()
        if model_name != "ref"
    }


@pytest.fixture
def lattice_parameter_errors(
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, float]:
    """
    Get equally polymorph-weighted lattice parameter relative errors.

    The two basal lengths share one parameter weight in each tetragonal phase.

    Parameters
    ----------
    reference_data
        DFT lattice parameters for the three polymorphs.
    relaxed_polymorphs
        Relaxed structures by model, in polymorph order.

    Returns
    -------
    dict[str, float]
        Mean absolute percentage error for each model.
    """
    errors = {}
    for model_name, structures in relaxed_polymorphs.items():
        polymorph_errors = []
        for name, atoms in zip(POLYMORPHS, structures, strict=True):
            if atoms is None or atoms.cell.rank != 3 or atoms.cell.volume <= 0:
                polymorph_errors.append(np.nan)
                continue
            lengths = atoms.cell.lengths()
            a_ref = float(reference_data[name]["lattice_a_angstrom"]["value"])
            if name == "L12":
                polymorph_errors.append(np.mean(np.abs(lengths / a_ref - 1)))
            else:
                c_ref = float(reference_data[name]["lattice_c_angstrom"]["value"])
                basal_error = np.mean(np.abs(lengths[:2] / a_ref - 1))
                c_error = abs(lengths[2] / c_ref - 1)
                polymorph_errors.append((basal_error + c_error) / 2)
        errors[model_name] = (
            float(100.0 * np.mean(polymorph_errors))
            if np.isfinite(polymorph_errors).all()
            else np.nan
        )
    return errors


@pytest.fixture
@build_table(
    filename=OUT_PATH / "al3ti_metrics.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    formation_energy_errors: dict[str, float],
    polymorph_order_accuracies: dict[str, float],
    volume_errors: dict[str, float],
    lattice_parameter_errors: dict[str, float],
) -> dict[str, dict]:
    """
    Get all Al3Ti metrics.

    Parameters
    ----------
    formation_energy_errors
        Mean absolute formation energy error averaged across all polymorphs.
    polymorph_order_accuracies
        Pairwise accuracy of the predicted polymorph stability ordering.
    volume_errors
        Volume-per-atom mean absolute errors across all polymorphs.
    lattice_parameter_errors
        Equally polymorph-weighted lattice parameter relative errors.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    copy_structures_to_app_data()
    return {
        "Formation Energy MAE": formation_energy_errors,
        "Relative Stability Ordering Accuracy": polymorph_order_accuracies,
        "Volume MAE": volume_errors,
        "Lattice Parameter MAPE": lattice_parameter_errors,
    }


def copy_structures_to_app_data() -> None:
    """Copy calculated structures into the app data directory."""
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        output_dir = OUT_PATH / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for name in POLYMORPHS:
            source = model_dir / f"{name}.xyz"
            if not source.exists():
                continue
            try:
                write(output_dir / source.name, read(source))
            except Exception as exc:
                warn(f"Unable to copy structure {source}: {exc}", stacklevel=2)


def test_al3ti_polymorphs(metrics: dict[str, dict[str, float]]) -> None:
    """
    Run al3ti polymorph regression analysis.

    Parameters
    ----------
    metrics
        Polymorph analysis metrics.
    """
    write_benchmark_info()
    return
