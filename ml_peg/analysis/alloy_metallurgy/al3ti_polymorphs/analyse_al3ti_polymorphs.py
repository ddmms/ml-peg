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


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_formation_energies.json",
    title="Al3Ti Formation Energies",
    x_label="Predicted formation energy / eV/atom",
    y_label="Reference formation energy / eV/atom",
)
def formation_energies() -> dict[str, list]:
    """
    Get formation energies for all Al3Ti polymorphs.

    Returns
    -------
    dict[str, list]
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
        al_energy = al_struct.get_potential_energy() / len(al_struct)
        ti_energy = ti_struct.get_potential_energy() / len(ti_struct)

        predictions = []
        for name in POLYMORPHS:
            poly_struct = load_structure(name, model_dir)
            if poly_struct is None:
                predictions.append(np.nan)
                continue
            poly_energy = poly_struct.get_potential_energy() / len(poly_struct)
            formation_energy = poly_energy - 0.75 * al_energy - 0.25 * ti_energy
            predictions.append(formation_energy)
        results[model_name] = predictions

    # Store reference energies
    references = load_references()
    missing = [name for name in POLYMORPHS if name not in references]
    if missing:
        raise ValueError(f"Missing DFT references for: {', '.join(missing)}")
    ref = [
        float(references[name]["formation_energy_0k_ev_atom"]["value"])
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
@build_table(
    filename=OUT_PATH / "al3ti_metrics.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    formation_energy_errors: dict[str, float],
    polymorph_order_accuracies: dict[str, float],
) -> dict[str, dict]:
    """
    Get all Al3Ti metrics.

    Parameters
    ----------
    formation_energy_errors
        Mean absolute formation energy error averaged across all polymorphs.
    polymorph_order_accuracies
        Pairwise accuracy of the predicted polymorph stability ordering.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    copy_structures_to_app_data()
    return {
        "Formation Energy MAE": formation_energy_errors,
        "Relative Stability Ordering Accuracy": polymorph_order_accuracies,
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
