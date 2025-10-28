"""Analyse GMTKN55 benchmark."""

from __future__ import annotations

from ase import units
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "molecular" / "GMTKN55" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "GMTKN55"

DEFAULT_WEIGHTS = {
    "Small systems": 0,
    "Large systems": 0,
    "Barrier heights": 0,
    "Intramolecular NCIs": 0,
    "Intermolecular NCIs": 0,
    "WTMAD": 1,
}

DEFAULT_THRESHOLDS = {
    "Small systems": (0.0, 100.0),
    "Large systems": (0.0, 100.0),
    "Barrier heights": (0.0, 100.0),
    "Intramolecular NCIs": (0.0, 100.0),
    "Intermolecular NCIs": (0.0, 100.0),
    "WTMAD": (0.0, 100.0),
}

# Unit conversion
EV_TO_KCAL_PER_MOL = units.mol / units.kcal

# Discard some structures for error calculations
ALLOWED_CHARGES = (0,)
ALLOWED_UHFS = (0,)


def strucutre_info() -> dict[str, float | str]:
    """
    Get info from all stored structures.

    Returns
    -------
    dict[str, float | str]
        Dictionary with weights, subset name and category for all systems.
    """
    info = {
        "weights": [],
        "categories": [],
        "subsets": [],
        "systems": [],
        "excluded": [],
        "charges": [],
        "uhfs": [],
    }
    for model_name in MODELS:
        for subset in [dir.name for dir in sorted((CALC_PATH / model_name).glob("*"))]:
            for system_path in sorted((CALC_PATH / model_name / subset).glob("*.xyz")):
                struct = read(system_path, index=0)
                info["weights"].append(struct.info["weight"])
                info["subsets"].append(struct.info["subset_name"])
                info["categories"].append(struct.info["category"])
                info["systems"].append(struct.info["system_name"])
                info["excluded"].append(struct.info["excluded"])
                info["charges"].append(struct.info["Charge"])
                info["uhfs"].append(struct.info["uhf"])
        # Only need to access info from one model
        return info
    return info


INFO = strucutre_info()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_rel_energies.json",
    title="Relative energies",
    x_label="Predicted relative energy / kcal/mol",
    y_label="Reference relative energy / kcal/mol",
    hoverdata={
        "Subset": INFO["subsets"],
        "Weight": INFO["weights"],
        "Category": INFO["categories"],
        "System": INFO["systems"],
        "Excluded": INFO["excluded"],
    },
)
def rel_energies() -> dict[str, list[float]]:
    """
    Calculate relative energies for all 1505 systems.

    Returns
    -------
    dict[str, list[float]]
        Dictionary of all reference and predicted relative energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False
    for model_name in MODELS:
        for subset in [dir.name for dir in sorted((CALC_PATH / model_name).glob("*"))]:
            for system_path in sorted((CALC_PATH / model_name / subset).glob("*.xyz")):
                structs = read(system_path, index=":")
                pred_rel_energy = 0

                for struct in structs:
                    # Count is defined to give the correct relative energy
                    pred_rel_energy += (
                        struct.get_potential_energy() * struct.info["count"]
                    )

                results[model_name].append(pred_rel_energy * EV_TO_KCAL_PER_MOL)

                # Only store reference results from first model
                # Shared by all structures in a system, so can use last structure
                if not ref_stored:
                    results["ref"].append(struct.info["ref_value"])

        ref_stored = True
    return results


@pytest.fixture
def all_errors(rel_energies: dict[str, list[float]]) -> dict[str, list[float]]:
    """
    Calculate MAD for all models for all systems with respect to reference.

    Parameters
    ----------
    rel_energies
        All reference and predicted relative energies, grouped by model.

    Returns
    -------
    dict[str, list[float]]
        Dictionary of relative MADs, grouped by model.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = np.abs(
            np.subtract(rel_energies[model_name], rel_energies["ref"])
        )
    return results


@pytest.fixture
def category_errors(all_errors: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """
    Calculate MAD for all models, grouped by category.

    Parameters
    ----------
    all_errors
        Dictionary of relative MADs, grouped by model.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dictionary of relative MADs, grouped by model and category.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = {}
        for category in set(INFO["categories"]):
            results[model_name][category] = np.mean(
                [
                    error * weight
                    for error, weight, cat, excluded, charge, multiplicity in zip(
                        all_errors[model_name],
                        INFO["weights"],
                        INFO["categories"],
                        INFO["excluded"],
                        INFO["charges"],
                        INFO["uhfs"],
                        strict=True,
                    )
                    if cat == category
                    and not excluded
                    and charge in ALLOWED_CHARGES
                    and multiplicity in ALLOWED_UHFS
                ]
            )
    return results


@pytest.fixture
def weighted_error(all_errors: dict[str, list[float]]) -> dict[str, float]:
    """
    Calculate weighted mean absolute deviation for all models.

    Parameters
    ----------
    all_errors
        Dictionary of relative MADs, grouped by model.

    Returns
    -------
    dict[str, dict[str, float]]
        Weighted mean absolute deviation for each model.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = np.mean(
            [
                error * weight
                for error, weight, excluded, charge, multiplicity in zip(
                    all_errors[model_name],
                    INFO["weights"],
                    INFO["excluded"],
                    INFO["charges"],
                    INFO["uhfs"],
                    strict=True,
                )
                if not excluded
                and charge in ALLOWED_CHARGES
                and multiplicity in ALLOWED_UHFS
            ]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "gmtkn55_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "Small systems": "Mean Absolute Deviation (kcal/mol)",
        "Large systems": "Mean Absolute Deviation (kcal/mol)",
        "Barrier heights": "Mean Absolute Deviation (kcal/mol)",
        "Intramolecular NCIs": "Mean Absolute Deviation (kcal/mol)",
        "Intermolecular NCIs": "Mean Absolute Deviation (kcal/mol)",
        "WTMAD": "Weighted Mean Absolute Deviation (kcal/mol)",
    },
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    category_errors: dict[str, dict[str, float]], weighted_error: dict[str, float]
) -> dict[str, dict]:
    """
    Get all GMTKN55 metrics.

    Parameters
    ----------
    category_errors
        Relative errors for each models, grouped by categories.
    weighted_error
        Weighted relative error for each model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    category_abbrevs = {
        "Basic properties and reaction energies for small systems": "Small systems",
        "Reaction energies for large systems and isomerisation reactions": "Large "
        "systems",
        "Reaction barrier heights": "Barrier heights",
        "Intramolecular noncovalent interactions": "Intramolecular NCIs",
        "Intermolecular noncovalent interactions": "Intermolecular NCIs",
        "All (WTMAD)": "All (WTMAD)",
    }

    metrics = {}
    for category in category_errors[MODELS[0]]:
        metrics[category_abbrevs[category]] = {
            model: category_errors[model][category] for model in MODELS
        }

    return metrics | {"WTMAD": weighted_error}


def test_gmtkn55(metrics):
    """
    Run GMTKN55 test.

    Parameters
    ----------
    metrics
        All GMTKN55 metrics.
    """
    return
