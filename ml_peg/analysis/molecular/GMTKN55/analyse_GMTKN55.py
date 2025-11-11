"""Analyse GMTKN55 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read
import numpy as np
from numpy.typing import NDArray
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "molecular" / "GMTKN55" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "GMTKN55"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KCAL_PER_MOL = units.mol / units.kcal

# Discard some structures for error calculations
ALLOWED_CHARGES = (0,)
ALLOWED_MULTIPLICITY = (1,)


def structure_info() -> dict[str, dict[str, float] | list | NDArray]:
    """
    Get info from all stored structures.

    Returns
    -------
    dict[str, dict[str, float] | NDArray]
        Dictionary with weights, subset name and category for all systems.
    """
    info = {
        "categories": [],
        "subsets": [],
        "systems": [],
        "excluded": [],
        "weights": {},
        "counts": {},
    }
    for model_name in MODELS:
        for subset in [dir.name for dir in sorted((CALC_PATH / model_name).glob("*"))]:
            count = 0
            for system_path in sorted((CALC_PATH / model_name / subset).glob("*.xyz")):
                count += 1
                structs = read(system_path, index=":")
                info["subsets"].append(subset)

                info["categories"].append(structs[0].info["category"])
                info["systems"].append(structs[0].info["system_name"])
                info["excluded"].append(
                    any(
                        struct.info["excluded"]
                        or struct.info["charge"] not in ALLOWED_CHARGES
                        or struct.info["spin"] not in ALLOWED_MULTIPLICITY
                        for struct in structs
                    )
                )
            info["weights"][subset] = structs[0].info["weight"]
            info["counts"][subset] = count

        # Convert to numpy arrays for filtering
        info["categories"] = np.array(info["categories"])
        info["subsets"] = np.array(info["subsets"])
        info["excluded"] = np.array(info["excluded"])
        # Only need to access info from one model
        return info
    return info


INFO = structure_info()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_rel_energies.json",
    title="Relative energies",
    x_label="Predicted relative energy / kcal/mol",
    y_label="Reference relative energy / kcal/mol",
    hoverdata={
        "Subset": INFO["subsets"],
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
    errors = {}
    for model_name in MODELS:
        errors[model_name] = np.abs(
            np.subtract(rel_energies[model_name], rel_energies["ref"])
        )
    return errors


@pytest.fixture
def subset_errors(all_errors: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """
    Calculate mean error for each subset for all models.

    Parameters
    ----------
    all_errors
        Dictionary of relative MADs, grouped by model.

    Returns
    -------
    dict[str, dict[str, float]]
        Mean error for all models, grouped by subset.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = {}

        # Filter excluded systems from subsets
        errors = all_errors[model_name][np.logical_not(INFO["excluded"])]
        subsets = INFO["subsets"][np.logical_not(INFO["excluded"])]

        for subset in set(subsets):
            results[model_name][subset] = np.mean(errors[subsets == subset])

    return results


@pytest.fixture
def category_errors(
    subset_errors: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Calculate MAD for all models, grouped by category.

    Parameters
    ----------
    subset_errors
        Nested dictionary of mean errors, grouped by model and subset.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dictionary of weighted mean MADs, grouped by model and category.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = {}

        all_categories = INFO["categories"]
        all_subsets = INFO["subsets"]
        all_weights = INFO["weights"]
        all_counts = INFO["counts"]
        excluded = INFO["excluded"]

        # Filter excluded systems
        categories = all_categories[np.logical_not(excluded)]

        for category in set(categories):
            # Filter non-excluded subsets in current category
            filtered_subsets = np.unique(
                all_subsets[np.logical_not(excluded)][categories == category]
            )

            # Get number of systems in each subset
            counts = np.array([all_counts[subset] for subset in filtered_subsets])

            # Get error for each subset
            errors = [subset_errors[model_name][subset] for subset in filtered_subsets]

            # Get weight and count for each subset
            weights = np.array([all_weights[subset] for subset in filtered_subsets])

            results[model_name][category] = np.sum(errors * weights * counts) / np.sum(
                counts
            )

    return results


@pytest.fixture
def weighted_error(subset_errors: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Calculate weighted mean absolute deviation for all models.

    Parameters
    ----------
    subset_errors
        Nested dictionary of mean errors, grouped by model and subset.

    Returns
    -------
    dict[str, dict[str, float]]
        Weighted mean absolute deviation for each model.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = {}

        all_subsets = INFO["subsets"]
        all_weights = INFO["weights"]
        all_counts = INFO["counts"]
        excluded = INFO["excluded"]

        # Filter all non-excluded subsets
        filtered_subsets = np.unique(all_subsets[np.logical_not(excluded)])

        # Get error for each subset
        errors = [subset_errors[model_name][subset] for subset in filtered_subsets]

        # Get weight and count for each subset
        weights = np.array([all_weights[subset] for subset in filtered_subsets])
        counts = np.array([all_counts[subset] for subset in filtered_subsets])

        results[model_name] = np.sum(errors * weights * counts) / np.sum(counts)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "gmtkn55_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
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
