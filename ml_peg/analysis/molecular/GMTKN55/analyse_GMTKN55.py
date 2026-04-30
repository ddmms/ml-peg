"""Analyse GMTKN55 benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
from numpy.typing import NDArray
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
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
        "elements": [],
        "excluded": [],
        "weights": {},
        "counts": {},
    }
    data_dir = CALC_PATH / "mock"
    structs_dir = OUT_PATH / "mock"
    if not data_dir.exists():
        raise ValueError(f"{data_dir} does not exist. Please run mock calculation.")
    structs_dir.mkdir(parents=True, exist_ok=True)

    for subset in [dir.name for dir in sorted((data_dir).glob("*"))]:
        count = 0
        for system_path in sorted((data_dir / subset).glob("*.xyz")):
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

            info["elements"].append(
                list(
                    set().union(*(struct.get_chemical_symbols() for struct in structs))
                )
            )
            write(structs_dir / f"{count}.xyz", structs)

        info["weights"][subset] = structs[0].info["weight"]
        info["counts"][subset] = count

    # Convert to numpy arrays for filtering
    info["categories"] = np.array(info["categories"])
    info["subsets"] = np.array(info["subsets"])
    info["excluded"] = np.array(info["excluded"])

    out_file = OUT_PATH / "info.json"
    with out_file.open("w", encoding="utf8") as f:
        json.dump({"elements": info["elements"]}, f, indent=1)

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
        count = 0
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

                # Write out all structs in system for app
                structs_dir = OUT_PATH / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / f"{count}.xyz", structs)
                count += 1

        ref_stored = True
    return results


def get_all_errors(rel_energies: dict[str, list[float]]) -> dict[str, list[float]]:
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


def get_subset_errors(
    rel_energies: dict[str, list[float]],
    mask: list[bool],
) -> dict[str, dict[str, float]]:
    """
    Calculate mean error for each subset for all models.

    Parameters
    ----------
    rel_energies
        All reference and predicted relative energies, grouped by model.
    mask
        Additional boolean mask to apply to info.

    Returns
    -------
    dict[str, dict[str, float]]
        Mean error for all models, grouped by subset.
    """
    all_errors = get_all_errors(rel_energies)
    results = {}

    excluded = INFO["excluded"][mask]
    subsets_info = INFO["subsets"][mask]

    valid = ~excluded
    subsets = subsets_info[valid]

    for model_name in MODELS:
        results[model_name] = {}

        # Filter excluded systems from subsets
        errors = all_errors[model_name][valid]

        for subset in set(subsets):
            results[model_name][subset] = np.mean(errors[subsets == subset])

    return results


def get_category_errors(
    subset_errors: dict[str, dict[str, float]],
    mask: list[bool],
) -> dict[str, dict[str, float]]:
    """
    Calculate MAD for all models, grouped by category.

    Parameters
    ----------
    subset_errors
        Nested dictionary of mean errors, grouped by model and subset.
    mask
        Additional boolean mask to apply to info.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Nested dictionary of weighted mean MADs, grouped by model and category.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = {}

        all_categories = INFO["categories"][mask]
        all_subsets = INFO["subsets"][mask]
        excluded = INFO["excluded"][mask]
        all_weights = INFO["weights"]
        all_counts = INFO["counts"]

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


def get_weighted_error(
    subset_errors: dict[str, dict[str, float]], mask: list[bool]
) -> dict[str, float]:
    """
    Calculate weighted mean absolute deviation for all models.

    Parameters
    ----------
    subset_errors
        Nested dictionary of mean errors, grouped by model and subset.
    mask
        Additional boolean mask to apply to info.

    Returns
    -------
    dict[str, dict[str, float]]
        Weighted mean absolute deviation for each model.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = {}

        all_subsets = INFO["subsets"][mask]
        excluded = INFO["excluded"][mask]
        all_weights = INFO["weights"]
        all_counts = INFO["counts"]

        # Filter all non-excluded subsets
        filtered_subsets = np.unique(all_subsets[np.logical_not(excluded)])

        # Get error for each subset
        errors = [subset_errors[model_name][subset] for subset in filtered_subsets]

        # Get weight and count for each subset
        weights = np.array([all_weights[subset] for subset in filtered_subsets])
        counts = np.array([all_counts[subset] for subset in filtered_subsets])

        results[model_name] = np.sum(errors * weights * counts) / np.sum(counts)

    return results


def get_metrics(
    rel_energies: dict[str, list[float]], mask: list[bool] | None = None
) -> dict[str, dict]:
    """
    Get all GMTKN55 metrics.

    Parameters
    ----------
    rel_energies
        All reference and predicted relative energies, grouped by model.
    mask
        Additional boolean mask to apply to info. Default is `True` for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    if mask is None:
        mask = [True] * len(INFO["subsets"])
    subset_errors = get_subset_errors(rel_energies, mask=mask)
    category_errors = get_category_errors(subset_errors, mask=mask)
    weighted_error = get_weighted_error(subset_errors, mask=mask)

    category_abbrevs = {
        "Basic properties and reaction energies for small systems": "Small systems",
        "Reaction energies for large systems and isomerisation reactions": "Large "
        "systems",
        "Reaction barrier heights": "Barrier heights",
        "Intramolecular noncovalent interactions": "Intramolecular NCIs",
        "Intermolecular noncovalent interactions": "Intermolecular NCIs",
    }

    metrics = {}
    for full_category, short_category in category_abbrevs.items():
        metrics[short_category] = {
            model: category_errors[model][full_category] for model in MODELS
        }

    return metrics | {"WTMAD": weighted_error}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "gmtkn55_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(rel_energies: dict[str, list[float]]) -> dict[str, dict]:
    """
    Get all GMTKN55 metrics.

    Parameters
    ----------
    rel_energies
        All reference and predicted relative energies, grouped by model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return get_metrics(rel_energies)


def test_gmtkn55(metrics):
    """
    Run GMTKN55 test.

    Parameters
    ----------
    metrics
        All GMTKN55 metrics.
    """
    return
