"""Analyse GMTKN55 benchmark."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

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

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*/*.xyz",
    index=0,
    info_keys=["category", "system_name", "weight"],
    include_filenames=True,
    include_dirs=True,
    per_file_info={
        "excluded": lambda structs: any(
            struct.info["excluded"]
            or struct.info["charge"] not in ALLOWED_CHARGES
            or struct.info["spin"] not in ALLOWED_MULTIPLICITY
            for struct in structs
        ),
    },
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
)

# Convert to numpy arrays for filtering
INFO["categories"] = np.array(INFO["category"])
INFO["subsets"] = np.array(INFO["dirs"])
INFO["excluded"] = np.array(INFO["excluded"])
INFO["systems"] = INFO["system_name"]

# Weight is shared by all systems in a subset
INFO["weights"] = dict(zip(INFO["dirs"], INFO["weight"], strict=True))
INFO["counts"] = Counter(INFO["dirs"])

_CATEGORY_LABELS = {
    "Basic properties and reaction energies for small systems": "Small systems",
    "Intermolecular noncovalent interactions": "Intermolecular NCIs",
    "Intramolecular noncovalent interactions": "Intramolecular NCIs",
    "Reaction barrier heights": "Barrier heights",
    "Reaction energies for large systems and isomerisation reactions": "Large systems",
}


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
    symbol_by=INFO["categories"].tolist(),
    symbol_labels=_CATEGORY_LABELS,
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
    systems = list(zip(INFO["dirs"], INFO["filenames"], strict=True))

    for model_name in MODELS:
        for count, (subset, label) in enumerate(systems):
            system_path = CALC_PATH / model_name / subset / f"{label}.xyz"
            if not system_path.exists():
                results[model_name].append(float("nan"))
                continue

            structs = read(system_path, index=":")
            pred_rel_energy = 0

            for struct in structs:
                # Count is defined to give the correct relative energy
                pred_rel_energy += struct.get_potential_energy() * struct.info["count"]

            results[model_name].append(pred_rel_energy * EV_TO_KCAL_PER_MOL)

            # Only store reference results from first model with all systems
            # Shared by all structures in a system, so can use last structure
            if not ref_stored:
                results["ref"].append(struct.info["ref_value"])

            # Write out all structs in system for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{count}.xyz", structs)

        if not ref_stored:
            if len(results["ref"]) == len(systems):
                ref_stored = True
            else:
                results["ref"] = []

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
    n_systems = len(INFO["filenames"])

    errors = {}
    for model_name in MODELS:
        # Errors must match INFO to allow filtering by excluded systems
        if len(rel_energies["ref"]) == n_systems == len(rel_energies[model_name]):
            errors[model_name] = np.abs(
                np.subtract(rel_energies[model_name], rel_energies["ref"])
            )
        else:
            errors[model_name] = np.full(n_systems, np.nan)
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
    mlip_name_map=DISPERSION_NAME_MAP,
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
    }

    metrics = {}
    for full_category, short_category in category_abbrevs.items():
        metrics[short_category] = {
            model: category_errors[model][full_category] for model in MODELS
        }

    return metrics | {"WTMAD": weighted_error}


@pytest.mark.framework("mace-multihead")
def test_gmtkn55(metrics):
    """
    Run GMTKN55 test.

    Parameters
    ----------
    metrics
        All GMTKN55 metrics.
    """
    return
