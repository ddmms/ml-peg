"""Analyse Relastab benchmark."""

from __future__ import annotations

import collections
from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest
from scipy.stats import spearmanr

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "defect" / "Relastab" / "outputs"
OUT_PATH = APP_ROOT / "data" / "defect" / "Relastab"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of system names.

    Returns
    -------
    list[str]
        List of system names.
    """
    system_names = []

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            # Use glob for flattened structure
            xyz_files = sorted(model_dir.glob("*.xyz"))
            system_names = [xyz.stem for xyz in xyz_files]
            if system_names:
                break
    return system_names


def get_subset_labels() -> list[str]:
    """
    Get subset label for each system, matching the order from get_system_names.

    Returns
    -------
    list[str]
        List of subset labels, one per system.
    """
    subset_labels = []

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            for xyz_file in xyz_files:
                atoms = read(xyz_file)
                subset_labels.append(atoms.info.get("subset", "unknown"))
            if subset_labels:
                break
    return subset_labels


@pytest.fixture
def grouped_data() -> dict[str, dict[str, list[dict]]]:
    """
    Get data grouped by model and subset.

    Returns
    -------
    dict
        Data grouped by model and subset.
    """
    results = {mlip: {} for mlip in MODELS}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            subset_name = atoms.info.get("subset", "unknown")

            if subset_name not in results[model_name]:
                results[model_name][subset_name] = []

            e_config = atoms.get_potential_energy()
            ref_energy = atoms.info.get("ref", None)

            if ref_energy is None:
                continue

            results[model_name][subset_name].append(
                {
                    "name": xyz_file.stem,
                    "atoms": atoms,
                    "pred": e_config,
                    "ref": ref_energy,
                }
            )

            # Copy structure to output for App
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / xyz_file.name, atoms)

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_energy.json",
    title="Relastab Energies (Shifted per subset)",
    x_label="Predicted Energy (Shifted) / eV",
    y_label="Reference Energy (Shifted) / eV",
    hoverdata={
        "System": get_system_names(),
        "Subset": get_subset_labels(),
    },
)
def stability_energies(grouped_data) -> dict[str, list]:
    """
    Get energies for Relastab systems, flattened for parity plot.

    Energies are shifted by mean PER SUBSET.

    Parameters
    ----------
    grouped_data
        Dictionary of data grouped by model and subset.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    # Gather all subset names
    all_subsets = set()
    for m in MODELS:
        all_subsets.update(grouped_data[m].keys())
    sorted_subsets = sorted(all_subsets)

    def process_model(model_name):
        """
        Collect shifted energies for a single model.

        Parameters
        ----------
        model_name
            Name of the model to process.

        Returns
        -------
        tuple
            Predicted and reference energy lists.
        """
        model_preds = []
        model_refs = []

        for subset in sorted_subsets:
            data_list = grouped_data[model_name].get(subset, [])
            if not data_list:
                continue

            entries = data_list  # already sorted by filename in grouped_data

            preds = [d["pred"] for d in entries]
            refs = [d["ref"] for d in entries]

            # Sub-set mean shift
            mean_pred = np.mean(preds)
            mean_ref = np.mean(refs)

            model_preds.extend([p - mean_pred for p in preds])
            model_refs.extend([r - mean_ref for r in refs])

        return model_preds, model_refs

    # Populate ref only once
    ref_populated = False

    for model_name in MODELS:
        preds, refs = process_model(model_name)
        if not preds:
            continue

        results[model_name] = preds

        if not ref_populated and refs:
            results["ref"] = refs
            ref_populated = True

    return results


@pytest.fixture
def ranking_metrics(
    grouped_data,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute ranking metrics per subset.

    (1) Check if global minimum is identified correctly.
    (2) Spearman rank correlation of the 5 highest (least stable) energy configurations.

    Parameters
    ----------
    grouped_data
        Dictionary of data grouped by model and subset.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Per-subset results.
    """
    subset_results = collections.defaultdict(dict)

    for model_name in MODELS:
        subsets = grouped_data[model_name]
        if not subsets:
            continue

        for subset_name, entries in subsets.items():
            if not entries:
                continue

            ref_values = np.array([d["ref"] for d in entries])
            pred_values = np.array([d["pred"] for d in entries])

            if len(ref_values) < 2:  # Need at least 2 for ranking
                continue

            # --- Metric 1: Global Min Match (per subset) ---
            idx_ref_min = np.argmin(ref_values)
            idx_pred_min = np.argmin(pred_values)
            match = 1.0 if idx_pred_min == idx_ref_min else 0.0

            # --- Metric 2: Top 5 Spearman (Highest/Least Stable) ---
            if len(ref_values) >= 5:
                idx_top5 = np.argsort(ref_values)[-5:]
            else:
                idx_top5 = np.arange(len(ref_values))

            ref_subset = ref_values[idx_top5]
            pred_subset = pred_values[idx_top5]

            with np.errstate(divide="ignore", invalid="ignore"):
                spearman, _ = spearmanr(ref_subset, pred_subset)
                if np.isnan(spearman):
                    spearman = 0.0

            subset_results[subset_name][model_name] = {
                "GlobalMin": match,
                "Top5_Spearman": spearman,
            }

    return dict(subset_results)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "relastab_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(ranking_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict]:
    """
    Get all metrics.

    Returns per-subset metrics as separate columns.

    Parameters
    ----------
    ranking_metrics
        Per-subset results.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    subset_metrics = ranking_metrics

    def pivot(data):
        """
        Pivot from model-keyed to metric-keyed layout.

        Parameters
        ----------
        data
            Dictionary keyed by model name with metric scores.

        Returns
        -------
        dict
            Dictionary keyed by metric name with model scores.
        """
        reshaped = {"Global Min": {}, "Top 5 Spearman": {}}
        for model, scores in data.items():
            reshaped["Global Min"][model] = scores.get("GlobalMin")
            reshaped["Top 5 Spearman"][model] = scores.get("Top5_Spearman")
        return reshaped

    subset_display_names = {
        "fe": "Fe",
        "cawo4": "CaWO4",
    }

    result = {}

    # Add per-subset columns with human-friendly names
    for subset_name, metrics_dict in subset_metrics.items():
        display = subset_display_names.get(subset_name, subset_name)
        subset_pivoted = pivot(metrics_dict)
        for metric_name, model_scores in subset_pivoted.items():
            result[f"{metric_name} {display}"] = model_scores

    return result


def test_relastab_analysis(
    metrics: dict[str, dict],
    stability_energies: dict[str, list],
) -> None:
    """
    Run Relastab analysis test.

    Parameters
    ----------
    metrics
        All Relastab metrics.
    stability_energies
        Parity plot data for shifted energies.
    """
    return
