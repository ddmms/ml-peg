"""Analyse diverse carbon structures benchmark (ACE training dataset, DFT/PBE)."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import plotly.graph_objects as go
import pytest
from tqdm import tqdm

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import (
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

CALC_PATH = CALCS_ROOT / "carbon" / "diverse_carbon_structures" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "diverse_carbon_structures"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Maps dataset category string -> folder name and table metric names
CATEGORIES = {
    "sp2 bonded": ("sp2", "sp2 Bonded MAE", "sp2 Bonded Force MAE"),
    "sp3 bonded": ("sp3", "sp3 Bonded MAE", "sp3 Bonded Force MAE"),
    "amorphous/liquid": (
        "amorphous",
        "Amorphous/Liquid MAE",
        "Amorphous/Liquid Force MAE",
    ),
    "general bulk": ("general_bulk", "General Bulk MAE", "General Bulk Force MAE"),
    "general clusters": (
        "general_clusters",
        "General Clusters MAE",
        "General Clusters Force MAE",
    ),
}


@pytest.fixture
def all_results() -> dict[str, dict]:
    """
    Load calc results for all models, split by category, write per-structure files.

    Returns
    -------
    dict[str, dict]
        Nested dict with energy, force, and label data by model and category.
    """
    data: dict[str, dict] = {
        model: {
            folder: {
                "ref": [],
                "pred": [],
                "labels": [],
                "force_mae": [],
                "force_abs_error_sum": 0.0,
                "force_component_count": 0,
            }
            for folder, _, _ in CATEGORIES.values()
        }
        for model in MODELS
    }

    for model in MODELS:
        cat_counters: dict[str, int] = {
            folder: 0 for folder, _, _ in CATEGORIES.values()
        }
        atoms_list = read(CALC_PATH / model / "results.xyz", ":")

        for atoms in tqdm(atoms_list):
            cat = atoms.info.get("category")
            if cat not in CATEGORIES:
                continue
            folder, _, _ = CATEGORIES[cat]

            n_atoms = len(atoms)
            ref_e = atoms.info["ref_energy"] / n_atoms
            pred_e = atoms.info["pred_energy"] / n_atoms
            ref_forces = np.asarray(atoms.arrays["ref_forces"], dtype=float)
            pred_forces = np.asarray(atoms.arrays["pred_forces"], dtype=float)
            force_abs_error = np.abs(pred_forces - ref_forces)
            idx = cat_counters[folder]
            cat_counters[folder] += 1

            data[model][folder]["ref"].append(ref_e)
            data[model][folder]["pred"].append(pred_e)
            data[model][folder]["labels"].append(str(idx))
            data[model][folder]["force_mae"].append(float(np.mean(force_abs_error)))
            data[model][folder]["force_abs_error_sum"] += float(np.sum(force_abs_error))
            data[model][folder]["force_component_count"] += int(force_abs_error.size)

            struct_dir = OUT_PATH / model / folder
            struct_dir.mkdir(parents=True, exist_ok=True)
            write(struct_dir / f"{idx}.xyz", atoms)

    return data


@pytest.fixture
@cell_to_scatter(
    filename=OUT_PATH / "diverse_carbon_structures_scatter.json",
    x_label="Predicted energy / eV atom⁻¹",
    y_label="Reference energy / eV atom⁻¹",
)
def interactive_dataset(all_results: dict) -> dict:
    """
    Build cell_to_scatter dataset with one metric per category.

    Parameters
    ----------
    all_results
        Nested dict with energy, force, and label data by model and category.

    Returns
    -------
    dict
        Data bundle consumed by cell_to_scatter decorator.
    """
    dataset: dict = {
        "metrics": {
            energy_metric: energy_metric for _, energy_metric, _ in CATEGORIES.values()
        },
        "models": {},
    }
    for model in MODELS:
        model_metrics = {}
        for _cat, (folder, energy_metric, _) in CATEGORIES.items():
            d = all_results[model][folder]
            points = [
                {"id": label, "ref": r, "pred": p}
                for label, r, p in zip(d["labels"], d["ref"], d["pred"], strict=True)
            ]
            model_metrics[energy_metric] = {"points": points}
        dataset["models"][model] = {"metrics": model_metrics}
    return dataset


@pytest.fixture
def force_error_plots(all_results: dict) -> dict[tuple[str, str], Path]:
    """
    Write per-structure force error plots for each model/category pair.

    Parameters
    ----------
    all_results
        Nested dict with energy, force, and label data by model and category.

    Returns
    -------
    dict[tuple[str, str], Path]
        ``(model, category_folder)`` -> written Plotly JSON path.
    """
    paths: dict[tuple[str, str], Path] = {}
    for model in MODELS:
        for _cat, (folder, _, force_metric) in CATEGORIES.items():
            d = all_results[model][folder]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=d["labels"],
                    y=d["force_mae"],
                    mode="markers",
                    hovertemplate=(
                        "<b>Structure: </b>%{x}<br>"
                        "<b>Force MAE: </b>%{y:.4f} eV/Å<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            fig.update_layout(
                title={"text": f"{model} - {force_metric}"},
                xaxis={"title": {"text": "Structure index"}},
                yaxis={"title": {"text": "Mean absolute force error / eV/Å"}},
            )
            path = OUT_PATH / f"figure_{model}_{folder}_force_mae.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_json(path)
            paths[(model, folder)] = path
    return paths


@pytest.fixture
@build_table(
    filename=OUT_PATH / "diverse_carbon_structures_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(all_results: dict) -> dict[str, dict]:
    """
    Compute per-category energy and force MAE for each model.

    Parameters
    ----------
    all_results
        Nested dict with energy, force, and label data by model and category.

    Returns
    -------
    dict[str, dict]
        Metric name → model → MAE value.
    """
    result: dict[str, dict] = {}
    for _cat, (folder, energy_metric, force_metric) in CATEGORIES.items():
        result[energy_metric] = {
            model: mae(
                all_results[model][folder]["ref"],
                all_results[model][folder]["pred"],
            )
            for model in MODELS
        }
        result[force_metric] = {
            model: (
                all_results[model][folder]["force_abs_error_sum"]
                / all_results[model][folder]["force_component_count"]
            )
            for model in MODELS
        }
    return result


def test_diverse_carbon_structures(
    interactive_dataset: dict,
    force_error_plots: dict[tuple[str, str], Path],
    metrics: dict,
) -> None:
    """
    Run diverse carbon structures analysis (drives all fixtures).

    Parameters
    ----------
    interactive_dataset
        Per-category scatter inputs from interactive_dataset fixture.
    force_error_plots
        Per-category force error plot paths.
    metrics
        Per-category energy and force MAE for each model.
    """
    return
