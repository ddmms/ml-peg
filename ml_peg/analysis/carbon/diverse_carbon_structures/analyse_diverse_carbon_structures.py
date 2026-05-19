"""Analyse diverse carbon structures benchmark (ACE training dataset, DFT/PBE)."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import (
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

CALC_PATH = CALCS_ROOT / "carbon" / "diverse_carbon_structures" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "diverse_carbon_structures"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Maps dataset category string -> (folder name for file paths, display metric name)
CATEGORIES = {
    "sp2 bonded": ("sp2", "sp2 bonded MAE"),
    "sp3 bonded": ("sp3", "sp3 bonded MAE"),
    "amorphous/liquid": ("amorphous", "amorphous/liquid MAE"),
    "general bulk": ("general_bulk", "general bulk MAE"),
    "general clusters": ("general_clusters", "general clusters MAE"),
}


@pytest.fixture
def all_energies() -> dict[str, dict]:
    """
    Load calc results for all models, split by category, write per-structure files.

    Returns
    -------
    dict[str, dict]
        Nested dict: model -> category_folder -> {ref, pred, labels}.
    """
    folders = [folder for folder, _ in CATEGORIES.values()]
    empty: dict = {
        "ref": [],
        "pred": [],
        "labels": [],
    }
    data: dict[str, dict] = {
        model: {folder: {k: list(v) for k, v in empty.items()} for folder in folders}
        for model in MODELS
    }

    for model in MODELS:
        cat_counters: dict[str, int] = dict.fromkeys(folders, 0)
        atoms_list = read(CALC_PATH / model / "results.xyz", ":")

        for atoms in tqdm(atoms_list):
            cat = atoms.info.get("category")
            if cat not in CATEGORIES:
                continue
            folder, _ = CATEGORIES[cat]

            n_atoms = len(atoms)
            ref_e = atoms.info["ref_energy"] / n_atoms
            pred_e = atoms.info["pred_energy"] / n_atoms
            idx = cat_counters[folder]
            cat_counters[folder] += 1

            data[model][folder]["ref"].append(ref_e)
            data[model][folder]["pred"].append(pred_e)
            data[model][folder]["labels"].append(str(idx))

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
def interactive_dataset(all_energies: dict) -> dict:
    """
    Build cell_to_scatter dataset with one metric per category.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict
        Data bundle consumed by cell_to_scatter decorator.
    """
    dataset: dict = {
        "metrics": {metric_name: metric_name for _, metric_name in CATEGORIES.values()},
        "models": {},
    }
    for model in MODELS:
        model_metrics = {}
        for _cat, (folder, metric_name) in CATEGORIES.items():
            d = all_energies[model][folder]
            points = [
                {"id": label, "ref": r, "pred": p}
                for label, r, p in zip(d["labels"], d["ref"], d["pred"], strict=True)
            ]
            model_metrics[metric_name] = {"points": points}
        dataset["models"][model] = {"metrics": model_metrics}
    return dataset


@pytest.fixture
@build_table(
    filename=OUT_PATH / "diverse_carbon_structures_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(all_energies: dict) -> dict[str, dict]:
    """
    Compute per-category energy MAE for each model.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Metric name → model → MAE value.
    """
    result: dict[str, dict] = {}
    for _cat, (folder, metric_name) in CATEGORIES.items():
        result[metric_name] = {
            model: mae(
                all_energies[model][folder]["ref"],
                all_energies[model][folder]["pred"],
            )
            for model in MODELS
        }
    return result


def test_diverse_carbon_structures(
    interactive_dataset: dict,
    metrics: dict,
) -> None:
    """
    Run diverse carbon structures analysis (drives all fixtures).

    Parameters
    ----------
    interactive_dataset
        Per-category scatter inputs from interactive_dataset fixture.
    metrics
        Per-category energy MAE for each model.
    """
    return
