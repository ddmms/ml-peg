"""Analyse diverse carbon structures benchmark (ACE training dataset, DFT/PBE)."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    load_metrics_config,
    mae,
    write_density_trajectories,
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
        Nested dict: model → category_folder → {ref, pred, labels}.
        Individual structure xyz files are written to OUT_PATH / model / folder /
        as a side effect (needed by write_density_trajectories).
    """
    folders = [folder for folder, _ in CATEGORIES.values()]
    empty: dict = {
        "ref": [],
        "pred": [],
        "labels": [],
        "ref_forces": [],
        "pred_forces": [],
        "force_labels": [],
    }
    data: dict[str, dict] = {
        model: {folder: {k: list(v) for k, v in empty.items()} for folder in folders}
        for model in MODELS
    }

    for model in MODELS:
        cat_counters: dict[str, int] = dict.fromkeys(folders, 0)
        atoms_list = read(CALC_PATH / model / "results.xyz", ":")

        for atoms in atoms_list:
            cat = atoms.info.get("category")
            if cat not in CATEGORIES:
                continue
            folder, _ = CATEGORIES[cat]

            n_atoms = len(atoms)
            ref_e = atoms.info["ref_energy"] / n_atoms
            pred_e = atoms.info["pred_energy"] / n_atoms
            ref_f = np.asarray(atoms.arrays["ref_forces"], dtype=float).reshape(-1)
            pred_f = np.asarray(atoms.arrays["pred_forces"], dtype=float).reshape(-1)
            idx = cat_counters[folder]
            cat_counters[folder] += 1

            data[model][folder]["ref"].append(ref_e)
            data[model][folder]["pred"].append(pred_e)
            data[model][folder]["labels"].append(str(idx))
            data[model][folder]["ref_forces"].extend(ref_f.tolist())
            data[model][folder]["pred_forces"].extend(pred_f.tolist())
            data[model][folder]["force_labels"].extend([str(idx)] * (3 * n_atoms))

            struct_dir = OUT_PATH / model / folder
            struct_dir.mkdir(parents=True, exist_ok=True)
            write(struct_dir / f"{idx}.xyz", atoms)

    return data


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_sp2_density.json",
    title="sp² bonded structures: graphene, graphite, fullerenes, nanotubes",
    x_label="Reference energy / eV atom⁻¹",
    y_label="Predicted energy / eV atom⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def sp2_density(all_energies: dict) -> dict[str, dict]:
    """
    Build density scatter inputs for sp2-bonded structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["sp2"]
        density_inputs[model] = {
            "ref": d["ref"],
            "pred": d["pred"],
            "meta": {"system_count": len(d["ref"])},
        }
        write_density_trajectories(
            labels_list=d["labels"],
            ref_vals=d["ref"],
            pred_vals=d["pred"],
            struct_dir=OUT_PATH / model / "sp2",
            traj_dir=OUT_PATH / model / "density_traj_sp2",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_sp3_density.json",
    title="sp³ bonded structures: diamond, high-pressure phases",
    x_label="Reference energy / eV atom⁻¹",
    y_label="Predicted energy / eV atom⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def sp3_density(all_energies: dict) -> dict[str, dict]:
    """
    Build density scatter inputs for sp3-bonded structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["sp3"]
        density_inputs[model] = {
            "ref": d["ref"],
            "pred": d["pred"],
            "meta": {"system_count": len(d["ref"])},
        }
        write_density_trajectories(
            labels_list=d["labels"],
            ref_vals=d["ref"],
            pred_vals=d["pred"],
            struct_dir=OUT_PATH / model / "sp3",
            traj_dir=OUT_PATH / model / "density_traj_sp3",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_amorphous_density.json",
    title="Amorphous and liquid carbon structures",
    x_label="Reference energy / eV atom⁻¹",
    y_label="Predicted energy / eV atom⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def amorphous_density(all_energies: dict) -> dict[str, dict]:
    """
    Build density scatter inputs for amorphous/liquid structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["amorphous"]
        density_inputs[model] = {
            "ref": d["ref"],
            "pred": d["pred"],
            "meta": {"system_count": len(d["ref"])},
        }
        write_density_trajectories(
            labels_list=d["labels"],
            ref_vals=d["ref"],
            pred_vals=d["pred"],
            struct_dir=OUT_PATH / model / "amorphous",
            traj_dir=OUT_PATH / model / "density_traj_amorphous",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_general_bulk_density.json",
    title="General bulk crystal structures (fcc, hcp, bcc, sc, A15, etc.)",
    x_label="Reference energy / eV atom⁻¹",
    y_label="Predicted energy / eV atom⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def general_bulk_density(all_energies: dict) -> dict[str, dict]:
    """
    Build density scatter inputs for general bulk structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["general_bulk"]
        density_inputs[model] = {
            "ref": d["ref"],
            "pred": d["pred"],
            "meta": {"system_count": len(d["ref"])},
        }
        write_density_trajectories(
            labels_list=d["labels"],
            ref_vals=d["ref"],
            pred_vals=d["pred"],
            struct_dir=OUT_PATH / model / "general_bulk",
            traj_dir=OUT_PATH / model / "density_traj_general_bulk",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_general_clusters_density.json",
    title="General carbon clusters (2–6 atoms, non-periodic)",
    x_label="Reference energy / eV atom⁻¹",
    y_label="Predicted energy / eV atom⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def general_clusters_density(all_energies: dict) -> dict[str, dict]:
    """
    Build density scatter inputs for general cluster structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["general_clusters"]
        density_inputs[model] = {
            "ref": d["ref"],
            "pred": d["pred"],
            "meta": {"system_count": len(d["ref"])},
        }
        write_density_trajectories(
            labels_list=d["labels"],
            ref_vals=d["ref"],
            pred_vals=d["pred"],
            struct_dir=OUT_PATH / model / "general_clusters",
            traj_dir=OUT_PATH / model / "density_traj_general_clusters",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_sp2_force_density.json",
    title="sp² bonded structures: force components",
    x_label="Reference force / eV Å⁻¹",
    y_label="Predicted force / eV Å⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def sp2_force_density(all_energies: dict) -> dict[str, dict]:
    """
    Build force density scatter inputs for sp2-bonded structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["sp2"]
        density_inputs[model] = {
            "ref": d["ref_forces"],
            "pred": d["pred_forces"],
            "meta": {"system_count": len(d["labels"])},
        }
        write_density_trajectories(
            labels_list=d["force_labels"],
            ref_vals=d["ref_forces"],
            pred_vals=d["pred_forces"],
            struct_dir=OUT_PATH / model / "sp2",
            traj_dir=OUT_PATH / model / "density_traj_sp2_force",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_sp3_force_density.json",
    title="sp³ bonded structures: force components",
    x_label="Reference force / eV Å⁻¹",
    y_label="Predicted force / eV Å⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def sp3_force_density(all_energies: dict) -> dict[str, dict]:
    """
    Build force density scatter inputs for sp3-bonded structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["sp3"]
        density_inputs[model] = {
            "ref": d["ref_forces"],
            "pred": d["pred_forces"],
            "meta": {"system_count": len(d["labels"])},
        }
        write_density_trajectories(
            labels_list=d["force_labels"],
            ref_vals=d["ref_forces"],
            pred_vals=d["pred_forces"],
            struct_dir=OUT_PATH / model / "sp3",
            traj_dir=OUT_PATH / model / "density_traj_sp3_force",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_amorphous_force_density.json",
    title="Amorphous and liquid carbon: force components",
    x_label="Reference force / eV Å⁻¹",
    y_label="Predicted force / eV Å⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def amorphous_force_density(all_energies: dict) -> dict[str, dict]:
    """
    Build force density scatter inputs for amorphous/liquid structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["amorphous"]
        density_inputs[model] = {
            "ref": d["ref_forces"],
            "pred": d["pred_forces"],
            "meta": {"system_count": len(d["labels"])},
        }
        write_density_trajectories(
            labels_list=d["force_labels"],
            ref_vals=d["ref_forces"],
            pred_vals=d["pred_forces"],
            struct_dir=OUT_PATH / model / "amorphous",
            traj_dir=OUT_PATH / model / "density_traj_amorphous_force",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_general_bulk_force_density.json",
    title="General bulk crystal structures: force components",
    x_label="Reference force / eV Å⁻¹",
    y_label="Predicted force / eV Å⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def general_bulk_force_density(all_energies: dict) -> dict[str, dict]:
    """
    Build force density scatter inputs for general bulk structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["general_bulk"]
        density_inputs[model] = {
            "ref": d["ref_forces"],
            "pred": d["pred_forces"],
            "meta": {"system_count": len(d["labels"])},
        }
        write_density_trajectories(
            labels_list=d["force_labels"],
            ref_vals=d["ref_forces"],
            pred_vals=d["pred_forces"],
            struct_dir=OUT_PATH / model / "general_bulk",
            traj_dir=OUT_PATH / model / "density_traj_general_bulk_force",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_general_clusters_force_density.json",
    title="General carbon clusters: force components",
    x_label="Reference force / eV Å⁻¹",
    y_label="Predicted force / eV Å⁻¹",
    annotation_metadata={"system_count": "Structures"},
)
def general_clusters_force_density(all_energies: dict) -> dict[str, dict]:
    """
    Build force density scatter inputs for general cluster structures.

    Parameters
    ----------
    all_energies
        Nested dict of calc results keyed by model then category folder.

    Returns
    -------
    dict[str, dict]
        Density scatter payload per model.
    """
    density_inputs: dict[str, dict] = {}
    for model in MODELS:
        d = all_energies[model]["general_clusters"]
        density_inputs[model] = {
            "ref": d["ref_forces"],
            "pred": d["pred_forces"],
            "meta": {"system_count": len(d["labels"])},
        }
        write_density_trajectories(
            labels_list=d["force_labels"],
            ref_vals=d["ref_forces"],
            pred_vals=d["pred_forces"],
            struct_dir=OUT_PATH / model / "general_clusters",
            traj_dir=OUT_PATH / model / "density_traj_general_clusters_force",
            struct_filename_builder=lambda label: f"{label}.xyz",
        )
    return density_inputs


@pytest.fixture
@build_table(
    filename=OUT_PATH / "diverse_carbon_structures_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(all_energies: dict) -> dict[str, dict]:
    """
    Compute per-category energy and force MAE for each model.

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

    force_metric_map = {
        "sp2": "sp2 bonded force MAE",
        "sp3": "sp3 bonded force MAE",
        "amorphous": "amorphous/liquid force MAE",
        "general_bulk": "general bulk force MAE",
        "general_clusters": "general clusters force MAE",
    }
    for folder, force_metric_name in force_metric_map.items():
        result[force_metric_name] = {
            model: mae(
                all_energies[model][folder]["ref_forces"],
                all_energies[model][folder]["pred_forces"],
            )
            for model in MODELS
        }
    return result


def test_diverse_carbon_structures(
    sp2_density: dict,
    sp3_density: dict,
    amorphous_density: dict,
    general_bulk_density: dict,
    general_clusters_density: dict,
    sp2_force_density: dict,
    sp3_force_density: dict,
    amorphous_force_density: dict,
    general_bulk_force_density: dict,
    general_clusters_force_density: dict,
    metrics: dict,
) -> None:
    """
    Run diverse carbon structures analysis (drives all fixtures).

    Parameters
    ----------
    sp2_density
        Energy density scatter inputs for sp2-bonded structures.
    sp3_density
        Energy density scatter inputs for sp3-bonded structures.
    amorphous_density
        Energy density scatter inputs for amorphous/liquid structures.
    general_bulk_density
        Energy density scatter inputs for general bulk structures.
    general_clusters_density
        Energy density scatter inputs for general cluster structures.
    sp2_force_density
        Force density scatter inputs for sp2-bonded structures.
    sp3_force_density
        Force density scatter inputs for sp3-bonded structures.
    amorphous_force_density
        Force density scatter inputs for amorphous/liquid structures.
    general_bulk_force_density
        Force density scatter inputs for general bulk structures.
    general_clusters_force_density
        Force density scatter inputs for general cluster structures.
    metrics
        Per-category energy and force MAE for each model.
    """
    return
