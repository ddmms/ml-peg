"""Analyse small-cluster force predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    load_metrics_config,
    mae,
    sample_density_grid,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

BENCHMARK_DIR = "cluster_forces"
BENCHMARK_FILENAME = "cluster_forces.extxyz"
CLUSTER_SIZES = (3, 4, 5, 6, 7, 8)
CALC_PATH = CALCS_ROOT / "clusters" / BENCHMARK_DIR / "outputs"
OUT_PATH = APP_ROOT / "data" / "clusters" / BENCHMARK_DIR

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


REFERENCES = (
    {
        "key": "mad2",
        "label": "MAD2",
        "force_key": "mad2_ref_forces",
        "level_of_theory": "r2SCAN",
    },
    {
        "key": "omol25",
        "label": "OMOL25",
        "force_key": "omol25_ref_forces",
        "level_of_theory": "ωB97M-V/def2-TZVPD",
    },
)
PRED_FORCES_KEY = "pred_forces"


def _metric_name(reference: dict[str, str], cluster_size: int) -> str:
    # build metric names consistently with metrics.yml and the app
    return f"{reference['label']} Force MAE ({cluster_size} atoms)"


def _load_structs(model: str) -> list[Atoms] | None:
    # load evaluated clusters for a model, if present
    path = CALC_PATH / model / BENCHMARK_FILENAME
    if not path.exists():
        return None
    structs = read(path, index=":")
    if not isinstance(structs, list) or not structs:
        raise ValueError(f"Unexpected output content: {path}")
    return structs


def _group_by_cluster_size(structs: list[Atoms]) -> dict[int, list[Atoms]]:
    # group evaluated clusters by atom count
    grouped: dict[int, list[Atoms]] = {size: [] for size in CLUSTER_SIZES}
    for atoms in structs:
        cluster_size = len(atoms)
        if cluster_size in grouped:
            grouped[cluster_size].append(atoms)
    return grouped


def _force_components(
    structs: list[Atoms],
    reference: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[Atoms], int]:
    # extract finite force components and remember source structures for plotting
    ref_components: list[np.ndarray] = []
    pred_components: list[np.ndarray] = []
    component_structs: list[Atoms] = []
    excluded_components = 0

    for atoms in structs:
        ref_forces = np.asarray(atoms.arrays[reference["force_key"]], dtype=float)
        ref_flat = ref_forces.reshape(-1)
        pred_flat = np.asarray(atoms.arrays[PRED_FORCES_KEY], dtype=float).reshape(-1)
        finite_mask = np.isfinite(ref_flat) & np.isfinite(pred_flat)

        excluded_components += int((~finite_mask).sum())
        if not finite_mask.any():
            continue

        ref_components.append(ref_flat[finite_mask])
        pred_components.append(pred_flat[finite_mask])
        component_structs.extend([atoms] * int(finite_mask.sum()))

    if not ref_components:
        return np.array([]), np.array([]), [], excluded_components

    return (
        np.concatenate(ref_components),
        np.concatenate(pred_components),
        component_structs,
        excluded_components,
    )


def _write_density_trajectories(
    *,
    ref_vals: np.ndarray,
    pred_vals: np.ndarray,
    component_structs: list[Atoms],
    traj_dir: Path,
) -> None:
    # write one extxyz trajectory per sampled density point for structure viewing
    if len(ref_vals) == 0 or len(pred_vals) == 0:
        return

    _, _, sampled_mapping = sample_density_grid(ref_vals, pred_vals)
    traj_dir.mkdir(parents=True, exist_ok=True)
    for point_idx, source_indices in enumerate(sampled_mapping):
        frames = [component_structs[source_idx] for source_idx in source_indices]
        write(traj_dir / f"{point_idx}.extxyz", frames)


@pytest.fixture
def force_data() -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """
    Load force components for all available model outputs.

    Returns
    -------
    dict[str, dict[str, dict[int, dict[str, Any]]]]
        Mapping from model name, reference key, and cluster size to force arrays and
        metadata.
    """
    results: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for model in MODELS:
        structs = _load_structs(model)
        if structs is None:
            continue

        grouped_structs = _group_by_cluster_size(structs)
        model_results: dict[str, dict[int, dict[str, Any]]] = {
            reference["key"]: {} for reference in REFERENCES
        }
        for reference in REFERENCES:
            for cluster_size, structs_for_size in grouped_structs.items():
                if not structs_for_size:
                    continue

                ref, pred, component_structs, excluded_components = _force_components(
                    structs_for_size, reference
                )
                if ref.size == 0:
                    continue

                model_results[reference["key"]][cluster_size] = {
                    "ref": ref,
                    "pred": pred,
                    "component_structs": component_structs,
                    "excluded_components": excluded_components,
                    "n_clusters": len(structs_for_size),
                    "n_components": int(ref.size),
                    "reference": reference["label"],
                    "level_of_theory": reference["level_of_theory"],
                }

        if any(model_results[reference["key"]] for reference in REFERENCES):
            results[model] = model_results
    return results


@pytest.fixture
def force_mae(
    force_data: dict[str, dict[str, dict[int, dict[str, Any]]]],
) -> dict[str, dict[str, float]]:
    """
    Calculate component-wise force MAE by reference and cluster size.

    Parameters
    ----------
    force_data
        Flattened force data by model, reference, and cluster size.

    Returns
    -------
    dict[str, dict[str, float]]
        Force MAE by metric and model.
    """
    results = {
        _metric_name(reference, size): {}
        for reference in REFERENCES
        for size in CLUSTER_SIZES
    }
    for model, model_data in force_data.items():
        for reference in REFERENCES:
            for cluster_size, data in model_data.get(reference["key"], {}).items():
                results[_metric_name(reference, cluster_size)][model] = float(
                    mae(data["ref"], data["pred"])
                )
    return results


def _write_force_parity_plot(
    reference: dict[str, str],
    cluster_size: int,
    force_data: dict[str, dict[str, dict[int, dict[str, Any]]]],
) -> dict[str, dict]:
    # write force parity plot data and sampled structure trajectories
    data_for_plot = {
        model: model_data[reference["key"]][cluster_size]
        for model, model_data in force_data.items()
        if cluster_size in model_data.get(reference["key"], {})
    }

    @plot_density_scatter(
        filename=OUT_PATH
        / f"figure_force_parity_{reference['key']}_{cluster_size}atoms.json",
        title=f"{reference['label']} {cluster_size}-atom cluster force components",
        x_label="Reference force / (eV/A)",
        y_label="Predicted force / (eV/A)",
        annotation_metadata={
            "level_of_theory": "Level",
            "n_clusters": "Clusters",
            "n_components": "Components",
            "excluded_components": "Excluded components",
        },
    )
    def plot() -> dict[str, dict]:
        # build density-scatter input
        return {
            model: {
                "ref": data["ref"],
                "pred": data["pred"],
                "meta": {
                    "level_of_theory": data["level_of_theory"],
                    "n_clusters": data["n_clusters"],
                    "n_components": data["n_components"],
                    "excluded_components": data["excluded_components"],
                },
            }
            for model, data in data_for_plot.items()
        }

    for model, data in data_for_plot.items():
        _write_density_trajectories(
            ref_vals=data["ref"],
            pred_vals=data["pred"],
            component_structs=data["component_structs"],
            traj_dir=OUT_PATH
            / model
            / "density_traj"
            / f"{reference['key']}_{cluster_size}atoms",
        )

    return plot()


@pytest.fixture
def force_parity_plots(
    force_data: dict[str, dict[str, dict[int, dict[str, Any]]]],
) -> dict[str, dict[str, dict]]:
    """
    Write force parity plots for each reference and cluster size.

    Parameters
    ----------
    force_data
        Flattened force data by model, reference, and cluster size.

    Returns
    -------
    dict[str, dict[str, dict]]
        Plot input data by metric name.
    """
    return {
        _metric_name(reference, cluster_size): _write_force_parity_plot(
            reference, cluster_size, force_data
        )
        for reference in REFERENCES
        for cluster_size in CLUSTER_SIZES
        if any(
            cluster_size in model_data.get(reference["key"], {})
            for model_data in force_data.values()
        )
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "cluster_forces_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    force_mae: dict[str, dict[str, float]],
    force_parity_plots: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """
    Build the benchmark table JSON.

    Parameters
    ----------
    force_mae
        Component-wise force MAE by metric and model.
    force_parity_plots
        Force parity plot data; included to ensure the figures are written.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return force_mae


def test_cluster_forces(metrics: dict[str, dict]) -> None:
    """
    Run analysis for the cluster-force benchmark.

    Parameters
    ----------
    metrics
        Benchmark metrics table.
    """
    return
