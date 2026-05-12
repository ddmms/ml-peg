"""Analyse small-cluster force predictions."""

from __future__ import annotations

from pathlib import Path

from ase.atoms import Atoms
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
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


def _metric_name(cluster_size: int) -> str:
    """
    Return the metric name for a cluster size.

    Parameters
    ----------
    cluster_size
        Number of atoms in the cluster.

    Returns
    -------
    str
        Metric label.
    """
    return f"Force MAE ({cluster_size} atoms)"


def _load_structs(model: str) -> list[Atoms] | None:
    """
    Load evaluated clusters for a model.

    Parameters
    ----------
    model
        Model identifier.

    Returns
    -------
    list[ase.atoms.Atoms] | None
        Evaluated clusters, or ``None`` when no output exists.
    """
    path = CALC_PATH / model / BENCHMARK_FILENAME
    if not path.exists():
        return None
    structs = read(path, index=":")
    if not isinstance(structs, list) or not structs:
        raise ValueError(f"Unexpected output content: {path}")
    return structs


def _forces(structs: list[Atoms]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract flattened reference and predicted force components.

    Parameters
    ----------
    structs
        Evaluated clusters.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Flattened reference and predicted force components.
    """
    ref = np.concatenate(
        [
            np.asarray(atoms.arrays["ref_forces"], dtype=float).reshape(-1)
            for atoms in structs
        ]
    )
    pred = np.concatenate(
        [
            np.asarray(atoms.arrays["pred_forces"], dtype=float).reshape(-1)
            for atoms in structs
        ]
    )
    return ref, pred


def _group_by_cluster_size(structs: list[Atoms]) -> dict[int, list[Atoms]]:
    """
    Group evaluated clusters by atom count.

    Parameters
    ----------
    structs
        Evaluated clusters.

    Returns
    -------
    dict[int, list[ase.atoms.Atoms]]
        Clusters grouped by number of atoms.
    """
    grouped: dict[int, list[Atoms]] = {size: [] for size in CLUSTER_SIZES}
    for atoms in structs:
        cluster_size = len(atoms)
        if cluster_size in grouped:
            grouped[cluster_size].append(atoms)
    return grouped


@pytest.fixture
def force_data() -> dict[str, dict[int, dict[str, object]]]:
    """
    Load force components for all available model outputs, split by cluster size.

    Returns
    -------
    dict[str, dict[int, dict[str, object]]]
        Mapping from model name and cluster size to flattened force arrays and metadata.
    """
    results: dict[str, dict[int, dict[str, object]]] = {}
    for model in MODELS:
        structs = _load_structs(model)
        if structs is None:
            continue

        model_results: dict[int, dict[str, object]] = {}
        for cluster_size, grouped_structs in _group_by_cluster_size(structs).items():
            if not grouped_structs:
                continue

            ref, pred = _forces(grouped_structs)
            reference_targets = sorted(
                {
                    str(atoms.info.get("reference_target", "unknown"))
                    for atoms in grouped_structs
                }
            )
            model_results[cluster_size] = {
                "ref": ref,
                "pred": pred,
                "reference_target": ", ".join(reference_targets),
                "n_clusters": len(grouped_structs),
                "n_components": int(ref.size),
                "cluster_size": cluster_size,
            }
        if model_results:
            results[model] = model_results
    return results


@pytest.fixture
def force_mae(
    force_data: dict[str, dict[int, dict[str, object]]],
) -> dict[str, dict[str, float]]:
    """
    Calculate component-wise force MAE by cluster size for all available models.

    Parameters
    ----------
    force_data
        Flattened force data by model and cluster size.

    Returns
    -------
    dict[str, dict[str, float]]
        Force MAE by cluster-size metric and model.
    """
    results = {_metric_name(size): {} for size in CLUSTER_SIZES}
    for model, model_data in force_data.items():
        for cluster_size, data in model_data.items():
            results[_metric_name(cluster_size)][model] = mae(data["ref"], data["pred"])
    return results


def _write_force_parity_plot(
    cluster_size: int,
    force_data: dict[str, dict[int, dict[str, object]]],
) -> dict[str, dict]:
    """
    Write force parity data for a cluster size.

    Parameters
    ----------
    cluster_size
        Number of atoms in the clusters to plot.
    force_data
        Flattened force data by model and cluster size.

    Returns
    -------
    dict[str, dict]
        Density-scatter input by model for the requested cluster size.
    """
    data_for_size = {
        model: model_data[cluster_size]
        for model, model_data in force_data.items()
        if cluster_size in model_data
    }

    @plot_density_scatter(
        filename=OUT_PATH / f"figure_force_parity_{cluster_size}mer.json",
        title=f"{cluster_size}-atom cluster force components",
        x_label="Reference force / (eV/A)",
        y_label="Predicted force / (eV/A)",
        annotation_metadata={
            "reference_target": "Reference",
            "n_clusters": "Clusters",
            "n_components": "Components",
        },
    )
    def plot() -> dict[str, dict]:
        """
        Build density-scatter input for the requested cluster size.

        Returns
        -------
        dict[str, dict]
            Density-scatter input by model.
        """
        return {
            model: {
                "ref": data["ref"],
                "pred": data["pred"],
                "meta": {
                    "reference_target": data["reference_target"],
                    "n_clusters": data["n_clusters"],
                    "n_components": data["n_components"],
                },
            }
            for model, data in data_for_size.items()
        }

    return plot()


@pytest.fixture
def force_parity_plots(
    force_data: dict[str, dict[int, dict[str, object]]],
) -> dict[int, dict[str, dict]]:
    """
    Write force parity plots for each cluster size.

    Parameters
    ----------
    force_data
        Flattened force data by model and cluster size.

    Returns
    -------
    dict[int, dict[str, dict]]
        Plot input data by cluster size.
    """
    return {
        cluster_size: _write_force_parity_plot(cluster_size, force_data)
        for cluster_size in CLUSTER_SIZES
        if any(cluster_size in model_data for model_data in force_data.values())
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
    force_parity_plots: dict[int, dict[str, dict]],
) -> dict[str, dict]:
    """
    Build the benchmark table JSON.

    Parameters
    ----------
    force_mae
        Component-wise force MAE by cluster-size metric and model.
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
