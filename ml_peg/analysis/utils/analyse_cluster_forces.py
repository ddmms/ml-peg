"""Shared analysis helpers for the small-cluster force benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from warnings import warn

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import get_struct_info, mae, sample_density_grid
from ml_peg.app.utils.utils import Thresholds
from ml_peg.calcs.utils.cluster_forces import (
    OUTPUT_FILENAME,
    PRED_FORCES_KEY,
    REF_FORCES_KEY,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CLUSTER_SIZES = (3, 4, 5, 6, 7, 8)

# Maximum clusters shown for each density point: the clicked point's cluster,
# followed by others from the same density cell. Dense cells can contain most
# of the 10,000 clusters of a given size.
MAX_FRAMES_PER_POINT = 5

AXES = ("x", "y", "z")


def metric_name(cluster_size: int) -> str:
    """
    Get the table column name for a cluster size.

    Parameters
    ----------
    cluster_size
        Number of atoms in the cluster.

    Returns
    -------
    str
        Metric name.
    """
    return f"Force MAE ({cluster_size} atoms)"


def figure_filename(cluster_size: int) -> str:
    """
    Get the density plot filename for a cluster size.

    Parameters
    ----------
    cluster_size
        Number of atoms in the cluster.

    Returns
    -------
    str
        Figure filename.
    """
    return f"figure_force_parity_{cluster_size}atoms.json"


def structs_filename(cluster_size: int) -> str:
    """
    Get the filename of the clusters of one size, shared by all models.

    Parameters
    ----------
    cluster_size
        Number of atoms in the cluster.

    Returns
    -------
    str
        Structure filename.
    """
    return f"clusters_{cluster_size}atoms.extxyz"


def frame_groups_filename(cluster_size: int) -> str:
    """
    Get the per-model filename of the frames shown for each density point.

    Parameters
    ----------
    cluster_size
        Number of atoms in the cluster.

    Returns
    -------
    str
        Frame groups filename.
    """
    return f"frame_groups_{cluster_size}atoms.json"


@dataclass
class SizeResults:
    """
    Force components of one model for all clusters of one size.

    Attributes
    ----------
    ref
        Reference force components of all clusters with converged references.
    pred
        Predicted force components, matching ``ref``. Contains NaN for failures.
    frames
        Index of each component's cluster in ``structs_filename(size)``.
    labels
        Hover label of each component.
    n_clusters
        Number of clusters with converged references.
    n_failed
        Number of those clusters for which the prediction failed.
    """

    ref: np.ndarray
    pred: np.ndarray
    frames: np.ndarray
    labels: list[str] = field(repr=False)
    n_clusters: int
    n_failed: int

    @property
    def mae(self) -> float:
        """
        Get the force MAE, which is NaN if any prediction failed.

        Returns
        -------
        float
            Component-wise mean absolute error.
        """
        # Infinite predictions would otherwise raise in the MAE calculation
        if not self.finite.all():
            return np.nan
        return float(mae(self.ref, self.pred))

    @property
    def finite(self) -> np.ndarray:
        """
        Get a mask of components with a finite prediction.

        Returns
        -------
        np.ndarray
            Boolean mask over components.
        """
        return np.isfinite(self.pred)


def _component_labels(atoms: Atoms) -> list[str]:
    """
    Build hover labels for every force component of a cluster.

    Parameters
    ----------
    atoms
        Cluster structure.

    Returns
    -------
    list[str]
        Labels of the form ``"LiH2: Li0 x"``, in flattened force order.
    """
    formula = atoms.get_chemical_formula()
    return [
        f"{formula}: {symbol}{index} {axis}"
        for index, symbol in enumerate(atoms.get_chemical_symbols())
        for axis in AXES
    ]


def _read_outputs(path: Path, reference_structs: list[Atoms]) -> list[Atoms] | None:
    """
    Read a model's outputs, checking they match the reference clusters.

    Parameters
    ----------
    path
        Output file of the model.
    reference_structs
        Clusters from the mock calculation, defining the reference data.

    Returns
    -------
    list[Atoms] | None
        Evaluated clusters, or None if the model was not run.
    """
    if not path.exists():
        return None
    structs = read(path, index=":")
    if len(structs) != len(reference_structs) or any(
        not np.array_equal(atoms.numbers, ref.numbers)
        for atoms, ref in zip(structs, reference_structs, strict=True)
    ):
        raise ValueError(
            f"Clusters in {path} do not match the mock calculation. "
            "Please rerun the calculation."
        )
    return structs


def load_reference(calc_path: Path) -> tuple[list[Atoms], dict[int, list[int]]]:
    """
    Load the reference clusters from the mock calculation, which is always complete.

    Parameters
    ----------
    calc_path
        Directory containing ``[model]/cluster_forces.extxyz``.

    Returns
    -------
    tuple[list[Atoms], dict[int, list[int]]]
        All clusters, and for each size the indices of clusters whose reference
        calculation converged.
    """
    structs = read(calc_path / "mock" / OUTPUT_FILENAME, index=":")
    by_size = {
        size: [
            index
            for index, atoms in enumerate(structs)
            if len(atoms) == size and np.isfinite(atoms.arrays[REF_FORCES_KEY]).all()
        ]
        for size in CLUSTER_SIZES
    }
    return structs, by_size


def write_cluster_structs(
    reference_structs: list[Atoms],
    by_size: dict[int, list[int]],
    out_path: Path,
) -> None:
    """
    Write the geometries of the clusters of each size, shared by all models.

    Parameters
    ----------
    reference_structs
        All clusters.
    by_size
        Indices of clusters with converged references for each size.
    out_path
        Directory to write ``structs_filename(size)`` to.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    for size, indices in by_size.items():
        frames = []
        for index in indices:
            atoms = Atoms(
                numbers=reference_structs[index].numbers,
                positions=reference_structs[index].positions,
            )
            atoms.info["structure_index"] = index
            frames.append(atoms)
        if frames:
            write(out_path / structs_filename(size), frames)


def get_force_data(
    calc_path: Path,
    models: list[str],
    reference_structs: list[Atoms],
    by_size: dict[int, list[int]],
) -> dict[str, dict[int, SizeResults]]:
    """
    Collect reference and predicted force components by model and cluster size.

    Parameters
    ----------
    calc_path
        Directory containing ``[model]/cluster_forces.extxyz``.
    models
        Models to load.
    reference_structs
        All clusters, with reference forces, from ``load_reference``.
    by_size
        Indices of clusters with converged references for each size.

    Returns
    -------
    dict[str, dict[int, SizeResults]]
        Force components by model and cluster size, for models that were run.
    """
    ref = {
        size: np.concatenate(
            [reference_structs[i].arrays[REF_FORCES_KEY].ravel() for i in indices]
        )
        for size, indices in by_size.items()
        if indices
    }
    labels = {
        size: [
            label
            for index in by_size[size]
            for label in _component_labels(reference_structs[index])
        ]
        for size in ref
    }

    results: dict[str, dict[int, SizeResults]] = {}
    for model in models:
        path = calc_path / model / OUTPUT_FILENAME
        try:
            structs = _read_outputs(path, reference_structs)
        except Exception as exc:
            # Treat every cluster as failed, rather than scoring a partial output
            warn(
                f"{model}: could not use {path} ({exc}). All of its cluster force "
                "MAEs are set to NaN.",
                stacklevel=2,
            )
            structs = []
        if structs is None:
            continue
        results[model] = {}
        for size in ref:
            pred = [
                structs[i].arrays[PRED_FORCES_KEY].ravel()
                if structs
                else np.full(3 * size, np.nan)
                for i in by_size[size]
            ]
            results[model][size] = SizeResults(
                ref=ref[size],
                pred=np.concatenate(pred).astype(float),
                frames=np.repeat(np.arange(len(by_size[size])), 3 * size),
                labels=labels[size],
                n_clusters=len(by_size[size]),
                n_failed=sum(not np.isfinite(forces).all() for forces in pred),
            )
    return results


def get_frame_groups(data: SizeResults) -> tuple[list[list[int]], list[int]]:
    """
    Choose the clusters shown when each plotted density point is clicked.

    Each group starts with the cluster of the plotted point, followed by up to
    ``MAX_FRAMES_PER_POINT - 1`` other clusters from the same density cell.

    Parameters
    ----------
    data
        Force components of one model and cluster size.

    Returns
    -------
    tuple[list[list[int]], list[int]]
        Frame indices for each plotted point, in plotting order, and the number of
        distinct clusters in the density cell of every finite component.
    """
    finite = data.finite
    frames = data.frames[finite]
    sampled, _, mapping = sample_density_grid(data.ref[finite], data.pred[finite])

    groups = []
    cell_clusters = [0] * len(frames)
    for own, members in zip(sampled, mapping, strict=True):
        cell_frames = list(dict.fromkeys(frames[[own, *members]].tolist()))
        for member in members:
            cell_clusters[member] = len(cell_frames)
        groups.append(cell_frames[:MAX_FRAMES_PER_POINT])
    return groups, cell_clusters


def analyse_cluster_forces(
    *,
    title: str,
    calc_path: Path,
    out_path: Path,
    thresholds: Thresholds,
    metric_tooltips: dict[str, str],
    weights: dict[str, float],
    mlip_name_map: dict[str, str] | None = None,
) -> None:
    """
    Write the metrics table, density plots, and structures for a benchmark.

    Parameters
    ----------
    title
        Name of the reference set, used in plot titles.
    calc_path
        Directory containing calculation outputs.
    out_path
        App data directory to write outputs to.
    thresholds
        Metric thresholds from ``metrics.yml``.
    metric_tooltips
        Metric tooltips from ``metrics.yml``.
    weights
        Metric weights from ``metrics.yml``.
    mlip_name_map
        Optional mapping of model names to display names in the table.
    """
    get_struct_info(
        calc_path=calc_path,
        glob_pattern=OUTPUT_FILENAME,
        write_info=True,
        write_structs=False,
        out_path=out_path,
    )
    reference_structs, by_size = load_reference(calc_path)
    write_cluster_structs(reference_structs, by_size, out_path)
    force_data = get_force_data(calc_path, MODELS, reference_structs, by_size)

    for size in CLUSTER_SIZES:
        size_data = {
            model: data[size] for model, data in force_data.items() if size in data
        }
        if not size_data:
            continue

        plot_inputs = {}
        for model, data in size_data.items():
            groups, cell_clusters = get_frame_groups(data)
            (out_path / model).mkdir(parents=True, exist_ok=True)
            (out_path / model / frame_groups_filename(size)).write_text(
                json.dumps(groups)
            )
            finite = data.finite
            plot_inputs[model] = {
                "ref": data.ref[finite],
                "pred": data.pred[finite],
                "meta": {
                    "n_clusters": data.n_clusters,
                    "n_failed": data.n_failed,
                },
                "point_meta": {
                    "component": [
                        label
                        for label, keep in zip(data.labels, finite, strict=True)
                        if keep
                    ],
                    "cell_clusters": cell_clusters,
                },
            }

        @plot_density_scatter(
            filename=out_path / figure_filename(size),
            title=f"{title}: {size}-atom cluster force components",
            x_label="Reference force / (eV/Å)",
            y_label="Predicted force / (eV/Å)",
            annotation_metadata={
                "n_clusters": "Clusters",
                "n_failed": "Failed predictions",
            },
            point_metadata={
                "component": "Component",
                "cell_clusters": "Clusters in cell",
            },
        )
        def plot(inputs: dict = plot_inputs) -> dict[str, dict]:
            """
            Plot predicted against reference force components.

            Parameters
            ----------
            inputs
                Density plot inputs for each model.

            Returns
            -------
            dict[str, dict]
                Density plot inputs for each model.
            """
            return inputs

        plot()

    @build_table(
        filename=out_path / "cluster_forces_metrics_table.json",
        metric_tooltips=metric_tooltips,
        thresholds=thresholds,
        weights=weights,
        mlip_name_map=mlip_name_map,
    )
    def metrics() -> dict[str, dict[str, float]]:
        """
        Get the force MAE of every model for each cluster size.

        Returns
        -------
        dict[str, dict[str, float]]
            Metric names and values for all models.
        """
        return {
            metric_name(size): {
                model: data[size].mae
                for model, data in force_data.items()
                if size in data
            }
            for size in CLUSTER_SIZES
        }

    metrics()
