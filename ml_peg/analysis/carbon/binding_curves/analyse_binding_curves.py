"""Analyse carbon binding curves against digitised PBE+D2 reference data."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase.io import read
from ase.io import write as ase_write
import numpy as np
import pytest

from ml_peg.analysis.carbon.curve_metrics import (
    SHAPE_METRICS,
    clip_curve,
    single_curve_metrics_with_ref,
)
from ml_peg.analysis.utils.decorators import build_table, plot_scatter_grouped
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

CATEGORY = "carbon"
BENCHMARK = "binding_curves"

CALC_PATH = CALCS_ROOT / CATEGORY / BENCHMARK / "outputs"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK

MODELS = get_model_names(current_models)

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

REFERENCE = json.loads((CALC_PATH / "reference.json").read_text())

STRUCTURE_NAMES = ("dimer", "graphene", "diamond", "sc", "bcc", "fcc")
STRUCTURE_TITLES = {
    "dimer": "Carbon dimer",
    "graphene": "Graphene",
    "diamond": "Diamond",
    "sc": "Simple cubic",
    "bcc": "BCC",
    "fcc": "FCC",
}

METRIC_COLUMNS = list(SHAPE_METRICS) + ["Min distance error", "Min energy error"]

PLOT_X_MAX = 4.5
PLOT_E_MIN = -12.0
PLOT_E_MAX = 8.0

OUT_PATH.mkdir(parents=True, exist_ok=True)


def load_model_structures(
    model_name: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load per-structure binding curves for one model.

    Parameters
    ----------
    model_name
        Model identifier.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Mapping of structure name to (distances, shifted energies), sorted by
        increasing distance. Energies are shifted so the largest-separation
        energy is zero.
    """
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for structure in STRUCTURE_NAMES:
        xyz_path = CALC_PATH / model_name / f"{structure}.xyz"
        if not xyz_path.exists():
            continue
        frames = read(xyz_path, index=":")
        distances = np.array([float(f.info["nn_distance"]) for f in frames])
        energies = np.array([float(f.info["energy_per_atom"]) for f in frames])

        order = np.argsort(distances)
        distances = distances[order]
        energies = energies[order]

        finite = energies[np.isfinite(energies)]
        if finite.size:
            energies = energies - finite[-1]
        curves[structure] = (distances, energies)
    return curves


@pytest.fixture
def model_curves() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Load binding curves for every model.

    Returns
    -------
    dict
        Mapping of model name to per-structure (distances, energies).
    """
    curves: dict[str, dict] = {}
    for model_name in MODELS:
        model = load_model_structures(model_name)
        if not model:
            warn(
                f"No outputs for model {model_name}; metrics set to NaN.",
                stacklevel=2,
            )
        curves[model_name] = model
    return curves


def compute_model_metrics(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """
    Aggregate metrics across all structures for one model.

    Parameters
    ----------
    curves
        Per-structure (distances, energies) for a single model.

    Returns
    -------
    dict[str, float]
        Mean of each metric across structures (NaN where unavailable).
    """
    per_structure = []
    for structure in STRUCTURE_NAMES:
        if structure not in curves:
            continue
        d, e = curves[structure]
        ref = REFERENCE.get(structure, {})
        result = single_curve_metrics_with_ref(
            d, e, ref.get("x"), ref.get("y"), METRIC_COLUMNS
        )
        if result is not None:
            per_structure.append(result)

    if not per_structure:
        return dict.fromkeys(METRIC_COLUMNS, np.nan)
    return {
        col: float(np.nanmean([m[col] for m in per_structure]))
        for col in METRIC_COLUMNS
    }


def write_model_figure(
    model_name: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Write the overlaid binding-curve figure for a model.

    Parameters
    ----------
    model_name
        Model identifier.
    curves
        Per-structure (distances, energies) for the model.
    """
    model_dir = OUT_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    @plot_scatter_grouped(
        groups=STRUCTURE_TITLES,
        filename=model_dir / "figure_binding_curves.json",
        title=f"Carbon binding curves — {model_name}",
        x_label="C-C nearest-neighbour distance / Å",
        y_label="Energy / eV atom⁻¹",
    )
    def _plot():
        """
        Return per-structure model and reference curves for the decorator.

        Returns
        -------
        dict[str, tuple[list, list]]
            Model curves keyed by structure name and reference curves keyed by
            ``"{structure}_ref"``, each as (x, y) tuples.
        """
        results = {}
        for structure in STRUCTURE_NAMES:
            if structure in curves:
                cx, cy, _ = clip_curve(
                    *curves[structure],
                    x_max=PLOT_X_MAX,
                    e_min=PLOT_E_MIN,
                    e_max=PLOT_E_MAX,
                )
            else:
                cx, cy = [], []
            results[structure] = (cx, cy)
            ref = REFERENCE.get(structure, {})
            results[f"{structure}_ref"] = (
                list(ref.get("x", [])),
                list(ref.get("y", [])),
            )
        return results

    _plot()


def write_model_structs(
    model_name: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Write one extxyz trajectory per structure for WEAS trajectory viewing.

    Frames are ordered and clipped to match the scatter plot so ``pointNumber``
    maps directly to the frame index in each trajectory file.

    Parameters
    ----------
    model_name
        Model identifier.
    curves
        Per-structure (distances, energies) for the model.
    """
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for structure in STRUCTURE_NAMES:
        xyz_path = CALC_PATH / model_name / f"{structure}.xyz"
        if not xyz_path.exists():
            continue

        frames = read(xyz_path, index=":")
        dists = np.array([float(f.info["nn_distance"]) for f in frames])
        sorted_frames = [frames[i] for i in np.argsort(dists)]

        distances, energies = curves.get(structure, (np.array([]), np.array([])))
        _, _, mask = clip_curve(
            distances,
            energies,
            x_max=PLOT_X_MAX,
            e_min=PLOT_E_MIN,
            e_max=PLOT_E_MAX,
        )
        traj = [sorted_frames[i].copy() for i in np.where(mask)[0]]
        for f in traj:
            f.calc = None
        ase_write(out_dir / f"{structure}.extxyz", traj, format="extxyz")


@pytest.fixture
def curve_plots(model_curves) -> None:
    """
    Write per-model binding-curve figures and trajectory files for WEAS.

    Parameters
    ----------
    model_curves
        Per-model, per-structure curve arrays.
    """
    for model_name, curves in model_curves.items():
        if curves:
            write_model_figure(model_name, curves)
            write_model_structs(model_name, curves)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "binding_curves_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(model_curves) -> dict[str, dict]:
    """
    Compute aggregated binding-curve metrics for all models.

    Parameters
    ----------
    model_curves
        Per-model, per-structure curve arrays.

    Returns
    -------
    dict[str, dict]
        Mapping of metric name to per-model values.
    """
    per_model = {
        model_name: compute_model_metrics(curves)
        for model_name, curves in model_curves.items()
    }
    metrics_dict: dict[str, dict[str, float | None]] = {}
    for column in METRIC_COLUMNS:
        metrics_dict[column] = {
            model_name: (values[column] if np.isfinite(values[column]) else None)
            for model_name, values in per_model.items()
        }
    return metrics_dict


def test_binding_curves(metrics: dict[str, dict], curve_plots: None) -> None:
    """
    Run the binding-curve analysis and write element info for filtering.

    Parameters
    ----------
    metrics
        Metrics table payload.
    curve_plots
        Per-model figures and trajectories (executed for their side effects).
    """
    with open(OUT_PATH / "info.json", "w") as f:
        json.dump({"elements": ["C"]}, f, indent=1)
