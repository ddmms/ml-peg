"""Analyse carbon binding curves against digitised PBE+D2 reference data."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.carbon.curve_metrics import (
    SHAPE_METRICS,
    curve_shape_metrics,
    reference_minimum,
)
from ml_peg.analysis.utils.decorators import build_table, plot_scatter
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

# Digitised PBE+D2 reference curves (energy in eV/atom relative to isolated atoms).
REFERENCE = json.loads((Path(__file__).with_name("reference.json")).read_text())

# Fixed display order and human-readable titles, applied to both the metric
# aggregation and the overlaid figure traces.
STRUCTURE_NAMES = ("dimer", "graphene", "diamond", "sc", "bcc", "fcc")
STRUCTURE_TITLES = {
    "dimer": "Carbon dimer",
    "graphene": "Graphene",
    "diamond": "Diamond",
    "sc": "Simple cubic",
    "bcc": "BCC",
    "fcc": "FCC",
}

# Metric columns: diatomics-style shape diagnostics plus minimum-location and
# minimum-depth errors against the reference curve.
METRIC_COLUMNS = list(SHAPE_METRICS) + ["Min distance error", "Min energy error"]

# Plotting window (crops the repulsive wall so curves stay legible). Metrics are
# computed on the full, unclipped curves; this only affects the figure.
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
        xyz_path = CALC_PATH / model_name / f"{structure}.extxyz"
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
    per_structure: list[dict[str, float]] = []
    for structure in STRUCTURE_NAMES:
        if structure not in curves:
            continue
        distances, energies = curves[structure]
        shape = curve_shape_metrics(distances, energies)
        if shape is None:
            continue

        ref = REFERENCE.get(structure)
        if ref is not None:
            r_min_ref, e_min_ref = reference_minimum(ref["x"], ref["y"])
            shape["Min distance error"] = abs(shape["r_min"] - r_min_ref)
            shape["Min energy error"] = abs(shape["e_min"] - e_min_ref)
        else:
            shape["Min distance error"] = np.nan
            shape["Min energy error"] = np.nan
        per_structure.append(shape)

    if not per_structure:
        return dict.fromkeys(METRIC_COLUMNS, np.nan)

    return {
        column: float(
            np.nanmean([metrics.get(column, np.nan) for metrics in per_structure])
        )
        for column in METRIC_COLUMNS
    }


def _clip_curve(
    distances: np.ndarray, energies: np.ndarray
) -> tuple[list[float], list[float]]:
    """
    Restrict a curve to the plotting window.

    Parameters
    ----------
    distances
        Scan coordinate (Angstrom).
    energies
        Shifted energies (eV/atom).

    Returns
    -------
    tuple[list[float], list[float]]
        Distances and energies within the display window (same order).
    """
    d = np.asarray(distances, dtype=float)
    e = np.asarray(energies, dtype=float)
    mask = (
        np.isfinite(d)
        & np.isfinite(e)
        & (d <= PLOT_X_MAX)
        & (e >= PLOT_E_MIN)
        & (e <= PLOT_E_MAX)
    )
    return list(d[mask]), list(e[mask])


def write_model_figure(
    model_name: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Write the overlaid binding-curve figure for a model.

    Trace order is fixed: the six model curves first (in ``STRUCTURE_NAMES``
    order), then the six PBE+D2 reference curves, so model and reference traces
    for the same structure line up in the legend.

    Parameters
    ----------
    model_name
        Model identifier.
    curves
        Per-structure (distances, energies) for the model.
    """
    model_dir = OUT_PATH / model_name

    clipped: dict[str, tuple[list[float], list[float]]] = {}
    for structure in STRUCTURE_NAMES:
        if structure not in curves:
            clipped[structure] = ([], [])
            continue
        distances, energies = curves[structure]
        clipped[structure] = _clip_curve(distances, energies)

    @plot_scatter(
        filename=model_dir / "figure_binding_curves.json",
        title=f"Carbon binding curves — {model_name}",
        x_label="C–C nearest-neighbour distance / Å",
        y_label="Energy / eV atom⁻¹",
        show_line=True,
    )
    def plot_curves() -> dict[str, list]:
        """
        Build the overlaid model and reference traces in the fixed order.

        Returns
        -------
        dict[str, list]
            Mapping of trace label to ``[xs, ys]``.
        """
        results: dict[str, list] = {}
        # Model curves first (curves 0..N-1).
        for structure in STRUCTURE_NAMES:
            x, y = clipped[structure]
            results[STRUCTURE_TITLES[structure]] = [x, y]
        # Reference curves next (curves N..2N-1).
        for structure in STRUCTURE_NAMES:
            ref = REFERENCE.get(structure)
            xs = list(ref["x"]) if ref else []
            ys = list(ref["y"]) if ref else []
            results[f"{STRUCTURE_TITLES[structure]} (PBE+D2)"] = [xs, ys]
        return results

    plot_curves()


@pytest.fixture
def curve_plots(model_curves) -> None:
    """
    Write per-model binding-curve figures.

    Parameters
    ----------
    model_curves
        Per-model, per-structure curve arrays.
    """
    for model_name, curves in model_curves.items():
        if curves:
            write_model_figure(model_name, curves)


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
