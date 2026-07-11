"""Analyse bilayer-graphene interlayer curves against a PBE+D2 reference."""

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
from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

CATEGORY = "carbon"
BENCHMARK = "graphene_interlayer"

CALC_PATH = CALCS_ROOT / CATEGORY / BENCHMARK / "outputs"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK

MODELS = get_model_names(current_models)

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Digitised reference curves (relative energy in meV/atom).
REFERENCE = json.loads((Path(__file__).with_name("reference.json")).read_text())
# Models are evaluated with dispersion, so the PBE+D2 curve is the scoring target.
REF_D2 = REFERENCE["PBE+D2"]

EV_TO_MEV = 1000.0

METRIC_COLUMNS = list(SHAPE_METRICS) + ["Min distance error", "Min energy error"]

# Plotting window (matches the physically relevant interlayer range).
PLOT_X_MIN = 2.0
PLOT_X_MAX = 5.5
PLOT_E_MIN = -100.0
PLOT_E_MAX = 200.0

OUT_PATH.mkdir(parents=True, exist_ok=True)


def load_model_curve(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the interlayer curve for one model in meV/atom, relative to large d.

    Parameters
    ----------
    model_name
        Model identifier.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Interlayer separations (Angstrom) and relative energies (meV/atom).
    """
    xyz_path = CALC_PATH / model_name / "interlayer.extxyz"
    if not xyz_path.exists():
        return np.array([]), np.array([])

    frames = read(xyz_path, index=":")
    separations = np.array([float(f.info["interlayer_separation"]) for f in frames])
    energies = np.array([float(f.info["energy_per_atom"]) for f in frames])

    order = np.argsort(separations)
    separations = separations[order]
    energies = energies[order]

    finite = energies[np.isfinite(energies)]
    if finite.size == 0:
        return separations, energies
    # Reference to the largest separation, then convert eV -> meV per atom.
    energies = (energies - finite[-1]) * EV_TO_MEV
    return separations, energies


@pytest.fixture
def model_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load interlayer curves for every model.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Mapping of model name to (separations, relative energies).
    """
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODELS:
        separations, energies = load_model_curve(model_name)
        if separations.size == 0:
            warn(
                f"No outputs for model {model_name}; metrics set to NaN.",
                stacklevel=2,
            )
        curves[model_name] = (separations, energies)
    return curves


def compute_model_metrics(
    separations: np.ndarray, energies: np.ndarray
) -> dict[str, float]:
    """
    Compute interlayer metrics for one model.

    Parameters
    ----------
    separations
        Interlayer separations (Angstrom).
    energies
        Relative energies (meV/atom).

    Returns
    -------
    dict[str, float]
        Metric values (NaN where unavailable).
    """
    shape = curve_shape_metrics(separations, energies)
    if shape is None:
        return dict.fromkeys(METRIC_COLUMNS, np.nan)

    r_min_ref, e_min_ref = reference_minimum(REF_D2["x"], REF_D2["y"])
    shape["Min distance error"] = abs(shape["r_min"] - r_min_ref)
    shape["Min energy error"] = abs(shape["e_min"] - e_min_ref)
    return {column: shape.get(column, np.nan) for column in METRIC_COLUMNS}


def _clip_curve(
    separations: np.ndarray, energies: np.ndarray
) -> tuple[list[float], list[float]]:
    """
    Restrict a curve to the plotting window.

    Parameters
    ----------
    separations
        Interlayer separations (Angstrom).
    energies
        Relative energies (meV/atom).

    Returns
    -------
    tuple[list[float], list[float]]
        Separations and energies within the display window.
    """
    d = np.asarray(separations, dtype=float)
    e = np.asarray(energies, dtype=float)
    mask = (
        np.isfinite(d)
        & np.isfinite(e)
        & (d >= PLOT_X_MIN)
        & (d <= PLOT_X_MAX)
        & (e >= PLOT_E_MIN)
        & (e <= PLOT_E_MAX)
    )
    return list(d[mask]), list(e[mask])


def write_model_figure(
    model_name: str, separations: np.ndarray, energies: np.ndarray
) -> None:
    """
    Write the interlayer curve figure for one model.

    Parameters
    ----------
    model_name
        Model identifier.
    separations
        Interlayer separations (Angstrom).
    energies
        Relative energies (meV/atom).
    """
    from ml_peg.analysis.utils.decorators import plot_scatter

    model_dir = OUT_PATH / model_name

    @plot_scatter(
        filename=model_dir / "figure_interlayer.json",
        title=f"Bilayer graphene interlayer curve — {model_name}",
        x_label="Interlayer separation / Å",
        y_label="Energy / meV atom⁻¹",
        show_line=True,
        show_markers=False,
    )
    def plot_curve() -> dict[str, list]:
        """
        Build the model and reference traces.

        Returns
        -------
        dict[str, list]
            Mapping of trace label to ``[xs, ys]``.
        """
        return {
            model_name: list(_clip_curve(separations, energies)),
            "PBE+D2": [list(REF_D2["x"]), list(REF_D2["y"])],
            "PBE": [list(REFERENCE["PBE"]["x"]), list(REFERENCE["PBE"]["y"])],
        }

    plot_curve()


@pytest.fixture
def curve_plots(model_curves) -> None:
    """
    Write per-model interlayer figures.

    Parameters
    ----------
    model_curves
        Per-model (separations, energies) arrays.
    """
    for model_name, (separations, energies) in model_curves.items():
        if separations.size and np.isfinite(energies).any():
            write_model_figure(model_name, separations, energies)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "graphene_interlayer_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(model_curves) -> dict[str, dict]:
    """
    Compute interlayer metrics for all models.

    Parameters
    ----------
    model_curves
        Per-model (separations, energies) arrays.

    Returns
    -------
    dict[str, dict]
        Mapping of metric name to per-model values.
    """
    per_model = {
        model_name: compute_model_metrics(separations, energies)
        for model_name, (separations, energies) in model_curves.items()
    }
    metrics_dict: dict[str, dict[str, float | None]] = {}
    for column in METRIC_COLUMNS:
        metrics_dict[column] = {
            model_name: (values[column] if np.isfinite(values[column]) else None)
            for model_name, values in per_model.items()
        }
    return metrics_dict


def test_graphene_interlayer(metrics: dict[str, dict], curve_plots: None) -> None:
    """
    Run the interlayer analysis and write element info for filtering.

    Parameters
    ----------
    metrics
        Metrics table payload.
    curve_plots
        Per-model figures (executed for their side effects).
    """
    with open(OUT_PATH / "info.json", "w") as f:
        json.dump({"elements": ["C"]}, f, indent=1)
