"""Analyse bilayer-graphene interlayer curves against a PBE+D2 reference."""

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
from ml_peg.analysis.utils.decorators import build_table, plot_scatter
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

REFERENCE = json.loads((CALC_PATH / "reference.json").read_text())
REF_D2 = REFERENCE["PBE+D2"]

EV_TO_MEV = 1000.0

METRIC_COLUMNS = list(SHAPE_METRICS) + ["Min distance error", "Min energy error"]

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
    xyz_path = CALC_PATH / model_name / "interlayer.xyz"
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
    result = single_curve_metrics_with_ref(
        separations, energies, REF_D2["x"], REF_D2["y"], METRIC_COLUMNS
    )
    return result if result is not None else dict.fromkeys(METRIC_COLUMNS, np.nan)


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
    model_dir = OUT_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    x_model, y_model, _ = clip_curve(
        separations,
        energies,
        x_min=PLOT_X_MIN,
        x_max=PLOT_X_MAX,
        e_min=PLOT_E_MIN,
        e_max=PLOT_E_MAX,
    )

    @plot_scatter(
        filename=model_dir / "figure_interlayer.json",
        title=f"Bilayer graphene interlayer curve — {model_name}",
        x_label="Interlayer separation / Å",
        y_label="Energy / meV atom⁻¹",
        show_line=True,
        show_markers=True,
        reference_mode="lines",
        reference_line_dash="dash",
    )
    def _plot():
        """
        Return the model and reference curves for the scatter decorator.

        Returns
        -------
        dict[str, tuple[list, list]]
            Model and ``"ref"`` traces as (x, y) tuples.
        """
        return {
            model_name: (x_model, y_model),
            "ref": (list(REF_D2["x"]), list(REF_D2["y"])),
        }

    _plot()


def write_model_structs(
    model_name: str, separations: np.ndarray, energies: np.ndarray
) -> None:
    """
    Write a trajectory file for WEAS, aligned to the scatter plot.

    Frames are ordered and clipped to match the figure so ``pointNumber`` maps
    directly to the frame index.

    Parameters
    ----------
    model_name
        Model identifier.
    separations
        Sorted interlayer separations (Angstrom).
    energies
        Relative energies (meV/atom).
    """
    xyz_path = CALC_PATH / model_name / "interlayer.xyz"
    if not xyz_path.exists():
        return

    frames = read(xyz_path, index=":")
    raw_seps = np.array([float(f.info["interlayer_separation"]) for f in frames])
    sorted_frames = [frames[i] for i in np.argsort(raw_seps)]

    _, _, mask = clip_curve(
        separations,
        energies,
        x_min=PLOT_X_MIN,
        x_max=PLOT_X_MAX,
        e_min=PLOT_E_MIN,
        e_max=PLOT_E_MAX,
    )
    traj = [sorted_frames[i].copy() for i in np.where(mask)[0]]
    for f in traj:
        f.calc = None

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ase_write(out_dir / "interlayer.extxyz", traj, format="extxyz")


@pytest.fixture
def curve_plots(model_curves) -> None:
    """
    Write per-model interlayer figures and trajectory files for WEAS.

    Parameters
    ----------
    model_curves
        Per-model (separations, energies) arrays.
    """
    for model_name, (separations, energies) in model_curves.items():
        if separations.size and np.isfinite(energies).any():
            write_model_figure(model_name, separations, energies)
            write_model_structs(model_name, separations, energies)


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
