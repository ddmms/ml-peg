"""Analyse the Landscape17 stationary point results and emit metrics.

Reads the raw per-model outputs written by ``calc_stationary_points.py`` and
derives the quantities declared in ``metrics.yml``. Derived properties are
computed here rather than in the calculation, so the raw outputs stay reusable
if a metric is later redefined.

Distributions are summarised by percentile rather than by the fraction crossing
a threshold; see the calculation module's docstring for why.

Scope
-----
Only the energy and force metrics are implemented, because only the energy and
force calculations exist. The drift, connectivity and Hessian metrics are held
in ``metrics.yml`` as a commented block and move into it as their calculations
land, so the file never declares a metric that nothing produces.
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_violin
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map()

EV_TO_MEV = 1000.0

CALC_PATH = CALCS_ROOT / "landscapes" / "stationary_points" / "outputs"
OUT_PATH = APP_ROOT / "data" / "landscapes" / "stationary_points"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

POINTS_GLOB = "*_stationary_points.extxyz"
PATHS_GLOB = "*_paths.extxyz"

# Labels for the stationary points, read from the mock outputs so that hover
# data lines up with the per-structure value lists. The path frames are
# deliberately not read this way: there are 11,173 of them, and writing them
# all out for the app would cost far more than the plot is worth.
INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern=POINTS_GLOB,
    info_keys=["molecule", "kind", "label"],
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


def _read_frames(model_name: str, pattern: str) -> list:
    """
    Read one model's outputs, in the order ``get_struct_info`` used.

    Parameters
    ----------
    model_name
        Model whose outputs to read.
    pattern
        Glob matching the output files.

    Returns
    -------
    list
        Frames concatenated across molecules.
    """
    frames = []
    for file in sorted((CALC_PATH / model_name).glob(pattern)):
        frames.extend(read(file, index=":"))
    return frames


def _relative_energies(frames: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Energies per atom within each molecule, measured from its mean.

    Each method is referenced to its own mean, so an overall offset cancels.
    The mean rather than the lowest structure, because the lowest is not a
    neutral choice: when a model badly over-stabilises one structure, that
    structure becomes its zero and every other energy is then measured from
    the model's worst error. A mean over tens of structures barely moves, so
    the mistake stays where it happened. It is also the constant shift that
    minimises the error, so no model is penalised for an arbitrary choice.

    Both means are taken over the structures where *both* methods have an
    energy. Otherwise a model that fails on a few structures has its baseline
    computed over a different set from the reference, which shifts every one
    of its energies.

    Energies are divided by atom count so that the six molecules, 9 to 21
    atoms, contribute on the same scale rather than the largest dominating a
    pooled error.

    Parameters
    ----------
    frames
        Frames carrying ``molecule``, ``ref_energy_ev`` and ``mlip_energy_ev``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Reference and predicted relative energies in meV/atom, aligned with
        ``frames``. Entries are NaN where the model produced no energy.
    """
    reference = np.array([a.info["ref_energy_ev"] for a in frames], dtype=float)
    predicted = np.array([a.info["mlip_energy_ev"] for a in frames], dtype=float)
    molecules = np.array([a.info["molecule"] for a in frames])
    natoms = np.array([len(a) for a in frames], dtype=float)

    ref_rel = np.full(len(frames), np.nan)
    pred_rel = np.full(len(frames), np.nan)
    for molecule in np.unique(molecules):
        mask = molecules == molecule
        shared = mask & np.isfinite(reference) & np.isfinite(predicted)
        if not shared.any():
            continue
        ref_rel[mask] = reference[mask] - reference[shared].mean()
        pred_rel[mask] = predicted[mask] - predicted[shared].mean()
    return ref_rel / natoms * EV_TO_MEV, pred_rel / natoms * EV_TO_MEV


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_stationary_energies.json",
    title="Stationary point energies",
    x_label="Predicted energy / meV/atom",
    y_label="Reference energy / meV/atom",
    hoverdata={
        "Molecule": INFO["molecule"],
        "Kind": INFO["kind"],
        "Label": INFO["label"],
    },
)
def stationary_energies() -> dict[str, list]:
    """
    Get relative energies at the DFT minima and transition states.

    Returns
    -------
    dict[str, list]
        Reference and predicted relative energies in meV/atom.
    """
    results: dict[str, list] = {"ref": []}
    for model_name in MODELS:
        frames = _read_frames(model_name, POINTS_GLOB)
        ref_rel, pred_rel = _relative_energies(frames)
        results[model_name] = list(pred_rel)
        if not results["ref"]:
            results["ref"] = list(ref_rel)
    return results


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_residual_forces.json",
    title="Residual force at the DFT stationary points",
    y_label="Maximum force component / meV/Å",
    hoverdata={
        "Molecule": INFO["molecule"],
        "Kind": INFO["kind"],
        "Label": INFO["label"],
    },
)
def residual_forces() -> dict[str, list]:
    """
    Get the model's own force at each reference stationary point.

    DFT forces are zero at these geometries by construction, so there is no
    reference to compare against: what the model reports is a residual, saying
    how far the DFT structure is from being stationary under the model. The
    maximum component is used rather than the RMS because it is the quantity a
    geometry optimiser converges on, and this number is a prediction of whether
    the structure will survive reconvergence.

    Returns
    -------
    dict[str, list]
        Per-structure maximum force components in meV/Å.
    """
    return {
        model_name: [
            float(np.max(np.abs(a.arrays["mlip_forces"])) * EV_TO_MEV)
            for a in _read_frames(model_name, POINTS_GLOB)
        ]
        for model_name in MODELS
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_path_forces.json",
    title="Force error along the steepest-descent paths",
    y_label="Per-frame force RMSE / meV/Å",
)
def path_force_errors() -> dict[str, list]:
    """
    Get the force error on each path frame, where DFT forces are non-zero.

    Returns
    -------
    dict[str, list]
        Per-frame force RMSE in meV/Å.
    """
    return {
        model_name: [
            rmse(a.arrays["ref_forces"].ravel(), a.arrays["mlip_forces"].ravel())
            * EV_TO_MEV
            for a in _read_frames(model_name, PATHS_GLOB)
        ]
        for model_name in MODELS
    }


@pytest.fixture
def get_energy_rmse(stationary_energies: dict[str, list]) -> dict[str, float]:
    """
    Get the energy RMSE at the stationary points.

    Parameters
    ----------
    stationary_energies
        Reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        RMSE in meV/atom for each model.
    """
    reference = np.array(stationary_energies["ref"], dtype=float)
    return {
        model_name: rmse(reference, np.array(stationary_energies[model_name]))
        for model_name in MODELS
    }


def _percentile(values: dict[str, list], percentile: float) -> dict[str, float]:
    """
    Get one percentile of each model's per-structure values.

    Parameters
    ----------
    values
        Per-structure values for each model.
    percentile
        Percentile to take, 0 to 100.

    Returns
    -------
    dict[str, float]
        The percentile for each model, NaN where nothing was evaluated.
    """
    results = {}
    for model_name in MODELS:
        entries = np.asarray(values[model_name], dtype=float)
        results[model_name] = (
            float(np.percentile(entries, percentile))
            if entries.size and np.isfinite(entries).all()
            else np.nan
        )
    return results


@pytest.fixture
def get_residual_force(residual_forces: dict[str, list]) -> dict[str, float]:
    """
    Get the median residual force at the stationary points.

    The median rather than the mean because the distribution is expected to be
    skewed: a model can hold most of the landscape and still disagree sharply
    about a few structures.

    Parameters
    ----------
    residual_forces
        Per-structure maximum force components.

    Returns
    -------
    dict[str, float]
        Median in meV/Å for each model.
    """
    return _percentile(residual_forces, 50)


@pytest.fixture
def get_residual_force_p90(residual_forces: dict[str, list]) -> dict[str, float]:
    """
    Get the 90th percentile of the residual force at the stationary points.

    Reported beside the median because the median alone cannot separate a model
    that is uniformly good from one that holds most of the landscape and is
    badly wrong about a tenth of it. Those two behave quite differently under
    reconvergence, and only the second loses structures.

    Parameters
    ----------
    residual_forces
        Per-structure maximum force components.

    Returns
    -------
    dict[str, float]
        90th percentile in meV/Å for each model.
    """
    return _percentile(residual_forces, 90)


@pytest.fixture
def get_path_energy_rmse() -> dict[str, float]:
    """
    Get the energy RMSE along the steepest-descent paths.

    Reported separately from the stationary point energies because it asks a
    different question: the stationary points test the depths of the wells and
    the heights of the barriers, while the paths test the shape of the surface
    in between.

    Returns
    -------
    dict[str, float]
        RMSE in meV/atom for each model.
    """
    results = {}
    for model_name in MODELS:
        ref_rel, pred_rel = _relative_energies(_read_frames(model_name, PATHS_GLOB))
        results[model_name] = rmse(ref_rel, pred_rel)
    return results


@pytest.fixture
def get_path_force_rmse() -> dict[str, float]:
    """
    Get the force RMSE over every component of every path frame.

    Computed from the components directly rather than by averaging the
    per-frame values plotted above. Those are not interchangeable: a frame of
    ethanol contributes 27 components and one of azobenzene 72, so averaging
    per-frame numbers weights the small molecules more heavily than their share
    of the data. On the reference set the two differ by about 2%, which is
    small but is the kind of silent deviation that makes a number
    non-comparable with force RMSEs quoted elsewhere.

    Returns
    -------
    dict[str, float]
        RMSE in meV/Å for each model.
    """
    results = {}
    for model_name in MODELS:
        frames = _read_frames(model_name, PATHS_GLOB)
        reference = np.concatenate([a.arrays["ref_forces"].ravel() for a in frames])
        predicted = np.concatenate([a.arrays["mlip_forces"].ravel() for a in frames])
        results[model_name] = rmse(reference, predicted) * EV_TO_MEV
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "stationary_points_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    get_energy_rmse: dict[str, float],
    get_residual_force: dict[str, float],
    get_residual_force_p90: dict[str, float],
    get_path_energy_rmse: dict[str, float],
    get_path_force_rmse: dict[str, float],
) -> dict[str, dict]:
    """
    Collect all metrics.

    The two measured at the stationary points come first, then the two measured
    along the paths joining them.

    Parameters
    ----------
    get_energy_rmse
        Energy RMSE at the stationary points.
    get_residual_force
        Median residual force at the stationary points.
    get_residual_force_p90
        90th percentile of the residual force at the stationary points.
    get_path_energy_rmse
        Energy RMSE along the paths.
    get_path_force_rmse
        Force RMSE along the paths.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Energy RMSE": get_energy_rmse,
        "Residual force (median)": get_residual_force,
        "Residual force (90th pct)": get_residual_force_p90,
        "Path energy RMSE": get_path_energy_rmse,
        "Path force RMSE": get_path_force_rmse,
    }


def test_stationary_points(metrics: dict[str, dict]) -> None:
    """
    Run the Landscape17 stationary point analysis.

    Parameters
    ----------
    metrics
        All benchmark metric names and dictionary of values for each model.
    """
    return
