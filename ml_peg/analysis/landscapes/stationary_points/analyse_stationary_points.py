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

from ase.io import read, write
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
MINIMA_GLOB = "*_reconverged_minima.extxyz"
SADDLES_GLOB = "*_reconverged_saddles.extxyz"
HESSIANS_GLOB = "*_hessians.extxyz"

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
# The reconverged structures are separate sets of 28 and 67, so they need their
# own labels rather than reusing the 95 above.
MINIMA_INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern=MINIMA_GLOB,
    info_keys=["molecule", "label"],
    write_info=False,
    write_structs=False,
    out_path=OUT_PATH,
)
HESSIAN_INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern=HESSIANS_GLOB,
    info_keys=["molecule", "kind", "label"],
    write_info=False,
    write_structs=False,
    out_path=OUT_PATH,
)
SADDLE_INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern=SADDLES_GLOB,
    info_keys=["molecule", "label", "connects"],
    write_info=False,
    write_structs=False,
    out_path=OUT_PATH,
)


def _stage_structures(model_name: str, pattern: str, subdirectory: str) -> None:
    """
    Write one file per structure so the app can show the one that was clicked.

    ``get_struct_info`` can stage structures, but it writes one file per input
    file -- six, one per molecule -- while the plots have one point per
    structure. Its ``struct_name_key`` option writes one file per structure
    instead, named by an info key, and no key here is unique: ``min00`` exists
    once per molecule and the files would overwrite each other.

    Naming by position rather than by label avoids inventing a unique key and
    guarantees the ordering the app depends on: it globs this directory and
    sorts, so zero-padded indices put the files in exactly the order the
    points appear in the figure. The structure's identity is on the point's
    hover text.

    Parameters
    ----------
    model_name
        Model whose outputs to stage.
    pattern
        Glob matching the output files, which fixes the order.
    subdirectory
        Directory under the model's app data, one per figure.
    """
    frames = _read_frames(model_name, pattern)
    if not frames:
        return
    out_dir = OUT_PATH / model_name / subdirectory
    out_dir.mkdir(parents=True, exist_ok=True)
    width = len(str(len(frames) - 1))
    for index, atoms in enumerate(frames):
        write(out_dir / f"{index:0{width}d}.xyz", atoms)


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
    for model_name in MODELS:
        _stage_structures(model_name, POINTS_GLOB, "points")

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
    y_label="RMS residual force / meV/Å",
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
    how far the DFT structure is from being stationary under the model.

    Reported as the RMS over the structure's force components, which against a
    reference of zeros is exactly an RMSE, and so is the same statistic as
    ``Path force RMSE``. That comparability is the point: on MACE-OFF the two
    come out at 159 and 153 meV/Å, which says the model's force error at the
    stationary points is the same size as along the paths between them -- one
    systematic offset across the whole surface.

    Returns
    -------
    dict[str, list]
        Per-structure RMS force in meV/Å.
    """
    return {
        model_name: [
            float(np.sqrt(np.mean(a.arrays["mlip_forces"] ** 2)) * EV_TO_MEV)
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
@plot_violin(
    filename=OUT_PATH / "figure_minima_drift.json",
    title="How far a reference minimum moves when relaxed",
    y_label="RMSD from the DFT minimum / Å",
    hoverdata={
        "Molecule": MINIMA_INFO["molecule"],
        "Label": MINIMA_INFO["label"],
    },
)
def minima_drift() -> dict[str, list]:
    """
    Get the distance each reference minimum travels under the model.

    A relaxation that stopped on the iteration limit contributes NaN rather
    than a distance, because how far it got is not a measurement of where the
    model's minimum is.

    Returns
    -------
    dict[str, list]
        Per-structure drift in Å.
    """
    for model_name in MODELS:
        _stage_structures(model_name, MINIMA_GLOB, "minima")

    return {
        model_name: [a.info["drift"] for a in _read_frames(model_name, MINIMA_GLOB)]
        for model_name in MODELS
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_saddle_drift.json",
    title="How far a reference transition state moves when reconverged",
    y_label="RMSD from the DFT saddle / Å",
    hoverdata={
        "Molecule": SADDLE_INFO["molecule"],
        "Label": SADDLE_INFO["label"],
        "Connects": SADDLE_INFO["connects"],
    },
)
def saddle_drift() -> dict[str, list]:
    """
    Get the distance each reference saddle travels under the model.

    Saddles the search did not reach within its step budget are left out of
    this distribution and counted by ``Saddles converged`` instead.

    They are left out rather than carried through as NaN because NaN would
    make the median and mean NaN for every model: typically half the reference
    saddles are not reached, so the metric would never produce a number. That
    is the opposite of the minima, where a single failure should void the
    result, and the difference is deliberate -- a minimum that will not relax
    is a broken measurement, while a saddle the model has nowhere near is a
    finding about the model.

    Returns
    -------
    dict[str, list]
        Per-structure drift in Å, NaN where the search did not converge.
    """
    for model_name in MODELS:
        _stage_structures(model_name, SADDLES_GLOB, "saddles")

    return {
        model_name: [a.info["drift"] for a in _read_frames(model_name, SADDLES_GLOB)]
        for model_name in MODELS
    }


def _summarise_over_all(values: dict[str, list], statistic) -> dict[str, float]:
    """
    Summarise every structure, so a single missing value voids the result.

    The counterpart to ``_summarise_over_reached``. The two differ only in what
    they do with a structure that produced no value, and that difference is the
    reason both exist: here a missing value makes the whole metric NaN.

    Parameters
    ----------
    values
        Per-structure values for each model.
    statistic
        ``numpy.median`` or ``numpy.mean``.

    Returns
    -------
    dict[str, float]
        One number per model, NaN if any value is NaN.
    """
    return {
        model_name: float(statistic(np.asarray(values[model_name], dtype=float)))
        if values[model_name]
        else np.nan
        for model_name in MODELS
    }


def _summarise_over_reached(values: dict[str, list], statistic) -> dict[str, float]:
    """
    Summarise only the structures that produced a value.

    The counterpart to ``_summarise_over_all``: here a structure the search never
    reached is left out rather than voiding the metric, and is counted by a
    separate coverage number instead.

    Parameters
    ----------
    values
        Per-structure values for each model.
    statistic
        ``numpy.median`` or ``numpy.mean``.

    Returns
    -------
    dict[str, float]
        One number per model over the finite entries.
    """
    results = {}
    for model_name in MODELS:
        finite = np.asarray(values[model_name], dtype=float)
        finite = finite[np.isfinite(finite)]
        results[model_name] = float(statistic(finite)) if finite.size else np.nan
    return results


@pytest.fixture
def get_minima_drift_median(minima_drift: dict[str, list]) -> dict[str, float]:
    """
    Get the median distance a reference minimum travels.

    Parameters
    ----------
    minima_drift
        Per-structure drift for each model.

    Returns
    -------
    dict[str, float]
        Median in Å for each model.
    """
    return _summarise_over_all(minima_drift, np.median)


@pytest.fixture
def get_minima_drift_mean(minima_drift: dict[str, list]) -> dict[str, float]:
    """
    Get the mean distance a reference minimum travels.

    Parameters
    ----------
    minima_drift
        Per-structure drift for each model.

    Returns
    -------
    dict[str, float]
        Mean in Å for each model.
    """
    return _summarise_over_all(minima_drift, np.mean)


@pytest.fixture
def get_saddles_converged() -> dict[str, float]:
    """
    Get the fraction of reference saddles the search reaches.

    The denominator is every reference saddle, not the ones that converged, so
    this and the drift metrics cannot be read as flattering each other: a model
    that reaches only a handful cannot hide behind their small drift.

    Returns
    -------
    dict[str, float]
        Fraction for each model.
    """
    results = {}
    for model_name in MODELS:
        frames = _read_frames(model_name, SADDLES_GLOB)
        results[model_name] = (
            float(np.mean([bool(a.info["converged"]) for a in frames]))
            if frames
            else np.nan
        )
    return results


@pytest.fixture
def get_saddle_drift_median(saddle_drift: dict[str, list]) -> dict[str, float]:
    """
    Get the median distance a reached saddle sits from the reference one.

    Parameters
    ----------
    saddle_drift
        Per-structure drift for each model.

    Returns
    -------
    dict[str, float]
        Median in Å over the saddles that converged.
    """
    return _summarise_over_reached(saddle_drift, np.median)


@pytest.fixture
def get_saddle_drift_mean(saddle_drift: dict[str, list]) -> dict[str, float]:
    """
    Get the mean distance a reached saddle sits from the reference one.

    Read against the median: a mean well above it means a few saddles landed a
    long way off. The step budget truncates the slowest searches, which are
    disproportionately the far-drifting ones, so this understates the skew of
    the distribution that an unlimited search would produce.

    Parameters
    ----------
    saddle_drift
        Per-structure drift for each model.

    Returns
    -------
    dict[str, float]
        Mean in Å over the saddles that converged.
    """
    return _summarise_over_reached(saddle_drift, np.mean)


@pytest.fixture
def get_connectivity() -> dict[str, float]:
    """
    Get the fraction of reference saddles that join the same two minima.

    Over every reference saddle rather than only the reconverged ones. A saddle
    can sit in the right place and still rewire the network, which changes the
    kinetics the landscape implies while leaving every structure looking right.

    Returns
    -------
    dict[str, float]
        Fraction for each model.
    """
    results = {}
    for model_name in MODELS:
        frames = _read_frames(model_name, SADDLES_GLOB)
        results[model_name] = (
            float(np.mean([bool(a.info["connectivity_preserved"]) for a in frames]))
            if frames
            else np.nan
        )
    return results


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_mode_overlaps.json",
    title="Agreement with the DFT normal modes",
    y_label="Median mode overlap (cosine)",
    hoverdata={
        "Molecule": HESSIAN_INFO["molecule"],
        "Kind": HESSIAN_INFO["kind"],
        "Label": HESSIAN_INFO["label"],
    },
)
def mode_overlaps() -> dict[str, list]:
    """
    Get how closely the model's normal modes match the DFT ones.

    One number per structure: the median over its 3N-6 modes, each matched to
    a DFT mode one to one. Both Hessians are taken at the same geometry, so
    this is a statement about the curvature the model assigns to the reference
    structure, not about where the model would put that structure.

    Returns
    -------
    dict[str, list]
        Per-structure median mode overlap, from 0 to 1.
    """
    return {
        model_name: [
            a.info["mode_overlap_median"]
            for a in _read_frames(model_name, HESSIANS_GLOB)
        ]
        for model_name in MODELS
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_reaction_coordinates.json",
    title="Agreement with the DFT reaction coordinate",
    y_label="Imaginary mode overlap (cosine)",
    hoverdata={
        "Molecule": HESSIAN_INFO["molecule"],
        "Kind": HESSIAN_INFO["kind"],
        "Label": HESSIAN_INFO["label"],
    },
)
def reaction_coordinate_overlaps() -> dict[str, list]:
    """
    Get whether the model agrees about the direction of each reaction.

    Only transition states have one, so minima contribute NaN and drop out of
    the plot. A sharper question than the average mode: a model can reproduce
    every ordinary vibration and still disagree about which way the reaction
    proceeds, and only this mode decides that.

    Returns
    -------
    dict[str, list]
        Per-structure imaginary mode overlap, NaN at minima.
    """
    return {
        model_name: [
            _reaction_overlap(a) for a in _read_frames(model_name, HESSIANS_GLOB)
        ]
        for model_name in MODELS
    }


def _reaction_overlap(atoms) -> float:
    """
    Get the overlap with the DFT reaction coordinate, where the model has one.

    Counted only when the model's matched mode is also negative. Without that
    condition the overlap is taken against a mode the model considers uphill,
    which credits it for agreeing about a reaction it does not believe has a
    barrier: on MACE-OFF, ethanol ts0_1_0 overlaps 0.98 while the model's
    lowest eigenvalue there is +0.10. Those 32 of 67 structures scored 0.974
    against the 35 real ones at 0.989, so including them moved the metric
    barely at all while halving what it described.

    Parameters
    ----------
    atoms
        A structure carrying the overlap and the paired eigenvalues.

    Returns
    -------
    float
        The overlap, or NaN where the model has no imaginary mode.
    """
    overlap = atoms.info.get("reaction_coordinate_overlap", np.nan)
    if not np.isfinite(overlap):
        return np.nan
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    return overlap if reference[0] < 0 and model[0] < 0 else np.nan


def _scaling(atoms) -> float:
    """
    Fit one factor relating the model's curvature to the reference's.

    The least-squares slope through the origin, which weights each mode by the
    square of its reference eigenvalue and so is a statement about the stiff
    modes: on this reference the eigenvalues span four orders of magnitude, and
    the ten stiffest modes of aspirin carry 76% of the fit.

    Weighting every mode equally instead was tried and is worse. The softest
    modes have reference eigenvalues near zero -- 1% of them below 0.05 eV/A^2
    -- so a small absolute error becomes an enormous ratio, and the resulting
    statistic is governed by where the cutoff is put rather than by the model:
    over the soft half it reads 1.12 with no floor and 1.00 with a floor of
    2 eV/A^2, while the stiff half holds at 0.959 to 0.953 throughout. The
    ratio-weighted version measures the cutoff; this one measures the model.

    Parameters
    ----------
    atoms
        A structure carrying paired reference and model eigenvalues.

    Returns
    -------
    float
        The scaling, 1 meaning agreement.
    """
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    # Real modes only. At a saddle the negative eigenvalue is the reaction
    # coordinate, which has its own metric, and leaving it here would put one
    # mode in two. It makes no numerical difference -- the slope agrees to six
    # decimals either way, since its weight is 3e-5 of the stiffest mode's --
    # so this is about the metrics not overlapping, not about the number.
    real = reference > 0
    reference, model = reference[real], model[real]
    return float(np.dot(reference, model) / np.dot(reference, reference))


def _soft_mode_error(atoms) -> float:
    """
    Get the model's curvature error on the softest modes, in absolute terms.

    Absolute rather than relative, because a ratio is unusable here. The
    softest modes have reference eigenvalues near zero -- paracetamol's
    lowest is 0.0026 eV/A^2 -- so an error of 0.18 eV/A^2, small beside the
    16 eV/A^2 the model is out on its stiffest mode, reads as a factor of 67.
    Averaging such ratios produces a number that tracks wherever the near-zero
    modes are cut off: 1.19 with no floor against 1.01 at 2 eV/A^2. The
    absolute error holds at 0.076, 0.074, 0.075 across the same range.

    The softest third is taken because that is where a landscape lives: the
    torsions that decide barrier heights and whether two minima are distinct,
    rather than the X-H stretches that dominate a least-squares slope.

    Parameters
    ----------
    atoms
        A structure carrying paired reference and model eigenvalues.

    Returns
    -------
    float
        Median absolute eigenvalue error over the softest third, in eV/A^2.
    """
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    positive = reference > 0
    reference, model = reference[positive], model[positive]
    if reference.size < 3:
        return np.nan
    softest = reference < np.percentile(reference, 33)
    return float(np.median(np.abs(model - reference)[softest]))


def _reaction_curvature(atoms) -> float:
    """
    Get how steeply the model falls away from the barrier top, against DFT.

    The imaginary eigenvalue sets the curvature along the reaction coordinate,
    which fixes the tunnelling correction and the transition state theory
    prefactor. A ratio is sound here where it is not for the soft modes: the
    reference imaginary eigenvalue is never near zero, ranging from -0.08 to
    -9.18 eV/A^2 across this reference.

    NaN unless both are negative. Where the model has no imaginary mode at the
    DFT geometry there is no curvature to compare, and that is not a small
    disagreement but a statement that the model does not see a barrier top
    here at all -- counted by ``Saddles converged``, which agrees with this
    condition on 65 of 67 structures.

    Parameters
    ----------
    atoms
        A structure carrying paired reference and model eigenvalues.

    Returns
    -------
    float
        Model imaginary eigenvalue over the reference one, 1 meaning
        agreement.
    """
    if atoms.info["kind"] != "transition_state":
        return np.nan
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    if reference[0] >= 0 or model[0] >= 0:
        return np.nan
    return float(model[0] / reference[0])


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_soft_mode_error.json",
    title="Curvature error on the softest modes",
    y_label="Median absolute eigenvalue error / eV/Å²",
    hoverdata={
        "Molecule": HESSIAN_INFO["molecule"],
        "Kind": HESSIAN_INFO["kind"],
        "Label": HESSIAN_INFO["label"],
    },
)
def soft_mode_errors() -> dict[str, list]:
    """
    Get the curvature error on the soft modes of each structure.

    Returns
    -------
    dict[str, list]
        Per-structure median absolute error over the softest third, eV/A^2.
    """
    return {
        model_name: [
            _soft_mode_error(a) for a in _read_frames(model_name, HESSIANS_GLOB)
        ]
        for model_name in MODELS
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_reaction_curvature.json",
    title="Curvature along the reaction coordinate",
    y_label="Model imaginary eigenvalue / DFT (1 = agreement)",
    hoverdata={
        "Molecule": HESSIAN_INFO["molecule"],
        "Kind": HESSIAN_INFO["kind"],
        "Label": HESSIAN_INFO["label"],
    },
)
def reaction_curvatures() -> dict[str, list]:
    """
    Get how the model's barrier curvature compares with DFT, saddle by saddle.

    Returns
    -------
    dict[str, list]
        Per-structure ratio, NaN where the model has no imaginary mode.
    """
    return {
        model_name: [
            _reaction_curvature(a) for a in _read_frames(model_name, HESSIANS_GLOB)
        ]
        for model_name in MODELS
    }


@pytest.fixture
def get_soft_mode_error(soft_mode_errors: dict[str, list]) -> dict[str, float]:
    """
    Get the median curvature error on the soft modes.

    Parameters
    ----------
    soft_mode_errors
        Per-structure soft mode error.

    Returns
    -------
    dict[str, float]
        Median over structures, in eV/A^2.
    """
    return _summarise_over_reached(soft_mode_errors, np.median)


@pytest.fixture
def get_reaction_curvature(reaction_curvatures: dict[str, list]) -> dict[str, float]:
    """
    Get the median curvature ratio along the reaction coordinate.

    Over the saddles where the model has an imaginary mode. How many it has one
    for is ``Saddles converged``, so the two are read together: this says how
    wrong the barrier shape is where the model sees a barrier at all.

    Parameters
    ----------
    reaction_curvatures
        Per-structure ratio.

    Returns
    -------
    dict[str, float]
        Median ratio, 1 meaning agreement.
    """
    return _summarise_over_reached(reaction_curvatures, np.median)


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_curvature_scaling.json",
    title="Curvature relative to DFT",
    y_label="Eigenvalue scaling (1 = agreement)",
    hoverdata={
        "Molecule": HESSIAN_INFO["molecule"],
        "Kind": HESSIAN_INFO["kind"],
        "Label": HESSIAN_INFO["label"],
    },
)
def curvature_scaling() -> dict[str, list]:
    """
    Get the slope of the model's eigenvalues against the DFT ones.

    Plotted as the raw scaling rather than its deviation from 1, because the
    direction matters: every model measured so far is soft, not stiff --
    MACE-OFF by 6 to 8%, GFN2-xTB by 15% -- and a plot of the absolute
    deviation would hide that they all err the same way.

    Returns
    -------
    dict[str, list]
        Per-structure eigenvalue scaling, 1 meaning agreement.
    """
    return {
        model_name: [_scaling(a) for a in _read_frames(model_name, HESSIANS_GLOB)]
        for model_name in MODELS
    }


@pytest.fixture
def get_mode_overlap(mode_overlaps: dict[str, list]) -> dict[str, float]:
    """
    Get the median agreement with the DFT normal modes.

    Parameters
    ----------
    mode_overlaps
        Per-structure median mode overlap.

    Returns
    -------
    dict[str, float]
        Median over structures, from 0 to 1.
    """
    return _summarise_over_all(mode_overlaps, np.median)


@pytest.fixture
def get_reaction_coordinate_overlap(
    reaction_coordinate_overlaps: dict[str, list],
) -> dict[str, float]:
    """
    Get the median agreement with the DFT reaction coordinates.

    Over the transition states only, since minima have no imaginary mode.

    Parameters
    ----------
    reaction_coordinate_overlaps
        Per-structure imaginary mode overlap, NaN at minima.

    Returns
    -------
    dict[str, float]
        Median over transition states, from 0 to 1.
    """
    return _summarise_over_reached(reaction_coordinate_overlaps, np.median)


@pytest.fixture
def get_curvature_scaling_error(
    curvature_scaling: dict[str, list],
) -> dict[str, float]:
    """
    Get how far the model's curvature is from the DFT curvature.

    The median of the per-structure distance from a scaling of 1, so a model
    uniformly 6% soft scores 0.06 whichever way it errs. Reported apart from
    the mode overlaps on purpose: a model can have every mode direction right
    and still be systematically soft, and those are different failures.
    Matching modes by overlap rather than by eigenvalue is what keeps the two
    separable.

    Parameters
    ----------
    curvature_scaling
        Per-structure eigenvalue scaling.

    Returns
    -------
    dict[str, float]
        Median absolute deviation from 1.
    """
    results = {}
    for model_name in MODELS:
        values = np.asarray(curvature_scaling[model_name], dtype=float)
        results[model_name] = (
            float(np.median(np.abs(values - 1.0))) if values.size else np.nan
        )
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
    get_path_energy_rmse: dict[str, float],
    get_path_force_rmse: dict[str, float],
    get_minima_drift_median: dict[str, float],
    get_minima_drift_mean: dict[str, float],
    get_saddles_converged: dict[str, float],
    get_saddle_drift_median: dict[str, float],
    get_saddle_drift_mean: dict[str, float],
    get_connectivity: dict[str, float],
    get_mode_overlap: dict[str, float],
    get_reaction_coordinate_overlap: dict[str, float],
    get_curvature_scaling_error: dict[str, float],
    get_soft_mode_error: dict[str, float],
    get_reaction_curvature: dict[str, float],
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
    get_path_energy_rmse
        Energy RMSE along the paths.
    get_path_force_rmse
        Force RMSE along the paths.
    get_minima_drift_median
        Median distance a reference minimum travels.
    get_minima_drift_mean
        Mean distance a reference minimum travels.
    get_saddles_converged
        Fraction of reference saddles the search reaches.
    get_saddle_drift_median
        Median distance a reached saddle sits from the reference one.
    get_saddle_drift_mean
        Mean distance a reached saddle sits from the reference one.
    get_connectivity
        Fraction of reference saddles joining the same two minima.
    get_mode_overlap
        Median agreement with the DFT normal modes.
    get_reaction_coordinate_overlap
        Median agreement with the DFT reaction coordinates.
    get_curvature_scaling_error
        Median distance of the curvature from the DFT curvature.
    get_soft_mode_error
        Median curvature error on the soft modes.
    get_reaction_curvature
        Median curvature ratio along the reaction coordinate.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Energy RMSE": get_energy_rmse,
        "Residual force": get_residual_force,
        "Path energy RMSE": get_path_energy_rmse,
        "Path force RMSE": get_path_force_rmse,
        "Minima drift (median)": get_minima_drift_median,
        "Minima drift (mean)": get_minima_drift_mean,
        "Saddles converged": get_saddles_converged,
        "Saddle drift (median)": get_saddle_drift_median,
        "Saddle drift (mean)": get_saddle_drift_mean,
        "Connectivity preserved": get_connectivity,
        "Normal mode overlap": get_mode_overlap,
        "Reaction coordinate overlap": get_reaction_coordinate_overlap,
        "Curvature scaling error": get_curvature_scaling_error,
        "Soft mode curvature error": get_soft_mode_error,
        "Reaction mode curvature": get_reaction_curvature,
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
