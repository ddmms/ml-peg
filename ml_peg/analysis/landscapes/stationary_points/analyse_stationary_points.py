"""
Analyse the Landscape17 stationary point results and emit metrics.

Reads the raw per-model outputs written by ``calc_stationary_points.py`` and
derives the quantities declared in ``metrics.yml``. Derived properties are
computed here rather than in the calculation, so the raw outputs stay reusable
if a metric is later redefined.

Distributions are summarised by percentile rather than by the fraction crossing
a threshold; see the calculation module's docstring for why.

Notes
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

from ml_peg.analysis.landscapes.stationary_points.hessian_metrics import (
    hessian_failed,
    ratio_agreement,
    reaction_curvature,
    reaction_overlap,
    soft_mode_error,
    stiffness_vs_dft,
)
from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_violin
from ml_peg.analysis.utils.utils import (
    get_struct_info,
    load_metrics_config,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

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


def _structure_ids(info: dict[str, list]) -> list[str]:
    """
    Get each structure's id: its molecule and label, joined.

    The label alone is not unique, since ``min00`` exists once per molecule.
    The id is the last hover entry of every figure whose points are structures,
    and the app opens the file ``structures`` wrote under it, so a click finds
    its structure however the plot filtered or ordered the points.

    Parameters
    ----------
    info
        Struct info carrying ``molecule`` and ``label``.

    Returns
    -------
    list[str]
        One id per structure, in the order of ``info``.
    """
    return [
        f"{molecule}_{label}"
        for molecule, label in zip(info["molecule"], info["label"], strict=True)
    ]


@pytest.fixture
def structures() -> None:
    """
    Write each DFT stationary point to its own file for the app, named by its id.

    One copy serves every model and every figure whose points are structures:
    the energies are evaluated at these DFT geometries, and the minima and
    saddles in the drift plots are these structures before the model moved
    them. Read from the mock outputs, as for any structure that is the same for
    all models. ``get_struct_info`` can write one file per structure, but only
    named by an info key, and no key here is unique.
    """
    out_dir = OUT_PATH / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _read_frames("mock", POINTS_GLOB)
    for structure_id, atoms in zip(_structure_ids(INFO), frames, strict=True):
        write(out_dir / f"{structure_id}.xyz", atoms)


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
    Get the energies per atom within each molecule, measured from its mean.

    Each method is referenced to its own mean, so an overall offset cancels.
    The mean rather than the lowest structure, because the lowest is not a
    neutral choice: when a model badly over-stabilises one structure, that
    structure becomes its zero and every other energy is then measured from
    the model's worst error. A mean over tens of structures barely moves, so
    the mistake stays where it happened. It is also the constant shift that
    minimises the error, so no model is penalised for an arbitrary choice.

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
        ``frames``.
    """
    reference = np.array([a.info["ref_energy_ev"] for a in frames], dtype=float)
    predicted = np.array([a.info["mlip_energy_ev"] for a in frames], dtype=float)
    molecules = np.array([a.info["molecule"] for a in frames])
    natoms = np.array([len(a) for a in frames], dtype=float)

    ref_rel = np.empty(len(frames))
    pred_rel = np.empty(len(frames))
    for molecule in np.unique(molecules):
        mask = molecules == molecule
        ref_rel[mask] = reference[mask] - reference[mask].mean()
        pred_rel[mask] = predicted[mask] - predicted[mask].mean()
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
        "Structure": _structure_ids(INFO),
    },
)
def stationary_energies() -> dict[str, list]:
    """
    Get relative energies at the DFT minima and transition states.

    Every model's output carries the same DFT energies for the same structures,
    so the reference is read from the first.

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

    One value per structure, the RMS over its force components, which against a
    reference of zeros is that structure's RMSE. ``Force RMSE`` pools every
    component instead, as ``Path force RMSE`` does along the paths, so the two
    compare directly: on mace-omol they come out at 129 and 126 meV/Å, which
    says the model's force error at the stationary points is the same size as
    along the paths between them -- one systematic offset across the whole
    surface.

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


@pytest.fixture
def get_force_rmse(residual_forces: dict[str, list]) -> dict[str, float]:
    """
    Get the force RMSE over every component at the stationary points.

    DFT's force is zero at each of these geometries, so the error is the
    model's own force. Pooled over every component of the 95 structures, as
    ``get_path_force_rmse`` is over the path frames, so the two compare
    directly.

    Parameters
    ----------
    residual_forces
        Per-structure RMS forces. Requested so that their violin is written;
        the RMSE itself is pooled from the components.

    Returns
    -------
    dict[str, float]
        RMSE in meV/Å for each model.
    """
    results = {}
    for model_name in MODELS:
        frames = _read_frames(model_name, POINTS_GLOB)
        predicted = np.concatenate([a.arrays["mlip_forces"].ravel() for a in frames])
        results[model_name] = rmse(np.zeros_like(predicted), predicted) * EV_TO_MEV
    return results


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
def get_path_force_rmse(path_force_errors: dict[str, list]) -> dict[str, float]:
    """
    Get the force RMSE over every component of every path frame.

    Computed from the components directly rather than by averaging the
    per-frame values plotted above. Those are not interchangeable: a frame of
    ethanol contributes 27 components and one of azobenzene 72, so averaging
    per-frame numbers weights the small molecules more heavily than their share
    of the data. On the reference set the two differ by about 2%, which is
    small but is the kind of silent deviation that makes a number
    non-comparable with force RMSEs quoted elsewhere.

    Parameters
    ----------
    path_force_errors
        Per-frame force RMSEs. Requested so that their violin is written; the
        RMSE itself is pooled from the components.

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
        "Structure": _structure_ids(MINIMA_INFO),
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
        "Structure": _structure_ids(SADDLE_INFO),
    },
)
def saddle_drift() -> dict[str, list]:
    """
    Get the distance each reference saddle travels under the model.

    Saddles the search did not reach within its step budget are left out of
    this distribution and counted by ``Saddles converged`` instead.

    They are left out rather than carried through as NaN because NaN would make
    the median and 90th percentile NaN for every model: typically half the
    reference saddles are not reached, so the metric would never produce a
    number. That is the opposite of the minima, where a single failure should
    void the result, and the difference is deliberate -- a minimum that will
    not relax is a broken measurement, while a saddle the model has nowhere
    near is a finding about the model.

    Returns
    -------
    dict[str, list]
        Per-structure drift in Å, NaN where the search did not converge.
    """
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
        ``numpy.median`` or ``_p90``.

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
        ``numpy.median`` or ``_p90``.

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


def _p90(values: np.ndarray) -> float:
    """
    Get the 90th percentile, the edge of a distribution's tail.

    Parameters
    ----------
    values
        Per-structure values.

    Returns
    -------
    float
        The value 90% of the structures fall at or below.
    """
    return float(np.percentile(values, 90))


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
def get_minima_drift_p90(minima_drift: dict[str, list]) -> dict[str, float]:
    """
    Get the 90th percentile of the distance a reference minimum travels.

    Read against the median: well above it means a few minima moved a long way.

    Parameters
    ----------
    minima_drift
        Per-structure drift for each model.

    Returns
    -------
    dict[str, float]
        90th percentile in Å for each model.
    """
    return _summarise_over_all(minima_drift, _p90)


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
def get_saddle_drift_p90(saddle_drift: dict[str, list]) -> dict[str, float]:
    """
    Get the 90th percentile distance of a reached saddle from the reference one.

    Read against the median: well above it means a few saddles landed a long
    way off. The step budget truncates the slowest searches, which are
    disproportionately the far-drifting ones, so this understates the tail of
    the distribution that an unlimited search would produce.

    Parameters
    ----------
    saddle_drift
        Per-structure drift for each model.

    Returns
    -------
    dict[str, float]
        90th percentile in Å over the saddles that converged.
    """
    return _summarise_over_reached(saddle_drift, _p90)


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
    title="Agreement with the DFT reaction mode",
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
            reaction_overlap(a) for a in _read_frames(model_name, HESSIANS_GLOB)
        ]
        for model_name in MODELS
    }


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
            soft_mode_error(a) for a in _read_frames(model_name, HESSIANS_GLOB)
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
            reaction_curvature(a) for a in _read_frames(model_name, HESSIANS_GLOB)
        ]
        for model_name in MODELS
    }


@pytest.fixture
def failed_hessians() -> dict[str, bool]:
    """
    Get whether each model's Hessian failed on any structure.

    A failure voids all five Hessian metrics for that model, rather than leaving
    the structure out and scoring the model on the rest.

    Returns
    -------
    dict[str, bool]
        True for a model with a Hessian the calculation recorded as failed.
    """
    return {
        model_name: any(
            hessian_failed(a) for a in _read_frames(model_name, HESSIANS_GLOB)
        )
        for model_name in MODELS
    }


def _void_failed(
    summary: dict[str, float], failed_hessians: dict[str, bool]
) -> dict[str, float]:
    """
    Set a Hessian metric to NaN for every model whose Hessian failed anywhere.

    Parameters
    ----------
    summary
        One number per model.
    failed_hessians
        Whether each model's Hessian failed on any structure.

    Returns
    -------
    dict[str, float]
        ``summary``, with NaN for those models.
    """
    return {
        model_name: np.nan if failed_hessians[model_name] else value
        for model_name, value in summary.items()
    }


@pytest.fixture
def get_soft_mode_error(
    soft_mode_errors: dict[str, list], failed_hessians: dict[str, bool]
) -> dict[str, float]:
    """
    Get the median curvature error on the soft modes.

    Parameters
    ----------
    soft_mode_errors
        Per-structure soft mode error.
    failed_hessians
        Whether each model's Hessian failed on any structure.

    Returns
    -------
    dict[str, float]
        Median over structures, in eV/A^2; NaN for a model whose
        Hessian failed anywhere.
    """
    summary = _summarise_over_reached(soft_mode_errors, np.median)
    return _void_failed(summary, failed_hessians)


@pytest.fixture
def get_reaction_curvature(
    reaction_curvatures: dict[str, list], failed_hessians: dict[str, bool]
) -> dict[str, float]:
    """
    Get the median agreement of the curvature along the reaction coordinate.

    Over the saddles where the model has an imaginary mode. How many it has one
    for is ``Saddles converged``, so the two are read together: this says how
    wrong the barrier shape is where the model sees a barrier at all. Each
    ratio is folded by ``ratio_agreement`` before the median; the violin plot
    keeps the raw ratios, which show whether a model errs sharp or flat.

    Parameters
    ----------
    reaction_curvatures
        Per-structure ratio.
    failed_hessians
        Whether each model's Hessian failed on any structure.

    Returns
    -------
    dict[str, float]
        Median of ``min(r, 1/r)``, 1 meaning agreement; NaN for a model whose
        Hessian failed anywhere.
    """
    agreements = {
        model_name: [ratio_agreement(ratio) for ratio in ratios]
        for model_name, ratios in reaction_curvatures.items()
    }
    return _void_failed(_summarise_over_reached(agreements, np.median), failed_hessians)


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_stiffness.json",
    title="Stiffness relative to DFT",
    y_label="Stiffness vs DFT / % (negative = softer)",
    hoverdata={
        "Molecule": HESSIAN_INFO["molecule"],
        "Kind": HESSIAN_INFO["kind"],
        "Label": HESSIAN_INFO["label"],
    },
)
def stiffness() -> dict[str, list]:
    """
    Get how much stiffer than DFT each structure's curvature is.

    Plotted signed rather than as its size, because the direction matters:
    every model measured so far is soft, not stiff -- MACE-OFF by 6 to 8%,
    GFN2-xTB by 15% -- and a plot of the size alone would hide that they all
    err the same way.

    Returns
    -------
    dict[str, list]
        Per-structure stiffness against DFT, in percent, negative meaning
        softer.
    """
    return {
        model_name: [
            stiffness_vs_dft(a) for a in _read_frames(model_name, HESSIANS_GLOB)
        ]
        for model_name in MODELS
    }


@pytest.fixture
def get_mode_overlap(
    mode_overlaps: dict[str, list], failed_hessians: dict[str, bool]
) -> dict[str, float]:
    """
    Get the median agreement with the DFT normal modes.

    Parameters
    ----------
    mode_overlaps
        Per-structure median mode overlap.
    failed_hessians
        Whether each model's Hessian failed on any structure.

    Returns
    -------
    dict[str, float]
        Median over structures, from 0 to 1; NaN for a model whose
        Hessian failed anywhere.
    """
    return _void_failed(_summarise_over_all(mode_overlaps, np.median), failed_hessians)


@pytest.fixture
def get_reaction_coordinate_overlap(
    reaction_coordinate_overlaps: dict[str, list],
    failed_hessians: dict[str, bool],
) -> dict[str, float]:
    """
    Get the median agreement with the DFT reaction coordinates.

    Over the transition states only, since minima have no imaginary mode.

    Parameters
    ----------
    reaction_coordinate_overlaps
        Per-structure imaginary mode overlap, NaN at minima.
    failed_hessians
        Whether each model's Hessian failed on any structure.

    Returns
    -------
    dict[str, float]
        Median over transition states, from 0 to 1; NaN for a model whose
        Hessian failed anywhere.
    """
    summary = _summarise_over_reached(reaction_coordinate_overlaps, np.median)
    return _void_failed(summary, failed_hessians)


@pytest.fixture
def get_stiffness_error(
    stiffness: dict[str, list], failed_hessians: dict[str, bool]
) -> dict[str, float]:
    """
    Get how far the model's overall stiffness is from DFT's.

    The median of the per-structure stiffness's size, so a model uniformly 6%
    soft scores 6 whichever way it errs; the plot keeps the sign. Reported apart from
    the mode overlaps on purpose: a model can have every mode direction right
    and still be systematically soft, and those are different failures.
    Matching modes by overlap rather than by eigenvalue is what keeps the two
    separable.

    Parameters
    ----------
    stiffness
        Per-structure stiffness against DFT, in percent.
    failed_hessians
        Whether each model's Hessian failed on any structure.

    Returns
    -------
    dict[str, float]
        Median absolute stiffness against DFT, in percent; NaN for a model whose
        Hessian failed anywhere.
    """
    results = {}
    for model_name in MODELS:
        values = np.asarray(stiffness[model_name], dtype=float)
        results[model_name] = (
            float(np.median(np.abs(values))) if values.size else np.nan
        )
    return _void_failed(results, failed_hessians)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "stationary_points_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    get_energy_rmse: dict[str, float],
    get_force_rmse: dict[str, float],
    get_path_energy_rmse: dict[str, float],
    get_path_force_rmse: dict[str, float],
    get_minima_drift_median: dict[str, float],
    get_minima_drift_p90: dict[str, float],
    get_saddles_converged: dict[str, float],
    get_saddle_drift_median: dict[str, float],
    get_saddle_drift_p90: dict[str, float],
    get_connectivity: dict[str, float],
    get_mode_overlap: dict[str, float],
    get_reaction_coordinate_overlap: dict[str, float],
    get_stiffness_error: dict[str, float],
    get_soft_mode_error: dict[str, float],
    get_reaction_curvature: dict[str, float],
    structures: None,
) -> dict[str, dict]:
    """
    Collect all metrics.

    The two measured at the stationary points come first, then the two measured
    along the paths joining them.

    Parameters
    ----------
    get_energy_rmse
        Energy RMSE at the stationary points.
    get_force_rmse
        Force RMSE at the stationary points.
    get_path_energy_rmse
        Energy RMSE along the paths.
    get_path_force_rmse
        Force RMSE along the paths.
    get_minima_drift_median
        Median distance a reference minimum travels.
    get_minima_drift_p90
        90th percentile of the distance a reference minimum travels.
    get_saddles_converged
        Fraction of reference saddles the search reaches.
    get_saddle_drift_median
        Median distance a reached saddle sits from the reference one.
    get_saddle_drift_p90
        90th percentile distance of a reached saddle from the reference one.
    get_connectivity
        Fraction of reference saddles joining the same two minima.
    get_mode_overlap
        Median agreement with the DFT normal modes.
    get_reaction_coordinate_overlap
        Median agreement with the DFT reaction coordinates.
    get_stiffness_error
        Median distance of the overall stiffness from DFT's, in percent.
    get_soft_mode_error
        Median curvature error on the soft modes.
    get_reaction_curvature
        Median curvature agreement along the reaction coordinate.
    structures
        Writes the structures the app opens when a point is clicked.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Energy RMSE": get_energy_rmse,
        "Force RMSE": get_force_rmse,
        "Path energy RMSE": get_path_energy_rmse,
        "Path force RMSE": get_path_force_rmse,
        "Reconverged minima RMSD (median)": get_minima_drift_median,
        "Reconverged minima RMSD (90th pct)": get_minima_drift_p90,
        "Saddles converged": get_saddles_converged,
        "Reconverged saddle RMSD (median)": get_saddle_drift_median,
        "Reconverged saddle RMSD (90th pct)": get_saddle_drift_p90,
        "Connectivity preserved": get_connectivity,
        "Normal mode overlap": get_mode_overlap,
        "Reaction mode overlap": get_reaction_coordinate_overlap,
        "Stiffness error": get_stiffness_error,
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
