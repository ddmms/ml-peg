"""Analyse SSE-MD benchmark."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import pickle
from warnings import warn

from ase import Atoms, io
from MDAnalysis import Universe
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    get_struct_info,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.electrolytes.SSEMD.calc_SSEMD import (
    DELTA_T_FS,
    FRAME_FREQUENCY,
    N_EQUI_FRAMES,
    N_SYSTEMS,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "electrolytes" / "SSEMD" / "outputs"
OUT_PATH = APP_ROOT / "data" / "electrolytes" / "SSEMD"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

BIN_SIZE: float = 0.05  # Angstrom

# Elemental info for filtering, and system labels for hoverdata
INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.traj",
    index=0,
    out_path=OUT_PATH,
    info_keys=["system"],
)
SYSTEM_NAMES: list[str] = INFO["system"]


def ase2mda(atoms: list[Atoms], time_between_frames: float) -> Universe:
    """
    Convert an ASE trajectory to an MDAnalysis Universe.

    Parameters
    ----------
    atoms
        List of ASE Atoms frames.
    time_between_frames
        Time between consecutive frames in fs.

    Returns
    -------
    Universe
        MDAnalysis Universe with loaded coordinates.
    """
    universe = Universe.empty(n_atoms=len(atoms[0]), trajectory=True)
    universe.add_TopologyAttr("name", atoms[0].get_chemical_symbols())
    universe.add_TopologyAttr("type", atoms[0].get_chemical_symbols())
    universe.add_TopologyAttr("masses", atoms[0].get_masses())
    coordinates = np.asarray([np.asarray(frame.positions) for frame in atoms])
    universe.load_new(
        coordinates,
        dimensions=np.asarray(atoms[0].cell.cellpar()),
        dt=time_between_frames * 0.001,
    )
    return universe


def get_rmax_from_cell(lattice_vectors: np.ndarray) -> float:
    """
    Compute the maximum RDF cutoff distance from lattice vectors.

    Parameters
    ----------
    lattice_vectors
        3x3 array of cell lattice vectors.

    Returns
    -------
    float
        Half the minimum lattice image distance.
    """
    min_dist = np.inf
    for n in itertools.product(range(-2, 2 + 1), repeat=3):
        if n == (0, 0, 0):
            continue
        lattice_image = np.array(n) @ lattice_vectors
        dist = np.linalg.norm(lattice_image)
        if dist < min_dist:
            min_dist = dist
    return min_dist * 0.5


def get_element_pairs(species_in_system: list) -> list:
    """
    Return unique sorted element pair combinations.

    Parameters
    ----------
    species_in_system
        Sorted list of unique element symbols.

    Returns
    -------
    list
        List of ``[element_a, element_b]`` pairs with ``a <= b``.
    """
    element_combs = list(itertools.product(species_in_system, repeat=2))
    return [list(comb) for comb in element_combs if comb[0] <= comb[1]]


def compute_rdf(
    traj: Universe,
    rmax: float,
    elements: list,
    bin_size: float,
    cell: np.ndarray,
) -> tuple[list, list]:
    """
    Compute the radial distribution function for a given element pair.

    Parameters
    ----------
    traj
        MDAnalysis Universe trajectory.
    rmax
        Maximum distance cutoff.
    elements
        Two-element list ``[element_a, element_b]``.
    bin_size
        Histogram bin width in Angstrom.
    cell
        3x3 cell matrix.

    Returns
    -------
    tuple[list, list]
        Bin centres and RDF values, as ``(bin_centres, rdf_values)``.
    """
    nbins = int(np.ceil(rmax / bin_size))
    edges = np.arange(0.0, nbins + 1) * bin_size
    edges[-1] = rmax

    rdf = np.zeros(nbins, dtype=float)

    ag1 = traj.select_atoms(f"name {elements[0]}")
    ag2 = traj.select_atoms(f"name {elements[1]}")
    vol_cum = 0
    n_frames = 0
    for frame in traj.trajectory:
        n_frames += 1
        r_ij = np.asarray(ag2.positions[None, :, :]) - np.asarray(
            ag1.positions[:, None, :],
        )
        s_ij = r_ij @ np.linalg.inv(cell)
        s_ij_mic = (s_ij + 0.5) % 1.0 - 0.5
        r_ij_mic = s_ij_mic @ cell

        dist = np.linalg.norm(r_ij_mic, axis=2).flatten()
        mask = (dist > 0.0) & (dist < rmax)

        counts, _ = np.histogram(dist[mask], bins=edges)
        rdf += counts
        vol_cum += frame.volume

    bins = 0.5 * (edges[1:] + edges[:-1])
    shell_volumes = 4 / 3 * math.pi * np.diff(np.power(edges, 3))
    density = ag2.n_atoms / (vol_cum / n_frames)
    n_id_gas = density * shell_volumes
    norm = n_id_gas * ag1.n_atoms * n_frames
    rdf /= norm

    return list(bins), list(rdf)


def compute_rdfs_all(
    traj: Universe,
    rmax: float,
    element_pairs: list,
    bin_size: float,
    cell: np.ndarray,
) -> dict:
    """
    Compute RDFs for all element pairs in a trajectory.

    Parameters
    ----------
    traj
        MDAnalysis Universe trajectory.
    rmax
        Maximum distance cutoff.
    element_pairs
        List of ``[element_a, element_b]`` pairs.
    bin_size
        Histogram bin width in Angstrom.
    cell
        3x3 cell matrix.

    Returns
    -------
    dict
        Mapping ``"A-B" -> (bin_centres, rdf_values)``.
    """
    rdfs = {}
    for ele in element_pairs:
        rdfs["-".join(ele)] = compute_rdf(traj, rmax, ele, bin_size, cell)
    return rdfs


def metric_pnas(rdf_ref: dict, model_rdf: dict) -> dict[str, float]:
    """
    Compute normalised MAEs relative to reference RDF data.

    Given two sets of RDFs returns the mean absolute error
    per element pair.

    Inspired by and partially taken from:
    https://github.com/MarsalekGroup/aml/blob/main/aml/score/util.py

    C. Schran, F. L. Thiemann, P. Rowe, E. A. Müller, O. Marsalek,
    A. Michaelides, "Machine learning potentials for complex aqueous
    systems made simple", PNAS 118, e2110077118 (2021),
    10.1073/pnas.2110077118

    Parameters
    ----------
    rdf_ref
        Reference RDFs ``{pair: (bins, values)}``.
    model_rdf
        Model RDFs ``{pair: (bins, values)}``.

    Returns
    -------
    dict[str, float]
        Normalised MAE per element pair.
    """
    error: dict[str, float] = {}
    for name, data in rdf_ref.items():
        ref_vals = np.asarray(data[1][:-1])
        mod_vals = np.asarray(model_rdf[name][1][:-1])
        diff = ref_vals - mod_vals
        mae_val = np.sum(np.absolute(diff)) / (np.sum(ref_vals) + np.sum(mod_vals))
        error[name] = float(mae_val)
    return error


def compute_rdf_score(g_aimd: dict, g_model: dict) -> float:
    """
    Compute RDF similarity score using PNAS metric.

    Returns the minimum ``(1 - error)`` across all element pairs for a
    single system.  A score of 1.0 indicates perfect agreement.

    Parameters
    ----------
    g_aimd
        Reference RDFs for one system.
    g_model
        Model RDFs for one system.

    Returns
    -------
    float
        Minimum RDF similarity score across element pairs.
    """
    errors = metric_pnas(g_aimd, g_model)
    scores = [1.0 - err for err in errors.values()]
    return float(np.min(scores))


def load_reference_rdfs() -> dict[str, dict]:
    """
    Load AIMD reference RDFs from the calculation outputs.

    Each ``rdf_aimd.pkl`` file contains a dict mapping element pair labels to
    ``(bins, rdf_values)`` tuples.

    Returns
    -------
    dict[str, dict]
        Mapping of ``system_name -> {pair_label: (bins, rdf_values)}``.
    """
    ref_rdfs: dict[str, dict] = {}
    for pkl_file in sorted(CALC_PATH.glob("*_rdf_aimd.pkl")):
        system_name = pkl_file.stem.removesuffix("_rdf_aimd")

        with open(pkl_file, "rb") as f:
            ref_rdfs[system_name] = pickle.load(f)

    return ref_rdfs


def load_status(model_name: str) -> dict[str, dict]:
    """
    Load MD status records written by ``calc_SSEMD.py`` for one model.

    One file is written per system, as each system is dispatched as a separate
    job. A system with no status file either never ran, or was interrupted
    before it could write one, and so is treated as an incomplete run.

    Parameters
    ----------
    model_name
        Name of the MLIP model.

    Returns
    -------
    dict[str, dict]
        Mapping of ``system_name -> status record``.
    """
    statuses: dict[str, dict] = {}
    for status_file in sorted((CALC_PATH / model_name).glob("*_status.json")):
        system_name = status_file.stem.removesuffix(f"_{model_name}_status")
        try:
            statuses[system_name] = json.loads(status_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            warn(f"Could not read {status_file}: {exc}", stacklevel=2)
    return statuses


def stable_percentage(statuses: dict[str, dict]) -> float:
    """
    Calculate the percentage of a model's MD runs that completed.

    Parameters
    ----------
    statuses
        MD status records for one model, keyed by system name.

    Returns
    -------
    float
        Percentage stable, or NaN unless every expected system is present, so a
        partially dispatched benchmark is not reported as a stability score.
    """
    if len(SYSTEM_NAMES) != N_SYSTEMS or set(statuses) != set(SYSTEM_NAMES):
        return np.nan
    return (
        100
        * sum(bool(status.get("stable")) for status in statuses.values())
        / len(statuses)
    )


def compute_model_rdfs(model_name: str) -> dict[str, dict]:
    """
    Compute RDFs from a model's MD trajectory outputs.

    Reads the saved ``.traj`` files produced by ``calc_SSEMD.py``, skips
    equilibration frames, subsamples, and computes RDFs for every element
    pair in each system.

    Systems whose MD did not run to completion, and trajectories that cannot be
    read, are omitted so they are scored as NaN rather than from partial data.

    Parameters
    ----------
    model_name
        Name of the MLIP model.

    Returns
    -------
    dict[str, dict]
        Mapping of ``system_name -> {pair_label: (bins, rdf_values)}``.
    """
    model_dir = CALC_PATH / model_name
    if not model_dir.exists():
        return {}

    system_rdfs: dict[str, dict] = {}
    traj_files = sorted(model_dir.glob("*.traj"))
    statuses = load_status(model_name)

    for traj_file in traj_files:
        system_name = traj_file.stem.removesuffix(f"_{model_name}")

        # Only score trajectories that ran to completion without exploding.
        status = statuses.get(system_name)
        if status is None:
            warn(f"Skipping {traj_file.name}: no status file", stacklevel=2)
            continue
        if not status.get("stable"):
            warn(
                f"Skipping {traj_file.name}: MD incomplete "
                f"({status.get('completed_steps')}/{status.get('expected_steps')} "
                f"steps, failure: {status.get('failure')})",
                stacklevel=2,
            )
            continue

        # Read trajectory, skip equilibration and subsample
        try:
            ase_traj = io.read(str(traj_file), index=f"{N_EQUI_FRAMES}:")
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not read {traj_file}: {exc}", stacklevel=2)
            continue
        if not ase_traj:
            warn(f"Skipping {traj_file.name}: no production frames", stacklevel=2)
            continue

        time_between_frames = DELTA_T_FS * FRAME_FREQUENCY
        mda_traj = ase2mda(ase_traj, time_between_frames)
        cell = np.array(ase_traj[0].cell)
        rmax = get_rmax_from_cell(cell)
        element_pairs = get_element_pairs(
            sorted(set(ase_traj[0].get_chemical_symbols()))
        )
        rdfs = compute_rdfs_all(mda_traj, rmax, element_pairs, BIN_SIZE, cell)
        system_rdfs[system_name] = rdfs

    return system_rdfs


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_ssemd_scores.json",
    title="SSE-MD Scores",
    x_label="Predicted RDF score",
    y_label="Reference RDF score (ideal = 1)",
    hoverdata={
        "System": SYSTEM_NAMES,
    },
)
def rdf_scores() -> dict[str, list]:
    """
    Get per-system RDF similarity scores for all models.

    Compares RDFs computed from each model's trajectories against the AIMD
    reference RDFs. Systems without both a model and a reference RDF score
    as NaN, as do systems whose MD was interrupted or exploded.

    Returns
    -------
    dict[str, list]
        Dictionary with ``"ref"`` key (ideal scores of 1.0) and one key
        per model containing per-system RDF scores.
    """
    results: dict[str, list] = {"ref": [1.0] * len(SYSTEM_NAMES)} | {
        mlip: [] for mlip in MODELS
    }

    # Pre-compute all model RDFs so each trajectory is read only once
    all_model_rdfs: dict[str, dict] = {}
    for model_name in MODELS:
        rdfs = compute_model_rdfs(model_name)
        if rdfs:
            all_model_rdfs[model_name] = rdfs

    # Load AIMD reference RDFs
    ref_rdfs = load_reference_rdfs()

    # Score each model against the reference
    for model_name in MODELS:
        model_rdfs = all_model_rdfs.get(model_name, {})

        for system_name in SYSTEM_NAMES:
            if system_name in model_rdfs and system_name in ref_rdfs:
                score = compute_rdf_score(
                    ref_rdfs[system_name], model_rdfs[system_name]
                )
                results[model_name].append(score)
            else:
                results[model_name].append(np.nan)

    return results


@pytest.fixture
def ssemd_errors(rdf_scores: dict[str, list]) -> dict[str, float]:
    """
    Compute mean RDF score for each model across all systems.

    Parameters
    ----------
    rdf_scores
        Per-system RDF scores for every model.

    Returns
    -------
    dict[str, float]
        Mean RDF score per model, ignoring systems scored as NaN, or NaN if
        the model has no scored systems.
    """
    results: dict[str, float] = {}
    for model_name in MODELS:
        scores = rdf_scores.get(model_name, [])
        valid = [s for s in scores if s is not None and not np.isnan(s)]
        results[model_name] = float(np.mean(valid)) if valid else np.nan
    return results


@pytest.fixture
def ssemd_stability() -> dict[str, float | None]:
    """
    Compute the percentage of systems each model completed without exploding.

    Returns
    -------
    dict[str, float | None]
        Percentage of stable systems per model, or None if the model has not
        been run on every system.
    """
    results: dict[str, float | None] = {}
    for model_name in MODELS:
        value = stable_percentage(load_status(model_name))
        results[model_name] = float(value) if np.isfinite(value) else None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ssemd_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    ssemd_errors: dict[str, float], ssemd_stability: dict[str, float | None]
) -> dict[str, dict]:
    """
    Get all SSE-MD metrics.

    Parameters
    ----------
    ssemd_errors
        Mean RDF scores for all models.
    ssemd_stability
        Percentage of systems completed without exploding, for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "RDF Score": ssemd_errors,
        "Stable trajectories": ssemd_stability,
    }


def test_ssemd(metrics: dict[str, dict]) -> None:
    """
    Run SSE-MD test.

    Parameters
    ----------
    metrics
        All SSE-MD metrics.
    """
    return
