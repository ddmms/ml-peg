"""Analyse SSE-MD benchmark."""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
import pickle

from ase import Atoms, io
from MDAnalysis import Universe
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.electrolytes.SSEMD.calc_SSEMD import (
    DELTA_T_FS,
    FRAME_FREQUENCY,
    N_EQUI_FRAMES,
)
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "electrolytes" / "SSEMD" / "outputs"
OUT_PATH = APP_ROOT / "electrolytes" / "SSEMD"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

BIN_SIZE: float = 0.05  # Angstrom


def get_system_names() -> list[str]:
    """
    Get list of SSE-MD system names from trajectory outputs.

    Returns
    -------
    list[str]
        List of system names derived from trajectory file names.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            traj_files = sorted(model_dir.glob("*.traj"))
            if traj_files:
                for traj_file in traj_files:
                    # Strip model name suffix to recover system name
                    system_name = traj_file.stem.removesuffix(f"_{model_name}")
                    system_names.append(system_name)
                break
    return system_names


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
        R = np.array(n) @ lattice_vectors
        dist = np.linalg.norm(R)
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
        ``(bin_centres, rdf_values)``
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


def metric_pnas(
    rdf_ref: dict, model_rdf: dict
) -> dict[str, float]:
    """Compute normalised MAEs relative to reference RDF data.

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
        mae_val = np.sum(np.absolute(diff)) / (
            np.sum(ref_vals) + np.sum(mod_vals)
        )
        error[name] = float(mae_val)
    return error


def compute_rdf_score(g_aimd: dict, g_model: dict) -> float:
    """Compute RDF similarity score using PNAS metric.

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
    """Load AIMD reference RDFs for all systems from ``rdf_aimd.pkl`` files.

    Extracts the SSEs_data zip (same approach as ``calc_SSEMD.py``) and
    walks the directory tree to find ``rdf_aimd.pkl`` files alongside each
    POSCAR.  Each pickle contains a dict mapping element pair labels to
    ``(bins, rdf_values)`` tuples.

    Returns
    -------
    dict[str, dict]
        Mapping of ``system_name -> {pair_label: (bins, rdf_values)}``.
    """
    data_dir = (
        download_s3_data(
            key="inputs/electrolytes/SSE/SSE.zip",
            filename="SSE.zip",
        )
        / "SSE"
    )

    ref_rdfs: dict[str, dict] = {}
    for pkl_file in sorted(data_dir.rglob("rdf_aimd.pkl")):
        temp_dir = pkl_file.parent
        compound_dir = temp_dir.parent.parent
        system_name = (
            f"{compound_dir.name}_{temp_dir.parent.name}_{temp_dir.name}"
        )

        with open(pkl_file, "rb") as f:
            rdf_data = pickle.load(f)  # noqa: S301

        ref_rdfs[system_name] = rdf_data

    return ref_rdfs


def compute_model_rdfs(model_name: str) -> dict[str, dict]:
    """Compute RDFs from a model's MD trajectory outputs.

    Reads the saved ``.traj`` files produced by ``calc_SSEMD.py``, skips
    equilibration frames, subsamples, and computes RDFs for every element
    pair in each system.

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

    for traj_file in traj_files:
        system_name = traj_file.stem.removesuffix(f"_{model_name}")

        # Read trajectory, skip equilibration and subsample
        ase_traj = io.read(str(traj_file), index=f"{N_EQUI_FRAMES}:")

        # if not ase_traj:
        #     continue

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
        "System": get_system_names(),
    },
)
def rdf_scores() -> dict[str, list]:
    """
    Get per-system RDF similarity scores for all models.

    Computes RDFs from model trajectories and compares them against
    AIMD reference data (or a pseudo-reference from the first available
    model while the reference loader is a placeholder).

    Returns
    -------
    dict[str, list]
        Dictionary with ``"ref"`` key (ideal scores of 1.0) and one key
        per model containing per-system RDF scores.
    """
    system_names = get_system_names()
    results: dict[str, list] = {"ref": [1.0] * len(system_names)} | {
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

        for system_name in system_names:
            if system_name in model_rdfs and system_name in ref_rdfs:
                score = compute_rdf_score(
                    ref_rdfs[system_name], model_rdfs[system_name]
                )
                results[model_name].append(score)
            else:
                results[model_name].append(None)

    return results


@pytest.fixture
def ssemd_errors(rdf_scores: dict[str, list]) -> dict[str, float | None]:
    """
    Compute mean RDF score for each model across all systems.

    Parameters
    ----------
    rdf_scores
        Per-system RDF scores for every model.

    Returns
    -------
    dict[str, float | None]
        Mean RDF score per model, or ``None`` if no data available.
    """
    results: dict[str, float | None] = {}
    for model_name in MODELS:
        scores = rdf_scores.get(model_name, [])
        valid = [s for s in scores if s is not None]
        if valid:
            results[model_name] = float(np.mean(valid))
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ssemd_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(ssemd_errors: dict[str, float | None]) -> dict[str, dict]:
    """
    Get all SSE-MD metrics.

    Parameters
    ----------
    ssemd_errors
        Mean RDF scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "RDF Score": ssemd_errors,
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
