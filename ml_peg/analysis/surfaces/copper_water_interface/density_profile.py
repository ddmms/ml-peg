"""
Water density-profile helpers for the copper_water_interface benchmark.

Builds the distribution of O and H atoms along the surface normal (z), referenced
per frame to the mean z of the top layer of Cu atoms (the highest ``atoms_per_layer``
Cu atoms), which is set to z = 0. The resulting per-element curves have the same
``{key: (x, y)}`` shape as the RDF/VDOS/VACF observables, so they plug directly into
the generic scoring helpers in :mod:`ml_peg.analysis.utils.md_water_analysis`
(``property_scores``, ``mean_score``, ``build_bar_data``) and the error primitives in
:mod:`ml_peg.analysis.utils.aml_md_analysis` (``compute_all_errors``).

This module is intentionally local to the benchmark: the shared analysis utilities are
reused by import only, never modified.
"""

from __future__ import annotations

from pathlib import Path
import pickle

from ase import Atoms
from ase.io import read
import numpy as np

# Default z-binning relative to the surface, as a single (start, stop, step) tuple in
# Å. Passed straight to ``np.arange`` to build the histogram edges. The water band sits
# a couple of Å above z = 0 and decays by ~30 Å, so this spans it with a small margin.
DEFAULT_Z_BINS = (-2.0, 35.0, 0.1)


def get_surface_z(
    atoms: Atoms, surface_idx_1: int = 90, surface_idx_2: int = 120
) -> float:
    """
    Get the mean z position of the top slab layer from a fixed atom-index slice.

    The surface is defined as the mean z of atoms ``[surface_idx_1:surface_idx_2]``
    (E.g. for a 4-layer, 5x6 Cu (111) surface: 30 atoms per layer) --> atoms 90-119.
    This defines the z = 0 origin of the density profile for a single frame.

    Parameters
    ----------
    atoms
        Single structure frame.
    surface_idx_1
        Start index of the surface-defining atom slice. Default is 90.
    surface_idx_2
        Stop index (exclusive) of the surface-defining atom slice. Default is 120.

    Returns
    -------
    float
        Mean z position of the top-layer atoms.
    """
    return float(np.mean(atoms.get_positions()[surface_idx_1:surface_idx_2, 2]))


def compute_density_profiles(
    frames: list[Atoms],
    elements: tuple[str, ...] = ("O", "H"),
    surface_idx_1: int = 90,
    surface_idx_2: int = 120,
    z_bins: tuple[float, float, float] = DEFAULT_Z_BINS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute the z density profile of each element relative to the slab surface.

    Parameters
    ----------
    frames
        List of trajectory frames.
    elements
        Chemical symbols to build a profile for. Default is ("O", "H").
    surface_idx_1
        Start index of the surface-defining atom slice (top Cu layer). Default is 90.
    surface_idx_2
        Stop index (exclusive) of the surface-defining atom slice. Default is 120.
    z_bins
        Histogram binning as a (start, stop, step) tuple in Å, passed to
        ``np.arange``. Default is :data:`DEFAULT_Z_BINS`.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Mapping of element -> (z bin centres, normalised density), each normalised to
        sum to 1 so the profile is independent of frame count and atom count.
    """
    edges = np.arange(*z_bins)
    centers = 0.5 * (edges[1:] + edges[:-1])

    zdists = {element: [] for element in elements}
    for frame in frames:
        top_z = get_surface_z(frame, surface_idx_1, surface_idx_2)
        symbols = np.array(frame.get_chemical_symbols())
        z = frame.get_positions()[:, 2] - top_z
        for element in elements:
            zdists[element].append(z[symbols == element])

    profiles = {}
    for element in elements:
        hist, _ = np.histogram(np.concatenate(zdists[element]), bins=edges)
        total = hist.sum()
        density = hist / total if total > 0 else hist.astype(float)
        profiles[element] = (centers, density)
    return profiles


def create_density_profiles(
    models: list,
    data_path: Path,
    calc_path: Path,
    curve_path: Path,
    elements: tuple[str, ...] = ("O", "H"),
) -> dict[str, dict]:
    """
    Create density profiles for the reference and all models.

    Mirrors :func:`ml_peg.analysis.utils.md_water_analysis.create_rdfs` but lives with
    the benchmark. The reference is loaded from the precomputed
    ``density_profile_reference.pkl`` shipped in the benchmark data, and each model
    profile is computed from its ``md-traj.extxyz`` trajectory.

    Parameters
    ----------
    models
        Names of MLIP models to analyse.
    data_path
        Path to the benchmark input data (containing the reference pickle).
    calc_path
        Path to the calculation outputs (one subdir per model).
    curve_path
        Path to write per-model density curve pickles for app use.
    elements
        Chemical symbols to build a profile for. Default is ("O", "H").

    Returns
    -------
    dict[str, dict]
        Dictionary of density profiles for the reference ("ref") and all models.
    """
    profiles = {"ref": None} | dict.fromkeys(models)

    # Load reference density profile
    with open(f"{data_path}/density_profile_reference.pkl", "rb") as f_in:
        profiles["ref"] = pickle.load(f_in)

    if not curve_path.exists():
        curve_path.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        model_dir = calc_path / model_name
        if not model_dir.exists():
            continue

        traj_file = model_dir / "md-traj.extxyz"
        if not traj_file.exists():
            continue

        if not (curve_path / model_name).exists():
            (curve_path / model_name).mkdir(parents=True, exist_ok=True)

        frames = read(traj_file, ":")
        profiles[model_name] = compute_density_profiles(frames, elements=elements)

        # Write density curves to file for app use
        with open(curve_path / model_name / "density_curves.pkl", "wb") as f_out:
            pickle.dump(profiles[model_name], f_out)

    return profiles
