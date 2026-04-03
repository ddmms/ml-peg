"""
Compute Li-O radial distribution functions for LiTFSI/H2O electrolyte.

Reads pre-computed NVT trajectories (.extxyz, 100 ps) for each MLIP model
and computes g(r) for Li-O_total, Li-O_water (O bonded to H, d_OH < 1.25 A),
and Li-O_TFSI (O bonded to S, d_OS < 1.75 A), plus coordination numbers by
integration of the first peak.

System: 21 m LiTFSI / H2O, p64_w170 cell (1534 atoms).
Experimental reference for coordination numbers:
Watanabe et al., J. Phys. Chem. B 125, 7477 (2021), DOI: 10.1021/acs.jpcb.1c04693.
Neutron diffraction with 6Li/7Li + H/D isotopic substitution, ~18.5 m LiTFSI/H2O.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ase.geometry import get_distances
from ase.io import iread
import numpy as np
import pytest

# --- Configuration -----------------------------------------------------------

DATA_ROOT = Path(
    os.environ.get(
        "ML_PEG_WISE_XRAY_DATA_ROOT",
        "/lus/work/CT9/cin1387/lbrugnoli/prove/ml_peg_benchmark/data/wise_electrolytes/xray_sf",
    )
)
OUT_PATH = Path(__file__).parent / "outputs"

MODELS = [
    "matpes-r2scan",
    "mace-mpa-0-medium",
    "mace-omat-0-medium",
    "mace-mp-0b3",
    "mace-mh-1-omat",
    "mace-mh-1-omol",
]

# RDF parameters
R_MAX = 6.0  # Å — covers first + second shell
DR = 0.02  # Å — bin width
R_CUT_COORD = 2.83  # Å — first minimum of Li-O g(r) from r2SCAN AIMD reference

# O-type identification cutoffs (applied on first frame only)
D_OH_CUT = 1.25  # Å — O-H bond in water
D_OS_CUT = 1.75  # Å — O-S bond in TFSI


# --- Helpers -----------------------------------------------------------------


def identify_o_types(atoms) -> tuple[np.ndarray, np.ndarray]:
    """
    Return indices of O_water and O_TFSI from a single ASE Atoms frame.

    Parameters
    ----------
    atoms : ase.Atoms
        A single frame from the trajectory.

    Returns
    -------
    o_water : np.ndarray
        Indices of oxygen atoms bonded to hydrogen (water oxygens).
    o_tfsi : np.ndarray
        Indices of oxygen atoms bonded to sulfur (TFSI oxygens).
    """
    syms = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    o_idx = np.where(syms == "O")[0]
    h_idx = np.where(syms == "H")[0]
    s_idx = np.where(syms == "S")[0]

    o_water, o_tfsi = [], []
    for o in o_idx:
        _, d_oh = get_distances(pos[o : o + 1], pos[h_idx], cell=cell, pbc=pbc)
        _, d_os = get_distances(pos[o : o + 1], pos[s_idx], cell=cell, pbc=pbc)
        if d_oh.min() < D_OH_CUT:
            o_water.append(o)
        elif d_os.min() < D_OS_CUT:
            o_tfsi.append(o)

    return np.array(o_water), np.array(o_tfsi)


def compute_rdf(
    traj_path: Path,
    o_water_idx: np.ndarray,
    o_tfsi_idx: np.ndarray,
    r_max: float = R_MAX,
    dr: float = DR,
    skip_frames: int = 0,  # trajectories are pre-equilibrated (50-100 ps window)
) -> dict:
    """
    Compute Li-O RDFs from an extxyz trajectory.

    Parameters
    ----------
    traj_path : Path
        Path to .extxyz trajectory file.
    o_water_idx : np.ndarray
        Atom indices of O_water (from first frame, fixed).
    o_tfsi_idx : np.ndarray
        Atom indices of O_TFSI (from first frame, fixed).
    r_max : float
        Maximum distance for RDF.
    dr : float
        Bin width.
    skip_frames : int
        Number of initial frames to skip (equilibration).

    Returns
    -------
    dict
        Dictionary with keys ``r``, ``gr_LiO_total``, ``gr_LiO_water``,
        ``gr_LiO_TFSI``, ``coord_LiO_total``, ``coord_LiO_water``,
        ``coord_LiO_TFSI``, ``n_li``, ``n_O_water``, ``n_O_TFSI``,
        ``n_frames_used``, and ``r_cut_coord``.
    """
    bins = np.arange(0, r_max + dr, dr)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    n_bins = len(r_centers)

    hist_total = np.zeros(n_bins)
    hist_water = np.zeros(n_bins)
    hist_tfsi = np.zeros(n_bins)

    n_frames = 0
    n_li = None
    volume = None
    o_all_idx = np.concatenate([o_water_idx, o_tfsi_idx])

    for frame_idx, atoms in enumerate(
        iread(str(traj_path), format="extxyz", index=":")
    ):
        if frame_idx < skip_frames:
            continue

        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        li_idx = np.where(syms == "Li")[0]
        if n_li is None:
            n_li = len(li_idx)
        if volume is None:
            volume = atoms.get_volume()

        pos_li = pos[li_idx]

        for o_set, hist in [
            (o_all_idx, hist_total),
            (o_water_idx, hist_water),
            (o_tfsi_idx, hist_tfsi),
        ]:
            pos_o = pos[o_set]
            _, dists = get_distances(pos_li, pos_o, cell=cell, pbc=pbc)
            dists_flat = dists.ravel()
            dists_flat = dists_flat[dists_flat < r_max]
            h, _ = np.histogram(dists_flat, bins=bins)
            hist += h

        n_frames += 1

    if n_frames == 0 or n_li is None:
        raise RuntimeError(f"No frames processed from {traj_path}")

    # Normalize to g(r)
    def normalize(hist, n_central, n_neighbor):
        """
        Normalize histogram to g(r).

        Parameters
        ----------
        hist : np.ndarray
            Raw pair-distance histogram.
        n_central : int
            Number of central atoms (Li).
        n_neighbor : int
            Number of neighbor atoms (O subset).

        Returns
        -------
        np.ndarray
            Normalized radial distribution function.
        """
        shell_vol = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        rho = n_neighbor / volume
        norm = n_central * n_frames * rho * shell_vol
        return hist / norm

    n_o_total = len(o_all_idx)
    n_o_water = len(o_water_idx)
    n_o_tfsi = len(o_tfsi_idx)

    gr_total = normalize(hist_total, n_li, n_o_total)
    gr_water = normalize(hist_water, n_li, n_o_water)
    gr_tfsi = normalize(hist_tfsi, n_li, n_o_tfsi)

    # Coordination numbers: integrate 4π ρ g(r) r² dr from 0 to R_CUT_COORD
    def coord_number(gr, n_neighbor):
        """
        Compute coordination number from g(r).

        Parameters
        ----------
        gr : np.ndarray
            Radial distribution function.
        n_neighbor : int
            Number of neighbor atoms (O subset).

        Returns
        -------
        float
            Integrated coordination number up to R_CUT_COORD.
        """
        rho = n_neighbor / volume
        integrand = 4.0 * np.pi * rho * gr * r_centers**2 * dr
        mask = r_centers <= R_CUT_COORD
        return float(np.sum(integrand[mask]))

    return {
        "r": r_centers.tolist(),
        "gr_LiO_total": gr_total.tolist(),
        "gr_LiO_water": gr_water.tolist(),
        "gr_LiO_TFSI": gr_tfsi.tolist(),
        "coord_LiO_total": coord_number(gr_total, n_o_total),
        "coord_LiO_water": coord_number(gr_water, n_o_water),
        "coord_LiO_TFSI": coord_number(gr_tfsi, n_o_tfsi),
        "n_li": n_li,
        "n_O_water": n_o_water,
        "n_O_TFSI": n_o_tfsi,
        "n_frames_used": n_frames,
        "r_cut_coord": R_CUT_COORD,
    }


def find_trajectory(model: str) -> Path | None:
    """
    Find NVT extxyz trajectory for a model.

    Parameters
    ----------
    model : str
        Name of the MLIP model.

    Returns
    -------
    Path or None
        Path to the trajectory file, or None if not found.
    """
    p = DATA_ROOT / model / "nvt_trajectory.extxyz"
    return p if p.exists() else None


# --- Pytest interface --------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_compute_rdf(model: str) -> None:
    """
    Compute and save Li-O RDFs and coordination numbers for one model.

    Parameters
    ----------
    model : str
        Name of the MLIP model to compute RDFs for.
    """
    traj_path = find_trajectory(model)
    if traj_path is None:
        pytest.skip(f"No NVT trajectory for {model}")

    # Identify O types from first frame (indices fixed throughout NVT)
    from ase.io import read as ase_read

    first_frame = ase_read(str(traj_path), index=0, format="extxyz")
    o_water_idx, o_tfsi_idx = identify_o_types(first_frame)

    assert len(o_water_idx) > 0, f"No O_water found for {model}"
    assert len(o_tfsi_idx) > 0, f"No O_TFSI found for {model}"

    result = compute_rdf(traj_path, o_water_idx, o_tfsi_idx)

    out_dir = OUT_PATH / model
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rdf.json", "w") as f:
        json.dump(
            {k: v for k, v in result.items() if not isinstance(v, list)}, f, indent=2
        )

    np.savez(
        out_dir / "rdf.npz",
        r=np.array(result["r"]),
        gr_LiO_total=np.array(result["gr_LiO_total"]),
        gr_LiO_water=np.array(result["gr_LiO_water"]),
        gr_LiO_TFSI=np.array(result["gr_LiO_TFSI"]),
    )

    # Sanity checks
    assert 2.0 < result["coord_LiO_total"] < 8.0, (
        f"Unexpected Li-O_total CN={result['coord_LiO_total']:.2f} for {model}"
    )
