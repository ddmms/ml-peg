"""
Compute X-ray structure factor S(q) for LiTFSI/H2O electrolyte.

Reads pre-computed NVT trajectories (.extxyz, 100 ps) for each MLIP model
and computes S(q) via dynasor using TRAVIS-matching settings: Cromer-Mann
4-Gaussian form factors (including H), Faber-Ziman normalization with Laue
monotonic term subtraction, and Savitzky-Golay smoothing
(window=27, order=3, dq=0.02 A^-1).

System: 21 m LiTFSI / H2O, p64_w170 cell (1534 atoms).
Experimental reference: SAXS, Zhang, Lewis, Mars, Wan et al.,
J. Phys. Chem. B 125, 4501 (2021), DOI: 10.1021/acs.jpcb.1c02189.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# --- Configuration -----------------------------------------------------------

# Path to converted .extxyz trajectories (will be S3 download for PR)
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

# S(q) computation parameters (TRAVIS-matching)
Q_MAX = 13.0  # Å⁻¹
Q_MIN = 0.5  # Å⁻¹
MAX_QPOINTS = 50000
DQ_BIN = 0.02  # Å⁻¹
SAVGOL_WINDOW = 27  # optimal for matching TRAVIS
SAVGOL_ORDER = 3

# LAMMPS type → element (for fallback)
TYPE_TO_ELEMENT = {1: "Li", 2: "C", 3: "F", 4: "S", 5: "N", 6: "O", 7: "H"}

# System composition (p64_w170: 64 LiTFSI + 170 H2O)
COMPOSITION = {"Li": 64, "C": 128, "F": 384, "S": 128, "N": 64, "O": 426, "H": 340}
N_ATOMS = sum(COMPOSITION.values())  # 1534
CONC = {k: v / N_ATOMS for k, v in COMPOSITION.items()}

# TRAVIS Cromer-Mann 4-Gaussian form factors (International Tables)
TRAVIS_FF = {
    "S": {
        "a": [6.905, 5.203, 1.438, 1.586],
        "b": [1.468, 22.215, 0.254, 56.172],
        "c": 0.867,
    },
    "F": {
        "a": [3.539, 2.641, 1.517, 1.024],
        "b": [10.283, 4.294, 0.262, 26.148],
        "c": 0.278,
    },
    "O": {
        "a": [3.049, 2.287, 1.546, 0.867],
        "b": [13.277, 5.701, 0.324, 32.909],
        "c": 0.251,
    },
    "N": {
        "a": [12.213, 3.132, 2.013, 1.166],
        "b": [0.006, 9.893, 28.997, 0.583],
        "c": -11.529,
    },
    "C": {
        "a": [2.310, 1.020, 1.589, 0.865],
        "b": [20.844, 10.208, 0.569, 51.651],
        "c": 0.216,
    },
    "Li": {
        "a": [1.128, 0.751, 0.618, 0.465],
        "b": [3.955, 1.052, 85.391, 168.261],
        "c": 0.038,
    },
    "H": {
        "a": [0.493, 0.323, 0.140, 0.041],
        "b": [10.511, 26.126, 3.142, 57.800],
        "c": 0.003,
    },
}


# --- Form factor helpers -----------------------------------------------------


def compute_fq_travis(elem: str, q_arr: np.ndarray) -> np.ndarray:
    """
    Compute X-ray form factor f(q) using TRAVIS 4-Gaussian (Cromer-Mann) params.

    Parameters
    ----------
    elem : str
        Chemical element symbol (e.g. ``"Li"``, ``"O"``).
    q_arr : np.ndarray
        Array of q values in inverse angstroms.

    Returns
    -------
    np.ndarray
        Form factor values evaluated at each q.
    """
    ff = TRAVIS_FF[elem]
    s2 = (q_arr / (4 * np.pi)) ** 2
    return sum(ff["a"][i] * np.exp(-ff["b"][i] * s2) for i in range(4)) + ff["c"]


# --- S(q) computation -------------------------------------------------------


def compute_sq_travis_style(traj_path: Path) -> dict:
    """
    Compute S(q) in TRAVIS Faber-Ziman convention using dynasor.

    Steps:

    1. Read trajectory with dynasor (LAMMPS dump via MDAnalysis).
    2. Compute partial S_ab(q) on a fine spherical q-grid.
    3. Apply TRAVIS Cromer-Mann form factors: I_xray = sum f_a * f_b * S_ab.
    4. Spherical binning at dq = 0.02 A^-1.
    5. Faber-Ziman normalization with Laue term: S_FZ = I/<f>^2 - <f^2>/<f>^2 + 1.
    6. Savitzky-Golay smoothing.

    Parameters
    ----------
    traj_path : Path
        Path to the trajectory file (.extxyz or LAMMPS dump).

    Returns
    -------
    dict
        Dictionary with keys ``q``, ``Sq``, ``n_qpoints``,
        ``n_qpoints_used``, ``cell``, ``atom_types``,
        ``particle_counts``, and ``params``.
    """
    from dynasor import (
        Trajectory,
        compute_static_structure_factors,
        get_spherical_qpoints,
    )

    traj_str = str(traj_path)
    is_extxyz = traj_str.endswith(".extxyz") or traj_str.endswith(".xyz")

    if is_extxyz:
        # For .extxyz: dynasor reads it directly, atomic_indices from element symbols
        from ase.io import read as ase_read

        first_frame = ase_read(traj_str, index=0)
        symbols = first_frame.get_chemical_symbols()
        atomic_indices = {}
        for i, sym in enumerate(symbols):
            atomic_indices.setdefault(sym, []).append(i)

        traj = Trajectory(
            traj_str,
            trajectory_format="ase",
            atomic_indices=atomic_indices,
        )
    else:
        # For LAMMPS dump: use MDAnalysis backend
        import MDAnalysis as mda  # noqa: N813

        u = mda.Universe(traj_str, format="LAMMPSDUMP")
        types = u.atoms.types
        atomic_indices = {}
        for t, elem in TYPE_TO_ELEMENT.items():
            mask = types == str(t)
            idx = np.where(mask)[0].tolist()
            if idx:
                atomic_indices[elem] = idx

        traj = Trajectory(
            traj_str,
            trajectory_format="lammps_mdanalysis",
            atomic_indices=atomic_indices,
        )

    q_points = get_spherical_qpoints(traj.cell, q_max=Q_MAX, max_points=MAX_QPOINTS)
    sample = compute_static_structure_factors(traj, q_points)

    q_norms = np.linalg.norm(sample.q_points, axis=1)
    atom_types = list(sample.particle_counts.keys())

    # Manual X-ray weighting with TRAVIS form factors (including H)
    ff_at_q = {at: compute_fq_travis(at, q_norms) for at in atom_types}
    i_xray = np.zeros(len(q_norms))
    for s1, s2 in sample.pairs:
        sq_ab = sample[f"Sq_{s1}_{s2}"].flatten()
        i_xray += ff_at_q[s1] * ff_at_q[s2] * sq_ab

    # Spherical binning
    q_bins = np.arange(0.0, Q_MAX + DQ_BIN, DQ_BIN)
    q_centers = 0.5 * (q_bins[:-1] + q_bins[1:])
    i_xray_binned = np.full(len(q_centers), np.nan)
    counts = np.zeros(len(q_centers), dtype=int)

    for i in range(len(q_centers)):
        mask = (q_norms >= q_bins[i]) & (q_norms < q_bins[i + 1])
        n = mask.sum()
        if n > 0:
            i_xray_binned[i] = np.mean(i_xray[mask])
            counts[i] = n

    # Faber-Ziman normalization with Laue subtraction
    f_avg = np.zeros(len(q_centers))
    f2_avg = np.zeros(len(q_centers))
    for elem, c in CONC.items():
        fq = compute_fq_travis(elem, q_centers)
        f_avg += c * fq
        f2_avg += c * fq**2
    f_avg_sq = f_avg**2

    sq_fz = np.where(
        f_avg_sq > 0,
        i_xray_binned / f_avg_sq - f2_avg / f_avg_sq + 1.0,
        np.nan,
    )

    # Savitzky-Golay smoothing
    valid = ~np.isnan(sq_fz) & (q_centers >= 0.3) & (q_centers <= Q_MAX)
    q_v = q_centers[valid]
    sq_v = sq_fz[valid]
    if len(sq_v) > SAVGOL_WINDOW:
        sq_smooth = savgol_filter(sq_v, SAVGOL_WINDOW, SAVGOL_ORDER)
    else:
        sq_smooth = sq_v

    # Restrict to useful range
    rmask = (q_v >= Q_MIN) & (q_v <= 12.0)

    return {
        "q": q_v[rmask].tolist(),
        "Sq": sq_smooth[rmask].tolist(),
        "n_qpoints": len(q_points),
        "n_qpoints_used": len(sample.q_points),
        "cell": traj.cell.tolist(),
        "atom_types": traj.atom_types,
        "particle_counts": {k: int(v) for k, v in sample.particle_counts.items()},
        "params": {
            "q_max": Q_MAX,
            "q_min": Q_MIN,
            "dq_bin": DQ_BIN,
            "max_qpoints": MAX_QPOINTS,
            "savgol_window": SAVGOL_WINDOW,
            "savgol_order": SAVGOL_ORDER,
            "form_factors": "cromer-mann-4gaussian",
            "normalization": "faber-ziman",
        },
    }


# --- Trajectory finding ------------------------------------------------------


def find_trajectory(model: str) -> Path | None:
    """
    Find NVT trajectory for a model.

    Prefers converted .extxyz, falls back to LAMMPS dump.

    Parameters
    ----------
    model : str
        Name of the MLIP model.

    Returns
    -------
    Path or None
        Path to the trajectory file, or None if not found.
    """
    # Check for converted .extxyz first
    extxyz = DATA_ROOT / model / "nvt_trajectory.extxyz"
    if extxyz.exists():
        return extxyz

    # Fall back to original LAMMPS dump
    runs_root = Path("/lus/work/CT9/cin1387/lbrugnoli/prove/runs")
    model_cell = runs_root / model / "p64_w170"
    for d in sorted(model_cell.glob("nvt_*"), reverse=True):
        traj = d / "nvt_long_trajectory.lammpstrj"
        if traj.exists():
            return traj
    for d in sorted(model_cell.glob("pipeline_*"), reverse=True):
        traj = d / "nvt_trajectory.lammpstrj"
        if traj.exists():
            return traj

    return None


# --- Pytest interface (ml-peg convention) ------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_compute_xray_sq(model: str) -> None:
    """
    Compute and save X-ray S(q) in Faber-Ziman convention for one model.

    Parameters
    ----------
    model : str
        Name of the MLIP model to compute S(q) for.
    """
    traj_path = find_trajectory(model)
    if traj_path is None:
        pytest.skip(f"No NVT trajectory for {model}")

    result = compute_sq_travis_style(traj_path)
    result["model"] = model
    result["traj_path"] = str(traj_path)

    out_dir = OUT_PATH / model
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "xray_sq.json", "w") as f:
        json.dump(result, f, indent=2)

    np.savez(
        out_dir / "xray_sq.npz",
        q=np.array(result["q"]),
        Sq=np.array(result["Sq"]),
    )

    assert len(result["q"]) > 10, f"Too few q-points for {model}"
