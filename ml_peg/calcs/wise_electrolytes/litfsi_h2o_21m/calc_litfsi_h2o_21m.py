"""
Consolidated WiSE 21 m LiTFSI/H2O electrolyte benchmark.

Extracts three observables for each registered MLIP model:

* Density from pre-computed NPT thermo logs (p16_w42 cell, 382 atoms).
* Li-O coordination numbers from NVT trajectories via radial distribution
  functions g(r) for Li-O_water (O bonded to H, d_OH < 1.25 A) and
  Li-O_TFSI (O bonded to S, d_OS < 1.75 A); CN integrated to first minimum
  R_CUT = 2.83 A from the r2SCAN AIMD reference.
* X-ray structure factor S(q) computed via dynasor in TRAVIS Faber-Ziman
  convention with Cromer-Mann 4-Gaussian form factors (including H) and
  Savitzky-Golay smoothing (window=27, order=3, dq=0.02 A^-1).

System: 21 m LiTFSI / H2O, p64_w170 cell (1534 atoms, 27.4938 A cubic) for
RDF and S(q); p16_w42 (382 atoms) for NPT density. Trajectories produced
with LAMMPS + symmetrix on Adastra (MI250X); the Janus recast lives in
``../md_reference/calc_md_reference.py``.

Experimental references:

* Density: 1.7126 g/cm3 (Gilbert et al., J. Chem. Eng. Data 62, 2056, 2017).
* Li-O CN: 2.0 each for water/TFSI (Watanabe et al., JPCB 125, 7477, 2021,
  ~18.5 m, neutron diffraction with isotopic substitution).
* S(q): SAXS (Zhang et al., J. Phys. Chem. B 125, 4501, 2021).

Source trajectories must live under ``DATA_ROOT/<registry_model_name>/``,
where ``DATA_ROOT`` defaults to ``./data`` next to this script and is
overridable via the ``ML_PEG_WISE_LITFSI_H2O_21M_DATA_ROOT`` environment
variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import warnings

from ase.geometry import get_distances
from ase.io import iread, read as ase_read
import numpy as np
import pytest
from scipy.signal import savgol_filter

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

warnings.filterwarnings("ignore")

# --- Configuration -----------------------------------------------------------

MODELS = load_models(current_models)

DATA_ROOT = Path(
    os.environ.get(
        "ML_PEG_WISE_LITFSI_H2O_21M_DATA_ROOT",
        Path(__file__).parent / "data",
    )
)
OUT_PATH = Path(__file__).parent / "outputs"

# --- Density references ------------------------------------------------------

RHO_EXP = 1.7126  # g/cm3 (Gilbert et al., JCED 2017)

# --- RDF parameters ----------------------------------------------------------

R_MAX = 6.0  # A
DR = 0.02  # A
R_CUT_COORD = 2.83  # A — first minimum of Li-O g(r) from r2SCAN AIMD
D_OH_CUT = 1.25  # A — O-H bond cutoff for water identification
D_OS_CUT = 1.75  # A — O-S bond cutoff for TFSI identification

# --- S(q) parameters (TRAVIS-matching) ---------------------------------------

Q_MAX = 13.0  # A^-1
Q_MIN = 0.5  # A^-1
MAX_QPOINTS = 50000
DQ_BIN = 0.02  # A^-1
SAVGOL_WINDOW = 27
SAVGOL_ORDER = 3

# LAMMPS atom-type → element fallback for raw dump files
TYPE_TO_ELEMENT = {1: "Li", 2: "C", 3: "F", 4: "S", 5: "N", 6: "O", 7: "H"}

# System composition (p64_w170: 64 LiTFSI + 170 H2O)
COMPOSITION = {"Li": 64, "C": 128, "F": 384, "S": 128, "N": 64, "O": 426, "H": 340}
N_ATOMS = sum(COMPOSITION.values())  # 1534
CONC = {k: v / N_ATOMS for k, v in COMPOSITION.items()}

# Cromer-Mann 4-Gaussian X-ray form factor parameters (International Tables)
TRAVIS_FF = {
    "S": {"a": [6.905, 5.203, 1.438, 1.586],
          "b": [1.468, 22.215, 0.254, 56.172], "c": 0.867},
    "F": {"a": [3.539, 2.641, 1.517, 1.024],
          "b": [10.283, 4.294, 0.262, 26.148], "c": 0.278},
    "O": {"a": [3.049, 2.287, 1.546, 0.867],
          "b": [13.277, 5.701, 0.324, 32.909], "c": 0.251},
    "N": {"a": [12.213, 3.132, 2.013, 1.166],
          "b": [0.006, 9.893, 28.997, 0.583], "c": -11.529},
    "C": {"a": [2.310, 1.020, 1.589, 0.865],
          "b": [20.844, 10.208, 0.569, 51.651], "c": 0.216},
    "Li": {"a": [1.128, 0.751, 0.618, 0.465],
           "b": [3.955, 1.052, 85.391, 168.261], "c": 0.038},
    "H": {"a": [0.493, 0.323, 0.140, 0.041],
          "b": [10.511, 26.126, 3.142, 57.800], "c": 0.003},
}


# =============================================================================
# Density extraction
# =============================================================================


def _density_source_path(model_name: str) -> Path:
    """
    Locate the pre-computed density JSON for a model.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.

    Returns
    -------
    Path
        Expected path to ``density.json`` under ``DATA_ROOT``.
    """
    return DATA_ROOT / model_name / "density.json"


# =============================================================================
# RDF computation
# =============================================================================


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
    skip_frames: int = 0,
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
        Number of initial frames to skip (equilibration). Trajectories are
        already pre-equilibrated (50–100 ps window).

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


# =============================================================================
# X-ray S(q) computation
# =============================================================================


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


def compute_sq_travis_style(traj_path: Path) -> dict:
    """
    Compute S(q) in TRAVIS Faber-Ziman convention using dynasor.

    Steps:

    1. Read trajectory with dynasor (ASE for .extxyz; MDAnalysis for raw dump).
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

    ff_at_q = {at: compute_fq_travis(at, q_norms) for at in atom_types}
    i_xray = np.zeros(len(q_norms))
    for s1, s2 in sample.pairs:
        sq_ab = sample[f"Sq_{s1}_{s2}"].flatten()
        i_xray += ff_at_q[s1] * ff_at_q[s2] * sq_ab

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

    valid = ~np.isnan(sq_fz) & (q_centers >= 0.3) & (q_centers <= Q_MAX)
    q_v = q_centers[valid]
    sq_v = sq_fz[valid]
    if len(sq_v) > SAVGOL_WINDOW:
        sq_smooth = savgol_filter(sq_v, SAVGOL_WINDOW, SAVGOL_ORDER)
    else:
        sq_smooth = sq_v

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


def find_trajectory(model_name: str) -> Path | None:
    """
    Find NVT trajectory for a model.

    Prefers the converted ``.extxyz`` and falls back to the raw LAMMPS dump.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model.

    Returns
    -------
    Path or None
        Path to the trajectory file, or ``None`` if not found.
    """
    extxyz = DATA_ROOT / model_name / "nvt_trajectory.extxyz"
    if extxyz.exists():
        return extxyz

    lammpstrj = DATA_ROOT / model_name / "nvt_trajectory.lammpstrj"
    if lammpstrj.exists():
        return lammpstrj

    return None


# =============================================================================
# Pytest interface (ml-peg convention)
# =============================================================================


@pytest.mark.parametrize("model_name", MODELS)
def test_extract_density(model_name: str) -> None:
    """
    Extract and save NPT density data for one model.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    """
    src = _density_source_path(model_name)
    if not src.exists():
        pytest.skip(f"No density data for {model_name} at {src}")

    with open(src) as f:
        result = json.load(f)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "density.json", "w") as f:
        json.dump(result, f, indent=2)

    assert result["rho_mean"] > 0, f"Negative density for {model_name}"
    assert abs(result["rho_error_pct"]) < 50, (
        f"Density error > 50% for {model_name}"
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_compute_rdf(model_name: str) -> None:
    """
    Compute and save Li-O RDFs and coordination numbers for one model.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    """
    traj_path = find_trajectory(model_name)
    if traj_path is None:
        pytest.skip(f"No NVT trajectory for {model_name}")

    first_frame = ase_read(str(traj_path), index=0, format="extxyz")
    o_water_idx, o_tfsi_idx = identify_o_types(first_frame)

    assert len(o_water_idx) > 0, f"No O_water found for {model_name}"
    assert len(o_tfsi_idx) > 0, f"No O_TFSI found for {model_name}"

    result = compute_rdf(traj_path, o_water_idx, o_tfsi_idx)

    out_dir = OUT_PATH / model_name
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

    assert 2.0 < result["coord_LiO_total"] < 8.0, (
        f"Unexpected Li-O_total CN={result['coord_LiO_total']:.2f} "
        f"for {model_name}"
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_compute_xray_sq(model_name: str) -> None:
    """
    Compute and save X-ray S(q) in Faber-Ziman convention for one model.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    """
    traj_path = find_trajectory(model_name)
    if traj_path is None:
        pytest.skip(f"No NVT trajectory for {model_name}")

    result = compute_sq_travis_style(traj_path)
    result["model"] = model_name
    result["traj_path"] = str(traj_path)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "xray_sq.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(
        out_dir / "xray_sq.npz",
        q=np.array(result["q"]),
        Sq=np.array(result["Sq"]),
    )

    assert len(result["q"]) > 10, f"Too few q-points for {model_name}"
