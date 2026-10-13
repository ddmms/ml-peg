"""
Utilities for the WiSE 21 m LiTFSI/H2O electrolyte benchmark.

This module provides the MD stage handling used by the reference protocol
(resuming a stage cut short by the walltime, skipping stages that finished),
the NPT density summary, and the structural observables computed from the NVT
trajectory: Li-O radial distribution functions, coordination numbers and the
X-ray structure factor S(q).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ase import Atoms
from ase.geometry import get_distances
from ase.io import iread
from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np
from scipy.signal import savgol_filter

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
# The q-grid of a periodic cell is quantised in units of 2*pi/L: 0.229 A^-1 for the
# p64_w170 box (27.49 A), eleven times coarser than the 0.02 A^-1 bin the spherical
# average is accumulated on. Below q ~ 2 A^-1 adjacent bins therefore sample different
# shells, whose degeneracies jump from 6 to 72 triplets, and the fine structure this
# produces is a finite-size effect rather than a property of the liquid. The filter has
# to be wider than the shell spacing: 27 points = 0.54 A^-1 = 2.4 * 2*pi/L. That is also
# the resolution at which TRAVIS returns S(q), since it transforms a g(r) truncated at
# L/2 (pi/r_max = 2*pi/L), and at this value dynasor and TRAVIS agree (RMSD 0.020,
# r = 0.996), whereas at 5 points (0.10 A^-1, less than one shell spacing) the filter
# averages nothing and redraws the quantisation of the box.
SAVGOL_WINDOW = 27
SAVGOL_ORDER = 3

# System composition (p64_w170: 64 LiTFSI + 170 H2O)
COMPOSITION = {"Li": 64, "C": 128, "F": 384, "S": 128, "N": 64, "O": 426, "H": 340}
N_ATOMS = sum(COMPOSITION.values())  # 1534
CONC = {k: v / N_ATOMS for k, v in COMPOSITION.items()}

# Cromer-Mann 4-Gaussian X-ray form factor parameters (International Tables)
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


# =============================================================================
# MD stages
# =============================================================================


def resume_state(file_prefix: Path, steps: int) -> tuple[Atoms, int] | None:
    """
    Find the restart point of a stage that a walltime cut short.

    janus-core writes ``{prefix}-res-{step}.extxyz`` every ``restart_every``
    steps, but :func:`run_stage` only skips stages that *finished*: without
    this, a stage killed by the walltime redoes all of its steps in the next
    job of the chain, and a stage longer than one walltime never finishes at
    all.

    The restart file is chosen by the step parsed from its name, never by
    mtime. janus-core's own auto-restart takes ``max(..., key=getmtime)``, and
    on a filesystem where an anti-purge cycle rewrites mtimes that resumes
    from an arbitrary frame. Statistics and trajectory are truncated back to
    that step, so that the offset janus-core reads from the last line of the
    statistics file matches the structure it is handed, and so the output it
    appends does not overlap what is already there.

    Returns ``None`` whenever anything fails to line up, which leaves the
    stage to start from scratch: losing a chunk is recoverable, resuming from
    an inconsistent state silently is not.

    Parameters
    ----------
    file_prefix : Path
        Prefix shared by this stage's output files.
    steps : int
        Total number of steps the stage must reach.

    Returns
    -------
    tuple[Atoms, int] | None
        Structure to resume from and the step it sits at, or None.
    """
    stats_file = Path(f"{file_prefix}-stats.dat")
    traj_file = Path(f"{file_prefix}-traj.extxyz")
    if not stats_file.exists() or not traj_file.exists():
        return None

    candidates = {}
    for path in Path(file_prefix).parent.glob(f"{Path(file_prefix).name}-res-*.extxyz"):
        tail = path.stem.rsplit("-", 1)[-1]
        if tail.isdigit():
            step = int(tail)
            if 0 < step < steps:
                candidates[step] = path
    if not candidates:
        return None
    step = max(candidates)

    # The statistics file is the offset janus-core will read back, so it has to
    # end exactly on the step the restart file holds.
    lines = stats_file.read_text().splitlines(keepends=True)
    kept = []
    found = False
    for line in lines:
        if line.startswith("#"):
            kept.append(line)
            continue
        first = line.split(maxsplit=1)
        if not first or not first[0].isdigit():
            continue
        if int(first[0]) > step:
            break
        kept.append(line)
        found = int(first[0]) == step
    if not found:
        return None

    frames = ase_read(traj_file, index=":")
    if any("step" not in frame.info for frame in frames):
        return None
    frames = [frame for frame in frames if int(frame.info["step"]) <= step]
    if not frames:
        return None

    resumed = ase_read(candidates[step])
    # The extxyz parser hands back info scalars as numpy types; orb's adapter
    # asserts charge/spin are Python float/int.
    for key in ("charge", "spin"):
        if key in resumed.info:
            resumed.info[key] = int(resumed.info[key])

    # Rewrite through temporaries: a job killed halfway through this must not
    # be able to leave a half-written history behind.
    tmp_stats = stats_file.with_suffix(stats_file.suffix + ".tmp")
    tmp_traj = traj_file.with_suffix(traj_file.suffix + ".tmp")
    tmp_stats.write_text("".join(kept))
    ase_write(tmp_traj, frames, format="extxyz")
    os.replace(tmp_stats, stats_file)
    os.replace(tmp_traj, traj_file)
    return resumed, step


def run_stage(
    ensemble_cls, struct: Atoms, *, steps: int, file_prefix: Path, **kwargs
) -> Atoms:
    """
    Run one MD stage, skipping it if it already completed.

    janus-core writes ``{file_prefix}-final.extxyz`` once a stage reaches its
    last step, so re-running the test picks up from the first stage that did
    not finish rather than repeating the whole protocol.

    Parameters
    ----------
    ensemble_cls
        The janus-core ensemble class, e.g. ``NVT_NH`` or ``NPT_MTK``.
    struct : Atoms
        Structure to propagate.
    steps : int
        Number of steps for this stage.
    file_prefix : Path
        Prefix for this stage's output files.
    **kwargs
        Further arguments forwarded to the ensemble class: temperature,
        timestep, thermostat/barostat settings and output cadence.

    Returns
    -------
    Atoms
        The propagated structure, to hand on to the next stage.
    """
    final_file = Path(f"{file_prefix}-final.extxyz")
    if final_file.exists():
        done = ase_read(final_file)
        # The extxyz parser hands back info scalars as numpy types; orb's
        # adapter asserts charge/spin are Python float/int, so a resumed run
        # crashes on its first evaluation. Coerce on every re-read.
        for key in ("charge", "spin"):
            if key in done.info:
                done.info[key] = int(done.info[key])
        done.calc = struct.calc
        return done

    resume = resume_state(Path(file_prefix), steps)
    if resume is not None:
        resumed, offset = resume
        resumed.calc = struct.calc
        struct = resumed
        kwargs = {**kwargs, "restart": True, "restart_auto": False}
        print(
            f"[resume] {Path(file_prefix).name}: from step {offset} to {steps}",
            flush=True,
        )

    md = ensemble_cls(
        struct=struct,
        steps=steps,
        file_prefix=file_prefix,
        **kwargs,
    )
    md.run()

    # Return the structure janus-core actually propagated rather than the one
    # passed in, which it is free to rebind.
    return md.struct


def write_density_json(
    model_name: str,
    stats_file: Path,
    out_file: Path,
    *,
    cell: str,
    rho_exp: float,
    window_ps: tuple[float, float],
) -> dict:
    """
    Average the NPT density over a time window and save it.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    stats_file : Path
        The janus-core stats file of the NPT stage.
    out_file : Path
        Where to write ``density.json``.
    cell : str
        Name of the NPT cell, recorded in the summary.
    rho_exp : float
        Experimental density in g/cm3.
    window_ps : tuple[float, float]
        Start and end of the averaging window in ps.

    Returns
    -------
    dict
        The density summary that was written.
    """
    from janus_core.helpers.stats import Stats

    stats = Stats(stats_file)
    time_ps = np.asarray(stats["Time"], dtype=float) / 1000.0  # fs -> ps
    rho = np.asarray(stats["Density"], dtype=float)

    lo, hi = window_ps
    window = rho[(time_ps >= lo) & (time_ps <= hi)]
    if window.size == 0:
        raise RuntimeError(
            f"No NPT samples in {lo}-{hi} ps for {model_name}; the run is too short."
        )

    summary = {
        "model": model_name,
        "cell": cell,
        "rho_exp": rho_exp,
        "rho_mean": float(window.mean()),
        "rho_std": float(window.std()),
        "rho_error_pct": float(100 * (window.mean() - rho_exp) / rho_exp),
        "rho_abs_error": float(abs(window.mean() - rho_exp)),
        "n_samples": int(window.size),
        "time_range_ps": [lo, hi],
        "time_full": time_ps.tolist(),
        "density_full": rho.tolist(),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    return summary


# =============================================================================
# RDF and coordination numbers
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
# X-ray S(q)
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

    1. Read the extxyz trajectory with dynasor (ASE reader).
    2. Compute partial S_ab(q) on a fine spherical q-grid.
    3. Apply TRAVIS Cromer-Mann form factors: I_xray = sum f_a * f_b * S_ab.
    4. Spherical binning at dq = 0.02 A^-1.
    5. Faber-Ziman normalization with Laue term: S_FZ = I/<f>^2 - <f^2>/<f>^2 + 1.
    6. Savitzky-Golay smoothing.

    Parameters
    ----------
    traj_path : Path
        Path to the ``.extxyz`` trajectory.

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
