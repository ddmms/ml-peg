"""
Consolidated WiSE 21 m LiTFSI/H2O electrolyte benchmark.

The benchmark has two halves, which can be run together or independently.

**Reference MD** (``test_reference_md``, marked ``very_slow``) reproduces the
protocol that generated the reference data, and writes its products into
``DATA_ROOT/<model>/``:

* ``p64_w170`` (1534 atoms, 27.4938 A cubic): Min -> NVT 50 ps equilibration
  -> NVT 50 ps production, held at the experimental volume throughout. No NPT:
  S(q) and the coordination numbers are compared at the reference density
  rather than at each model's own. Produces ``nvt_trajectory.extxyz``.
* ``p16_w42`` (382 atoms): Min -> NVT 50 ps -> NPT 200 ps. NPT is used only on
  this smaller cell, where the density converges far more cheaply. Produces
  ``density.json``, averaged over the last 150 ps.

**Extraction** (the remaining tests, fast) reads those files back and computes
three observables for each registered MLIP model:

* Density from the NPT run (p16_w42).
* Li-O coordination numbers from the NVT trajectory via radial distribution
  functions g(r) for Li-O_water (O bonded to H, d_OH < 1.25 A) and
  Li-O_TFSI (O bonded to S, d_OS < 1.75 A); CN integrated to first minimum
  R_CUT = 2.83 A from the r2SCAN AIMD reference.
* X-ray structure factor S(q) computed via dynasor in TRAVIS Faber-Ziman
  convention with Cromer-Mann 4-Gaussian form factors (including H) and
  Savitzky-Golay smoothing (window=5, order=3, dq=0.02 A^-1; physical width 0.10 A^-1).

The published reference data were produced with LAMMPS + symmetrix on Adastra
(MI250X) following the same protocol; ``test_reference_md`` is the janus-core
recast of it, and lets the data be regenerated for any registered model.
Integrators are matched where janus-core exposes a choice: NVT uses the
Nose-Hoover chain (``NVT_NH``, equivalent to LAMMPS ``fix nvt``) and NPT the
Martyna-Tobias-Klein chain (``NPT_MTK``, the formulation used by LAMMPS
``fix npt``; plain ``NPT`` in janus-core is Melchionna and would not match).

Experimental references:

* Density: 1.7126 g/cm3 (Gilbert et al., J. Chem. Eng. Data 62, 2056, 2017).
* Li-O CN: 2.0 each for water/TFSI (Watanabe et al., JPCB 125, 7477, 2021,
  ~18.5 m, neutron diffraction with isotopic substitution).
* S(q): SAXS (Zhang et al., J. Phys. Chem. B 125, 4501, 2021).

Data live under ``DATA_ROOT/<registry_model_name>/``, where ``DATA_ROOT``
defaults to ``./data`` next to this script and is overridable via the
``ML_PEG_WISE_LITFSI_H2O_21M_DATA_ROOT`` environment variable. The starting
structures are expected in ``DATA_ROOT/structures/``.
"""

from __future__ import annotations

from copy import copy
import json
import os
from pathlib import Path
from typing import Any
import warnings

from ase import Atoms
from ase.geometry import get_distances
from ase.io import iread
from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np
import pytest
from scipy.signal import savgol_filter

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

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
SAVGOL_WINDOW = 5
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

# --- Reference MD protocol ---------------------------------------------------

TEMPERATURE_K = 298.15
PRESSURE_BAR = 1.01325  # 1 atm
PRESSURE_GPA = PRESSURE_BAR * 1e-4  # janus-core takes pressure in GPa
TIMESTEP_FS = 0.5

# Nose-Hoover damping times (LAMMPS TDAMP/PDAMP: 100*dt and 1000*dt)
THERMOSTAT_TIME_FS = 50.0
BAROSTAT_TIME_FS = 500.0
NH_CHAIN = 3  # LAMMPS default for both thermostat and barostat sub-chains

# Step counts at 0.5 fs/step
NVT_EQUIL_STEPS = 100_000  # 50 ps, both cells
NVT_PROD_STEPS = 100_000  # 50 ps, p64_w170 only (S(q), RDF)
NPT_PROD_STEPS = 400_000  # 200 ps, p16_w42 only (density)

# IO cadence (LAMMPS THERMO_EVERY / DUMP_EVERY)
STATS_EVERY = 100  # 0.05 ps
TRAJ_EVERY = 200  # 0.1 ps -> 501 frames over the 50 ps production run

# Minimization (LAMMPS: min_style cg; minimize 1.0e-6 0.2 2000 20000)
MIN_FMAX = 0.2  # eV/A
MIN_STEPS = 2000

SEED = 42

# Density is averaged over the last 150 ps of the 200 ps NPT run.
DENSITY_WINDOW_PS = (50.0, 200.0)

STRUCTURE_DIR = DATA_ROOT / "structures"
CELL_P64 = "p64_w170"
CELL_P16 = "p16_w42"


# =============================================================================
# Reference MD (slow: regenerates the data consumed by the extraction below)
# =============================================================================


def load_initial_structure(cell: str) -> Atoms:
    """
    Load a packed starting configuration at the experimental density.

    Parameters
    ----------
    cell : str
        Cell name, ``"p64_w170"`` (1534 atoms) or ``"p16_w42"`` (382 atoms).

    Returns
    -------
    Atoms
        ASE Atoms at rho = 1.7126 g/cm3.

    Raises
    ------
    FileNotFoundError
        If the structure is not available locally.
    """
    path = STRUCTURE_DIR / f"{cell}_initial.xyz"
    if not path.exists():
        raise FileNotFoundError(
            f"Starting structure for {cell} not found at {path}. "
            "Both cells (p64_w170, p16_w42) are needed to run the reference MD."
        )
    return ase_read(path)


def _run_stage(
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
        janus-core ensemble class, e.g. ``NVT_NH`` or ``NPT_MTK``.
    struct : Atoms
        Structure to propagate.
    steps : int
        Number of steps for this stage.
    file_prefix : Path
        Prefix for this stage's output files.
    **kwargs
        Further arguments forwarded to the ensemble class.

    Returns
    -------
    Atoms
        The propagated structure, to hand on to the next stage.
    """
    final_file = Path(f"{file_prefix}-final.extxyz")
    if final_file.exists():
        done = ase_read(final_file)
        done.calc = struct.calc
        return done

    md = ensemble_cls(
        struct=struct,
        steps=steps,
        temp=TEMPERATURE_K,
        timestep=TIMESTEP_FS,
        thermostat_time=THERMOSTAT_TIME_FS,
        stats_every=STATS_EVERY,
        seed=SEED,
        file_prefix=file_prefix,
        **kwargs,
    )
    md.run()

    # Return the structure janus-core actually propagated rather than the one
    # passed in, which it is free to rebind.
    return md.struct


def _write_density_json(model_name: str, stats_file: Path, out_file: Path) -> dict:
    """
    Average the NPT density over ``DENSITY_WINDOW_PS`` and save it.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    stats_file : Path
        janus-core stats file of the NPT stage.
    out_file : Path
        Where to write ``density.json``.

    Returns
    -------
    dict
        The density summary that was written.
    """
    from janus_core.helpers.stats import Stats

    stats = Stats(stats_file)
    time_ps = np.asarray(stats["Time"], dtype=float) / 1000.0  # fs -> ps
    rho = np.asarray(stats["Density"], dtype=float)

    lo, hi = DENSITY_WINDOW_PS
    window = rho[(time_ps >= lo) & (time_ps <= hi)]
    if window.size == 0:
        raise RuntimeError(
            f"No NPT samples in {lo}-{hi} ps for {model_name}; the run is too short."
        )

    summary = {
        "model": model_name,
        "cell": CELL_P16,
        "rho_exp": RHO_EXP,
        "rho_mean": float(window.mean()),
        "rho_std": float(window.std()),
        "rho_error_pct": float(100 * (window.mean() - RHO_EXP) / RHO_EXP),
        "rho_abs_error": float(abs(window.mean() - RHO_EXP)),
        "n_samples": int(window.size),
        "time_range_ps": [lo, hi],
        "time_full": time_ps.tolist(),
        "density_full": rho.tolist(),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    return summary


def run_reference_md(model_name: str, model: Any) -> None:
    """
    Run the reference protocol for one model and write its data products.

    Produces ``nvt_trajectory.extxyz`` (p64_w170) and ``density.json``
    (p16_w42) under ``DATA_ROOT/<model_name>/``, which the extraction tests
    then read back.

    Parameters
    ----------
    model_name : str
        Registry name of the MLIP model.
    model : Any
        Model object from :func:`load_models`.
    """
    # Imported lazily: janus-core pulls in torch, which is not needed by the
    # extraction tests.
    from janus_core.calculations.geom_opt import GeomOpt
    from janus_core.calculations.md import NPT_MTK, NVT_NH

    calc = model.get_calculator(precision="high")

    data_dir = DATA_ROOT / model_name
    work_dir = OUT_PATH / model_name / "reference_md"
    work_dir.mkdir(parents=True, exist_ok=True)

    for cell in (CELL_P64, CELL_P16):
        struct = load_initial_structure(cell)
        struct.calc = copy(calc)

        # filter_class=None relaxes the atoms only: LAMMPS `minimize` leaves the
        # cell alone, and both cells must start from the experimental volume.
        GeomOpt(
            struct=struct,
            fmax=MIN_FMAX,
            optimizer="FIRE",
            steps=MIN_STEPS,
            filter_class=None,
            write_traj=False,
            file_prefix=work_dir / f"{cell}-minimize",
        ).run()

        # The equilibration trajectory is written but not analysed, as in the
        # LAMMPS pipeline. (traj_every=0 is not a way to disable it: ASE reads a
        # non-positive interval as "call once at step 0", which would divide by
        # zero in janus-core's trajectory writer.)
        struct = _run_stage(
            NVT_NH,
            struct,
            steps=NVT_EQUIL_STEPS,
            file_prefix=work_dir / f"{cell}-nvt_equil",
            traj_every=TRAJ_EVERY,
        )

        if cell == CELL_P64:
            # Production at the experimental volume: this is what S(q) and the
            # RDF are computed from.
            prefix = work_dir / f"{cell}-nvt_prod"
            _run_stage(
                NVT_NH,
                struct,
                steps=NVT_PROD_STEPS,
                file_prefix=prefix,
                traj_every=TRAJ_EVERY,
            )
            traj = ase_read(f"{prefix}-traj.extxyz", index=":")
            data_dir.mkdir(parents=True, exist_ok=True)
            ase_write(data_dir / "nvt_trajectory.extxyz", traj, format="extxyz")
        else:
            # NPT only on the small cell, for the density.
            prefix = work_dir / f"{cell}-npt_prod"
            _run_stage(
                NPT_MTK,
                struct,
                steps=NPT_PROD_STEPS,
                file_prefix=prefix,
                pressure=PRESSURE_GPA,
                barostat_time=BAROSTAT_TIME_FS,
                thermostat_chain=NH_CHAIN,
                barostat_chain=NH_CHAIN,
                traj_every=TRAJ_EVERY,
            )
            _write_density_json(
                model_name,
                Path(f"{prefix}-stats.dat"),
                data_dir / "density.json",
            )


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_reference_md(mlip: tuple[str, Any]) -> None:
    """
    Regenerate the benchmark data for one model with the reference protocol.

    Around 300 ps of cumulative MD on 1534 and 382 atoms: expect on the order
    of a day of GPU time per model.

    Parameters
    ----------
    mlip : tuple[str, Any]
        ``(model_name, model)`` pair from the ml-peg registry.
    """
    model_name, model = mlip
    run_reference_md(model_name, model)

    assert (DATA_ROOT / model_name / "nvt_trajectory.extxyz").exists()
    assert (DATA_ROOT / model_name / "density.json").exists()


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
        pytest.skip(
            f"No density data for {model_name} at {src}. Fetch the reference "
            "data or generate it with test_reference_md (-m very_slow)."
        )

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
        pytest.skip(
            f"No NVT trajectory for {model_name}. Fetch the reference data or "
            "generate it with test_reference_md (-m very_slow)."
        )

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
        pytest.skip(
            f"No NVT trajectory for {model_name}. Fetch the reference data or "
            "generate it with test_reference_md (-m very_slow)."
        )

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
