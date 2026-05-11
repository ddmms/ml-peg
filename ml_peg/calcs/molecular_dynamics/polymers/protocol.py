"""
Run the simpoly 21-step Polymatic-style equilibration of polymer cells in ASE.

ASE translation of ``simpoly.poly_arena.simulation`` (LAMMPS). The protocol
mirrors simpoly's ``build_21steps_protocol`` and consists of 24 sequential
stages: geometry minimization, Maxwell-Boltzmann velocity initialization, two
NVT preheats, three "upward shaking" cycles at increasing high pressures
(each NPT-NVT-NVT), three "downward shaking" cycles at decreasing high
pressures, an 800 ps NPT equilibration, and a 500 ps NPT production stage.

Compared with simpoly, the LAMMPS ``nve_preheat`` stage (``langevin`` +
``nve/limit 0.1`` for 10 ps) is omitted: ASE has no ``nve/limit`` analogue
and ML potentials produce smooth-enough forces that the violent-overlap
protection is unnecessary.

Reference: Abbott, L. J.; Hart, K. E.; Colina, C. M. Polymatic: A Generalized
Simulated Polymerization Algorithm for Amorphous Polymers. Theor Chem Acc
2013, 132 (3), 1334. https://doi.org/10.1007/s00214-013-1334-z

LAMMPS ``npt aniso`` (independent x/y/z box lengths, no shear) maps to
``ase.md.nose_hoover_chain.MaskedMTKNPT(mask=(True, True, True))``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import time
import typing as ty

import ase
import ase.calculators.calculator as ase_calc
import ase.io.trajectory as ase_traj
import ase.md.md as ase_md
import ase.md.nose_hoover_chain as ase_nhc
import ase.md.velocitydistribution as ase_vd
import ase.optimize as ase_opt
import ase.units as ase_units
import numpy as np

LOG = logging.getLogger(__name__)

TIMESTEP: ty.Final[float] = 0.5 * ase_units.fs
LOG_INTERVAL: ty.Final[int] = 100
ATM: ty.Final[float] = 1.01325 * ase_units.bar
ROOM_TEMP_K: ty.Final[float] = 293.15
AU_TO_G_CM3: ty.Final[float] = 1e24 / ase_units.mol
N_DAMPING_STEPS: ty.Final[int] = 100
TDAMP: ty.Final[float] = N_DAMPING_STEPS * TIMESTEP
PDAMP: ty.Final[float] = 10 * N_DAMPING_STEPS * TIMESTEP
P_MAX_ATM: ty.Final[float] = 49346.2  # 50 000 bar in atm
MIN_FMAX: ty.Final[float] = 0.1  # eV / Å
MIN_STEPS: ty.Final[int] = 200

StageFn = ty.Callable[[ase.Atoms, pathlib.Path], ase.Atoms]


def _density_g_cm3(atoms: ase.Atoms) -> float:
    """
    Compute the density of ``atoms`` in g/cm³.

    Parameters
    ----------
    atoms
        Periodic structure.

    Returns
    -------
    float
        Density in g/cm³.
    """
    mass = float(atoms.get_masses().sum())
    volume = float(atoms.get_volume())
    return float(AU_TO_G_CM3 * mass / volume)


def _log_md(dyn: ase_md.MolecularDynamics, start_time: float) -> None:
    """
    Log step, T, ρ, walltime, KE, PE per ``LOG_INTERVAL`` steps.

    The format mirrors the per-line schema used by ``liquid_densities`` and
    ``water_density`` so the existing log-parsing pattern carries over.

    Parameters
    ----------
    dyn
        Active ASE integrator (the source of step / time / atoms).
    start_time
        Wall-clock seconds at which the stage was launched (``time.time()``).
    """
    wall_time = time.time() - start_time
    energy = dyn.atoms.get_potential_energy()
    density = _density_g_cm3(dyn.atoms)
    temperature = dyn.atoms.get_temperature()
    t_ps = dyn.get_time() / (1000 * ase_units.fs)
    LOG.info(
        f"t: {t_ps:>8.3f} ps  Walltime: {wall_time:>10.3f} s  "
        f"T: {temperature:.1f} K  Epot: {energy:.2f} eV  "
        f"density: {density:.3f} g/cm^3"
    )


def _state_path(out_dir: pathlib.Path) -> pathlib.Path:
    """
    Return the path to the per-polymer state.json file.

    Parameters
    ----------
    out_dir
        Per-polymer output directory.

    Returns
    -------
    pathlib.Path
        ``out_dir / "state.json"``.
    """
    return out_dir / "state.json"


def _load_completed(out_dir: pathlib.Path) -> set[str]:
    """
    Load the set of completed stage names from ``state.json`` (or empty).

    Parameters
    ----------
    out_dir
        Per-polymer output directory containing (or about to contain)
        ``state.json``.

    Returns
    -------
    set[str]
        Stage names that previous runs have already finished.
    """
    path = _state_path(out_dir)
    if not path.exists():
        return set()
    with open(path, encoding="utf8") as f:
        return set(json.load(f).get("completed", []))


def _save_completed(out_dir: pathlib.Path, completed: set[str]) -> None:
    """
    Atomically write the completed-stage list to ``state.json``.

    Parameters
    ----------
    out_dir
        Per-polymer output directory.
    completed
        Names of stages already finished.
    """
    path = _state_path(out_dir)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)
    tmp.replace(path)


def _last_frame(traj_path: pathlib.Path) -> ase.Atoms:
    """
    Return the last frame from a trajectory, with calculator detached.

    Parameters
    ----------
    traj_path
        ASE trajectory file to read.

    Returns
    -------
    ase.Atoms
        The final frame of ``traj_path``, with ``atoms.calc`` set to ``None``.
    """
    traj = ase_traj.Trajectory(str(traj_path))
    atoms = traj[-1]
    atoms.calc = None
    return atoms


def _resumed_nsteps(traj_path: pathlib.Path) -> int:
    """
    Return the number of MD steps already completed in ``traj_path``.

    Uses ``LOG_INTERVAL`` because each integrator dumps one frame every
    ``LOG_INTERVAL`` steps. Returns 0 if the file is missing or unreadable.

    Parameters
    ----------
    traj_path
        Per-stage trajectory path.

    Returns
    -------
    int
        Number of MD steps already represented in the trajectory.
    """
    if not traj_path.exists():
        return 0
    try:
        traj = ase_traj.Trajectory(str(traj_path))
    except (OSError, ValueError, RuntimeError) as err:
        LOG.warning(f"Could not read {traj_path} for resume ({err}); restarting stage")
        return 0
    return max(len(traj) - 1, 0) * LOG_INTERVAL


def _n_md_steps(time_ps: float) -> int:
    """
    Convert a stage duration in picoseconds into a number of MD steps.

    Parameters
    ----------
    time_ps
        Stage duration in picoseconds.

    Returns
    -------
    int
        Number of integrator steps for the stage at ``TIMESTEP``.
    """
    return int(math.ceil(time_ps * 1000 * ase_units.fs / TIMESTEP))


def _minimize(
    atoms: ase.Atoms,
    calc: ase_calc.Calculator,
    traj_path: pathlib.Path,
) -> ase.Atoms:
    """
    Geometry-optimize ``atoms`` and write the relaxed frame to ``traj_path``.

    Parameters
    ----------
    atoms
        Starting structure (modified in place).
    calc
        ASE Calculator attached to ``atoms`` for the optimization.
    traj_path
        Trajectory file to overwrite with the single relaxed frame.

    Returns
    -------
    ase.Atoms
        The optimized atoms object.
    """
    atoms.calc = calc
    opt = ase_opt.LBFGS(atoms, logfile=os.devnull, maxstep=0.05)
    opt.run(fmax=MIN_FMAX, steps=MIN_STEPS)

    with ase_traj.Trajectory(str(traj_path), "w", atoms=atoms) as traj:
        traj.write()
    return atoms


def _set_velocity(
    atoms: ase.Atoms,
    traj_path: pathlib.Path,
    *,
    temp_k: float,
    seed: int,
) -> ase.Atoms:
    """
    Initialize velocities from Maxwell-Boltzmann; zero net translation and rotation.

    Parameters
    ----------
    atoms
        Structure whose ``momenta`` will be populated.
    traj_path
        Trajectory file to overwrite with the single post-init frame.
    temp_k
        Target temperature for the Maxwell-Boltzmann distribution, in K.
    seed
        Seed for the numpy random generator used by ASE.

    Returns
    -------
    ase.Atoms
        ``atoms`` with momenta set and net translation/rotation removed.
    """
    rng = np.random.default_rng(seed=seed)
    ase_vd.MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k, rng=rng)
    ase_vd.Stationary(atoms)
    ase_vd.ZeroRotation(atoms)

    with ase_traj.Trajectory(str(traj_path), "w", atoms=atoms) as traj:
        traj.write()
    return atoms


def _prepare_md_resume(
    atoms: ase.Atoms,
    calc: ase_calc.Calculator,
    traj_path: pathlib.Path,
    n_total: int,
) -> tuple[ase.Atoms, int] | None:
    """
    Compute resume state for an MD stage.

    Returns ``None`` if the stage is already complete. Otherwise returns
    ``(atoms, n_done)`` ready for an integrator to consume.

    Parameters
    ----------
    atoms
        Carry-over structure from the previous stage.
    calc
        ASE Calculator to attach to the (possibly reloaded) atoms.
    traj_path
        Trajectory file for this stage; if it exists, used for resume.
    n_total
        Total number of MD steps the stage is supposed to run.

    Returns
    -------
    tuple[ase.Atoms, int] | None
        ``(atoms, n_done)`` when the stage still has steps to run, or
        ``None`` if ``n_done >= n_total``.
    """
    n_done = _resumed_nsteps(traj_path)
    if n_done >= n_total:
        return None
    if n_done > 0:
        atoms = _last_frame(traj_path)
    atoms.calc = calc
    return atoms, n_done


def _run_md(
    *,
    dyn: ase_md.MolecularDynamics,
    n_total: int,
    n_done: int,
) -> None:
    """
    Run an integrator for the remaining steps, with thermo logging hooked in.

    Parameters
    ----------
    dyn
        Already-constructed ASE integrator.
    n_total
        Total number of steps the stage is configured for.
    n_done
        Number of steps already completed in the trajectory (for resume).
    """
    dyn.nsteps = n_done
    dyn.attach(_log_md, interval=LOG_INTERVAL, dyn=dyn, start_time=time.time())
    dyn.run(steps=n_total - n_done)


def _nvt(
    atoms: ase.Atoms,
    calc: ase_calc.Calculator,
    traj_path: pathlib.Path,
    *,
    time_ps: float,
    temp_k: float,
) -> ase.Atoms:
    """
    Run a Nose-Hoover-chain NVT stage.

    Parameters
    ----------
    atoms
        Carry-over structure from the previous stage.
    calc
        ASE Calculator to attach to ``atoms``.
    traj_path
        Per-stage trajectory path (created or appended for resume).
    time_ps
        Stage duration in picoseconds (already scaled by ``time_prefactor``).
    temp_k
        Target temperature, in K.

    Returns
    -------
    ase.Atoms
        The final-frame atoms after the stage.
    """
    n_total = _n_md_steps(time_ps)
    prep = _prepare_md_resume(atoms, calc, traj_path, n_total)
    if prep is None:
        return _last_frame(traj_path)
    atoms, n_done = prep

    dyn = ase_nhc.NoseHooverChainNVT(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=temp_k,
        tdamp=TDAMP,
        trajectory=str(traj_path),
        loginterval=LOG_INTERVAL,
        append_trajectory=True,
    )
    _run_md(dyn=dyn, n_total=n_total, n_done=n_done)
    return atoms


def _npt(
    atoms: ase.Atoms,
    calc: ase_calc.Calculator,
    traj_path: pathlib.Path,
    *,
    time_ps: float,
    temp_k: float,
    pressure_atm: float,
) -> ase.Atoms:
    """
    Run an anisotropic NPT stage matching LAMMPS ``npt aniso``.

    Parameters
    ----------
    atoms
        Carry-over structure from the previous stage.
    calc
        ASE Calculator to attach to ``atoms``.
    traj_path
        Per-stage trajectory path (created or appended for resume).
    time_ps
        Stage duration in picoseconds (already scaled by ``time_prefactor``).
    temp_k
        Target temperature, in K.
    pressure_atm
        Target pressure, in atm.

    Returns
    -------
    ase.Atoms
        The final-frame atoms after the stage.
    """
    n_total = _n_md_steps(time_ps)
    prep = _prepare_md_resume(atoms, calc, traj_path, n_total)
    if prep is None:
        return _last_frame(traj_path)
    atoms, n_done = prep

    dyn = ase_nhc.MaskedMTKNPT(
        mask=(True, True, True),
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=temp_k,
        pressure_au=pressure_atm * ATM,
        tdamp=TDAMP,
        pdamp=PDAMP,
        trajectory=str(traj_path),
        loginterval=LOG_INTERVAL,
        append_trajectory=True,
    )
    _run_md(dyn=dyn, n_total=n_total, n_done=n_done)
    return atoms


def run_polymer_protocol(
    atoms: ase.Atoms,
    calc: ase_calc.Calculator,
    out_dir: pathlib.Path,
    *,
    temp_final_k: float = ROOM_TEMP_K,
    p_final_atm: float = 1.0,
    time_prefactor: float = 1.0,
    seed: int = 42,
) -> None:
    """
    Run simpoly's 24-stage Polymatic-style equilibration in ASE.

    The schedule is materialised as a 24-element ``(name, callable)`` list so
    the science maps line-by-line to the published protocol. ``state.json``
    records completed stages; per-stage trajectories handle within-stage
    resumption.

    Parameters
    ----------
    atoms
        Starting polymer cell; must have a periodic box (3 PBC).
    calc
        ASE Calculator (will be attached to ``atoms``).
    out_dir
        Output directory (created if missing). Trajectories and ``state.json``
        are written here.
    temp_final_k
        Target equilibration temperature, in K.
    p_final_atm
        Target equilibration pressure, in atm. Default: 1.0.
    time_prefactor
        Multiplier on every stage duration. Default 1.0; use a small value
        (e.g. 0.05) for end-to-end smoke tests.
    seed
        Random-number seed for velocity initialization.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    completed = _load_completed(out_dir)
    tp = time_prefactor
    t_max = min(max(temp_final_k + 100.0, 700.0), 1000.0)
    t_fin = temp_final_k
    p_up_1, p_up_2, p_up_3 = 0.02 * P_MAX_ATM, 0.6 * P_MAX_ATM, 1.0 * P_MAX_ATM
    p_dn_1, p_dn_2, p_dn_3 = 0.5 * P_MAX_ATM, 0.1 * P_MAX_ATM, 0.01 * P_MAX_ATM

    # Per-kind factories that each return a `(name, callable(atoms, traj_path))`
    # tuple. The helpers (`_minimize`, `_set_velocity`, `_nvt`, `_npt`) have
    # heterogeneous signatures, so we capture `calc` / `seed` / per-stage
    # parameters in a closure here. This lets the schedule below stay one line
    # per stage, which is the entire point of having a procedural protocol.
    def _min(name: str) -> tuple[str, StageFn]:
        """
        Build a (name, callable) tuple for a minimization stage.

        Parameters
        ----------
        name
            Stage name used for the trajectory file and ``state.json``.

        Returns
        -------
        tuple[str, StageFn]
            ``(name, fn)`` for the schedule loop.
        """
        return name, lambda a, p: _minimize(a, calc, p)

    def _vel(name: str, *, t_k: float) -> tuple[str, StageFn]:
        """
        Build a (name, callable) tuple for a set-velocity stage.

        Parameters
        ----------
        name
            Stage name used for the trajectory file and ``state.json``.
        t_k
            Maxwell-Boltzmann temperature, in K.

        Returns
        -------
        tuple[str, StageFn]
            ``(name, fn)`` for the schedule loop.
        """
        return name, lambda a, p: _set_velocity(a, p, temp_k=t_k, seed=seed)

    def _nvt_(name: str, *, t_ps: float, t_k: float) -> tuple[str, StageFn]:
        """
        Build a (name, callable) tuple for an NVT stage.

        Parameters
        ----------
        name
            Stage name used for the trajectory file and ``state.json``.
        t_ps
            Stage duration in picoseconds.
        t_k
            Target temperature, in K.

        Returns
        -------
        tuple[str, StageFn]
            ``(name, fn)`` for the schedule loop.
        """
        return name, lambda a, p: _nvt(a, calc, p, time_ps=t_ps, temp_k=t_k)

    def _npt_(
        name: str, *, t_ps: float, t_k: float, p_atm: float
    ) -> tuple[str, StageFn]:
        """
        Build a (name, callable) tuple for an NPT stage.

        Parameters
        ----------
        name
            Stage name used for the trajectory file and ``state.json``.
        t_ps
            Stage duration in picoseconds.
        t_k
            Target temperature, in K.
        p_atm
            Target pressure, in atm.

        Returns
        -------
        tuple[str, StageFn]
            ``(name, fn)`` for the schedule loop.
        """
        return name, lambda a, p: _npt(
            a, calc, p, time_ps=t_ps, temp_k=t_k, pressure_atm=p_atm
        )

    schedule: list[tuple[str, StageFn]] = [
        _min("00_minimization"),
        _vel("01_set_velocity", t_k=t_max),
        _nvt_("02_step1_highT_preheat", t_ps=50 * tp, t_k=t_max),
        _nvt_("03_step2_lowT_preheat", t_ps=50 * tp, t_k=t_fin),
        _npt_("04_step3_upward_shaking_highP", t_ps=50 * tp, t_k=t_fin, p_atm=p_up_1),
        _nvt_("05_step4_upward_shaking_highT", t_ps=50 * tp, t_k=t_max),
        _nvt_("06_step5_upward_shaking_lowT", t_ps=100 * tp, t_k=t_fin),
        _npt_("07_step6_upward_shaking_highP", t_ps=50 * tp, t_k=t_fin, p_atm=p_up_2),
        _nvt_("08_step7_upward_shaking_highT", t_ps=50 * tp, t_k=t_max),
        _nvt_("09_step8_upward_shaking_lowT", t_ps=100 * tp, t_k=t_fin),
        _npt_("10_step9_upward_shaking_highP", t_ps=50 * tp, t_k=t_fin, p_atm=p_up_3),
        _nvt_("11_step10_upward_shaking_highT", t_ps=50 * tp, t_k=t_max),
        _nvt_("12_step11_upward_shaking_lowT", t_ps=100 * tp, t_k=t_fin),
        _npt_("13_step12_downward_shaking_highP", t_ps=5 * tp, t_k=t_fin, p_atm=p_dn_1),
        _nvt_("14_step13_downward_shaking_highT", t_ps=5 * tp, t_k=t_max),
        _nvt_("15_step14_downward_shaking_lowT", t_ps=10 * tp, t_k=t_fin),
        _npt_("16_step15_downward_shaking_highP", t_ps=5 * tp, t_k=t_fin, p_atm=p_dn_2),
        _nvt_("17_step16_downward_shaking_highT", t_ps=5 * tp, t_k=t_max),
        _nvt_("18_step17_downward_shaking_lowT", t_ps=10 * tp, t_k=t_fin),
        _npt_("19_step18_downward_shaking_highP", t_ps=5 * tp, t_k=t_fin, p_atm=p_dn_3),
        _nvt_("20_step19_downward_shaking_highT", t_ps=5 * tp, t_k=t_max),
        _nvt_("21_step20_downward_shaking_lowT", t_ps=10 * tp, t_k=t_fin),
        _npt_(
            "22_step21_npt_equilibration", t_ps=800 * tp, t_k=t_fin, p_atm=p_final_atm
        ),
        _npt_("23_step22_final_npt", t_ps=500 * tp, t_k=t_fin, p_atm=p_final_atm),
    ]

    for name, fn in schedule:
        traj_path = out_dir / f"{name}.traj"
        if name in completed:
            try:
                atoms = _last_frame(traj_path)
                continue
            except (OSError, ValueError, RuntimeError) as err:
                LOG.warning(
                    f"Stage {name} marked complete in state.json but its "
                    f"trajectory could not be loaded ({err}); re-running."
                )
                completed.discard(name)
                _save_completed(out_dir, completed)
        LOG.info(f"# Stage: {name}")
        atoms = fn(atoms, traj_path)
        completed.add(name)
        _save_completed(out_dir, completed)
