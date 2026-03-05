"""Code for molecular-dynamics simulation workflows."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any

from ase.io import Trajectory, read, write
from ase.md import Langevin, MDLogger
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import FIRE
from ase.units import bar, fs
import numpy as np

from ml_peg.calcs.liquids.ethanol_water_density._io_tools import DensityTimeseriesLogger


def total_mass_kg(atoms):
    """
    Return atomic-system mass in kilograms.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.

    Returns
    -------
    float
        Total mass in kilograms.
    """
    amu_to_kg = 1.66053906660e-27
    return atoms.get_masses().sum() * amu_to_kg


def density_g_cm3(atoms):
    """
    Return density in g/cm^3.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration with periodic cell volume.

    Returns
    -------
    float
        Density in g/cm^3.
    """
    v_a3 = atoms.get_volume()
    v_m3 = v_a3 * 1e-30
    m_kg = total_mass_kg(atoms)
    rho_kg_m3 = m_kg / v_m3
    return rho_kg_m3 / 1000.0


def attach_basic_logging(dyn, atoms, md_logfile, log_every, t0):
    """
    Attach text and progress loggers to an ASE dynamics object.

    Parameters
    ----------
    dyn : ase.md.md.MolecularDynamics
        Dynamics object to attach callbacks to.
    atoms : ase.Atoms
        Current atomic system.
    md_logfile : str | pathlib.Path
        Path to ASE MD log file.
    log_every : int
        Logging interval in MD steps.
    t0 : float
        Start timestamp from ``time.time()``.

    Returns
    -------
    None
        This function mutates ``dyn`` by attaching callbacks.
    """
    logger = MDLogger(
        dyn,
        atoms,
        md_logfile,
        header=True,
        stress=False,
        peratom=False,
        mode="a",
    )
    dyn.attach(logger, interval=log_every)

    def progress():
        """Print one progress line with thermodynamic state."""
        step = dyn.get_number_of_steps()
        rho = density_g_cm3(atoms)
        volume = atoms.get_volume()
        temperature = atoms.get_temperature()
        elapsed = time.time() - t0

        print(
            f"[step {step:>8}] "
            f"T={temperature:7.2f} K | "
            f"V={volume:10.2f} A^3 | "
            f"rho={rho:7.4f} g/cm^3 | "
            f"elapsed={elapsed:6.1f}s"
        )

    dyn.attach(progress, interval=log_every)


@contextmanager
def traj_logging(dyn, atoms, workdir, traj_every: int, name="md.traj"):
    """
    Attach trajectory logging inside a context manager.

    Parameters
    ----------
    dyn : ase.md.md.MolecularDynamics
        Dynamics object receiving callback.
    atoms : ase.Atoms
        Atomic system written to trajectory.
    workdir : pathlib.Path
        Output directory.
    traj_every : int
        Trajectory write interval in steps.
    name : str, optional
        Trajectory filename within ``workdir``.

    Yields
    ------
    ase.io.trajectory.Trajectory | None
        Open trajectory handle when enabled, otherwise ``None``.
    """
    traj = None
    if traj_every and traj_every > 0:
        traj = Trajectory(str(workdir / name), "a", atoms)
        dyn.attach(traj.write, interval=traj_every)
    try:
        yield traj
    finally:
        if traj is not None:
            traj.close()


def run_one_case(
    struct_path: Path,
    calc: Any,
    *,
    temperature: float = 298.15,
    p_bar: float = 1.0,
    dt_fs: float = 1.0,
    nvt_steps: int = 50_000,
    npt_steps: int = 1000_000,
    sample_every: int = 20,
    log_every: int = 250,
    log_trajectory_every: int = 500,
    dummy_data=False,
    continue_running=False,
    workdir: Path,
) -> Iterable[float]:
    """
    Run the full MD workflow for one composition case.

    Parameters
    ----------
    struct_path : pathlib.Path
        Input structure path.
    calc : Any
        ASE-compatible calculator.
    temperature : float, optional
        Target temperature in kelvin.
    p_bar : float, optional
        Target pressure in bar.
    dt_fs : float, optional
        Time step in femtoseconds.
    nvt_steps : int, optional
        Initial NVT stabilisation and mixing steps.
    npt_steps : int, optional
        NPT steps.
    sample_every : int, optional
        Sampling interval for density collection.
    log_every : int, optional
        Logging interval in MD steps.
    log_trajectory_every : int, optional
        Trajectory write interval in MD steps.
    dummy_data : bool, optional
        If ``True``, skip simulation and generate synthetic data.
    continue_running : bool, optional
        If ``True``, continue running from the previous trajectory.
    workdir : pathlib.Path
        Output directory for logs and trajectories.

    Returns
    -------
    collections.abc.Iterable[float]
        Density time series in g/cm^3.
    """
    ts_path = workdir / "density_timeseries.csv"
    traj_path = workdir / "md.traj"

    if dummy_data:
        rho_series = np.random.normal(
            loc=0.9, scale=0.05, size=npt_steps // sample_every
        )
        with DensityTimeseriesLogger(ts_path) as density_log:
            for rho in rho_series:
                density_log.write(rho)
        return rho_series

    workdir.mkdir(parents=True, exist_ok=True)

    dt = dt_fs * fs
    t0 = time.time()
    ps = 1000 * fs
    thermostat_tau = 50 * ps
    barostat_tau = 100 * ps

    target_samples = npt_steps // sample_every

    if continue_running:
        # 1) Load last saved state (positions + cell + momenta if stored)
        if traj_path.exists():
            with Trajectory(str(traj_path), "r") as tr:
                if len(tr) == 0:
                    raise RuntimeError(f"{traj_path} exists but contains no frames.")
                atoms = tr[-1]
        else:
            raise RuntimeError(f"No traj path at {traj_path}. Cannot continue running")

        # 2) Count how many samples are already present
        if ts_path.exists():
            # assume one rho per non-empty line; ignore a header if present
            n_lines = 0
            with open(ts_path, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    # skip header-ish lines
                    if any(c.isalpha() for c in s):
                        continue
                    n_lines += 1
            already_samples = n_lines
        else:
            raise RuntimeError(f"no ts_path at {ts_path}. Cannot continue running")

        # If we've already finished, just return what we have
        if already_samples >= target_samples:
            # load and return existing rhos
            rhos_existing = []
            if ts_path.exists():
                with open(ts_path, encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or any(c.isalpha() for c in s):
                            continue
                        # tolerate csv with extra columns
                        rhos_existing.append(float(s.split(",")[0]))
            return np.array(rhos_existing)
        atoms.calc = calc
    else:
        already_samples = 0
        atoms = read(struct_path)
        atoms.set_pbc(True)
        atoms.wrap()
        atoms.calc = calc

        # fast pre-relax
        opt = FIRE(atoms, logfile=str(workdir / "opt.log"))
        opt.run(fmax=0.15)

        # velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms)
        ZeroRotation(atoms)

        # NVT
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            friction=1 / (500 * fs),
        )
        attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)
        with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
            dyn.run(nvt_steps)

    # NPT
    dyn = IsotropicMTKNPT(
        atoms,
        timestep=dt,
        temperature_K=temperature,
        pressure_au=p_bar * bar,  # TODO check units !!!
        tdamp=thermostat_tau,
        pdamp=barostat_tau,
    )
    dyn.nsteps = already_samples * sample_every  # seems public enough to me

    attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)

    remaining_samples = target_samples - already_samples
    rhos = []

    with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
        with DensityTimeseriesLogger(
            ts_path, overwrite=not continue_running
        ) as density_log:
            for _ in range(remaining_samples):
                dyn.run(sample_every)
                rho = density_g_cm3(atoms)
                rhos.append(rho)
                density_log.write(rho)

    # save final structure for debugging/repro
    write(workdir / "final.extxyz", atoms)

    # If resuming, return the whole series (existing + new) for convenience
    if continue_running and ts_path.exists():
        rhos_all = []
        with open(ts_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or any(c.isalpha() for c in s):
                    continue
                rhos_all.append(float(s.split(",")[0]))
        return np.array(rhos_all)

    return np.array(rhos)
