"""code for md simulation."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any

from ase.io import Trajectory, read, write
from ase.md import Langevin, MDLogger
from ase.md.langevinbaoab import LangevinBAOAB
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
    """Return the mass in kg for ase atoms."""
    amu_to_kg = 1.66053906660e-27
    return atoms.get_masses().sum() * amu_to_kg


def density_g_cm3(atoms):
    """Return density in g/cm^3."""
    v_a3 = atoms.get_volume()
    v_m3 = v_a3 * 1e-30
    m_kg = total_mass_kg(atoms)
    rho_kg_m3 = m_kg / v_m3
    return rho_kg_m3 / 1000.0


def attach_basic_logging(dyn, atoms, md_logfile, log_every, t0):
    """Attach a logger to an ase md simulation."""
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
    """Context manager for logging trajectory."""
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
    dt_fs: float = 0.5,
    nvt_steps: int = 10_000,
    npt_steps: int = 50_000,
    sample_every: int = 20,
    log_every: int = 200,
    log_trajectory_every: int = 400,
    dummy_data=False,
    workdir: Path,
) -> Iterable[float]:
    """
    Run NPT and return (mean_density, std_density).

    TODO: use lammps? Though I would guess GPU is the bottleneck so it wouldn't matter?
    """
    ts_path = workdir / "density_timeseries.csv"
    if dummy_data:
        rho_series = np.random.normal(
            loc=0.9, scale=0.05, size=npt_steps // sample_every
        )
        with DensityTimeseriesLogger(ts_path) as density_log:
            for rho in rho_series:
                density_log.write(rho)
        return rho_series
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

    dt = dt_fs * fs
    t0 = time.time()
    ps = 1000 * fs
    T_tau = 0.5 * ps

    dyn = Langevin(atoms, timestep=dt, temperature_K=temperature, friction=1 / (T_tau))
    attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)
    with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
        dyn.run(nvt_steps)
    # real NPT
    dyn = LangevinBAOAB(  # use MTK?
        atoms,
        timestep=dt,
        temperature_K=temperature,
        externalstress=p_bar * bar,
        T_tau=T_tau,
        P_tau=0.5
        * ps,  # same timeconstants for baro/thermostat is fine for stochastic ones
        hydrostatic=True,
        rng=0,
    )
    attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)
    with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
        rhos = []
        n_samples = npt_steps // sample_every
        with DensityTimeseriesLogger(ts_path) as density_log:
            for _ in range(n_samples):
                dyn.run(sample_every)
                rho = density_g_cm3(atoms)
                rhos.append(rho)
                density_log.write(rho)

    # save final structure for debugging/repro
    write(workdir / "final.extxyz", atoms)

    return np.array(rhos)
