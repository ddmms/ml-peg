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
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import FIRE
from ase.units import bar, fs
import numpy as np

from ml_peg.calcs.liquids.ethanol_water_density.io_tools import DensityTimeseriesLogger


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
    nvt_stabilise_steps: int = 4_000,
    npt_settle_steps=7_500,
    nvt_thermalise_steps: int = 1_000,
    npt_equil_steps: int = 10_000,
    npt_prod_steps: int = 25_000,
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
            loc=0.9, scale=0.05, size=npt_prod_steps // sample_every
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

    # the used pre-relax is not good enough
    # do some Langevin NVT steps before starting NPT
    dyn = Langevin(atoms, timestep=dt, temperature_K=temperature, friction=0.02)
    attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)
    with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
        dyn.run(nvt_stabilise_steps)

    # quick Berendsen settle close to target density
    ps = 1000 * fs
    dyn = NPTBerendsen(
        atoms,
        timestep=dt,
        temperature_K=temperature,
        pressure_au=p_bar * bar,
        taut=0.07 * ps,
        taup=0.4 * ps,
        compressibility=4.5e-5,
    )
    attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)
    with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
        dyn.run(npt_settle_steps)

    # thermalise
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = Langevin(atoms, timestep=dt, temperature_K=temperature, friction=0.03)
    attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)
    with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
        dyn.run(nvt_thermalise_steps)

    # real NPT
    dyn = LangevinBAOAB(  # MTK
        atoms,
        timestep=dt,
        temperature_K=temperature,
        externalstress=p_bar * bar,
        T_tau=0.1 * ps,
        P_tau=1 * ps,
        hydrostatic=True,
        rng=0,
    )
    attach_basic_logging(dyn, atoms, str(workdir / "md.log"), log_every, t0)
    with traj_logging(dyn, atoms, workdir, traj_every=log_trajectory_every):
        dyn.run(npt_equil_steps)

        rhos = []
        n_samples = npt_prod_steps // sample_every
        with DensityTimeseriesLogger(ts_path) as density_log:
            for _ in range(n_samples):
                dyn.run(sample_every)
                rho = density_g_cm3(atoms)
                rhos.append(rho)
                density_log.write(rho)

    # save final structure for debugging/repro
    write(workdir / "final.extxyz", atoms)

    return np.array(rhos)
