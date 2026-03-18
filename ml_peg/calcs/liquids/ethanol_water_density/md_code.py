"""Code for molecular-dynamics simulation workflows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any

from ase import units
from ase.io import Trajectory, read
from ase.md import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

NUM_NPT_STEPS = 1000_000
NUM_NVT_STEPS = 50_000
TIMESTEP = 1 * units.fs
LOG_INTERVAL = 100
ATM = 1.01325 * units.bar
TEMPERATURE = 298.15 * units.K
LANGEVIN_FRICTION = 1 / (500 * units.fs)


def get_density_g_cm3(atoms):
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
    amu_to_kg = 1.66053906660e-27
    v_a3 = atoms.get_volume()
    v_m3 = v_a3 * 1e-30
    m_kg = atoms.get_masses().sum() * amu_to_kg
    rho_kg_m3 = m_kg / v_m3
    return rho_kg_m3 / 1000.0


def log_md(dyn, start_time):
    """
    Log molecular dynamics simulation.

    Parameters
    ----------
    dyn
        ASE molecular dynamics object.
    start_time
        Real time of the simulation start, in seconds.
    """
    current_time = time.time() - start_time
    energy = dyn.atoms.get_potential_energy()
    density = get_density_g_cm3(dyn.atoms)
    temperature = dyn.atoms.get_temperature()
    t = dyn.get_time() / (1000 * units.fs)
    logging.info(
        f"""t: {t:>8.3f} ps\
            Walltime: {current_time:>10.3f} s\
            T: {temperature:.1f} K\
            Epot: {energy:.2f} eV\
            density: {density:.3f} g/cm^3\
        """
    )


def run_npt(atoms, calc, output_fname):
    """
    Run NPT molecular dynamics using the isotropic MTK barostat.

    Parameters
    ----------
    atoms
        ASE Atoms of the system.
    calc
        ASE Calculator.
    output_fname
        File name to save the trajectory to.
    """
    if os.path.exists(output_fname):
        try:
            traj = Trajectory(output_fname)
            atoms = traj[-1]
            nsteps = (len(traj) - 1) * LOG_INTERVAL
        except Exception as e:
            print(e)
            nsteps = 0
    else:
        nsteps = 0

    atoms.calc = calc

    dyn = IsotropicMTKNPT(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=TEMPERATURE,
        pressure_au=ATM,
        tdamp=50 * units.fs,
        pdamp=500 * units.fs,
        trajectory=output_fname,
        loginterval=LOG_INTERVAL,
        append_trajectory=True,
    )
    dyn.nsteps = nsteps
    dyn.attach(log_md, interval=LOG_INTERVAL, dyn=dyn, start_time=time.time())
    dyn.run(steps=NUM_NPT_STEPS)


def run_one_case(
    struct_path: str | Path,
    calc: Any,
    output_fname: str | Path,
):
    """
    Run the full MD workflow for one composition case.

    Parameters
    ----------
    struct_path : str | pathlib.Path
        Input structure path.
    calc : Any
        ASE-compatible calculator.
    output_fname : str | pathlib.Path
        The output file path.

    Returns
    -------
    collections.abc.Iterable[float]
        Density time series in g/cm^3.
    """
    atoms = read(struct_path)
    atoms.set_pbc(True)
    atoms.wrap()
    atoms.calc = calc

    # velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
    Stationary(atoms)
    ZeroRotation(atoms)
    if os.path.exists(output_fname):
        # NVT
        dyn = Langevin(
            atoms,
            timestep=TIMESTEP,
            temperature_K=TEMPERATURE,
            friction=LANGEVIN_FRICTION,
        )
        dyn.run(NUM_NVT_STEPS)

    # NPT
    run_npt(atoms, calc, output_fname)
