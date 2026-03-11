"""
Calculate densities using NPT of 63 liquids.

Liquids include 62 from arXiv:2505.06462 [physics.chem-ph],
in addition to 1-octanol.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any

from ase import Atoms, units
from ase.io import Trajectory, read
from ase.md.nose_hoover_chain import IsotropicMTKNPT
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"

AU_TO_G_CM3 = 1e24 / units.mol
G_CM3_TO_AU = 1 / AU_TO_G_CM3
NUM_MD_STEPS = 2
TIMESTEP = 1 * units.fs
LOG_INTERVAL = 100
ATM = 1.01325 * units.bar


def get_density_g_cm3(atoms: Atoms):
    """
    Get the density of the system in g/cm^3.

    Parameters
    ----------
    atoms
        ASE.Atoms object of the periodic system.

    Returns
    -------
    float
        Density in g/cm^3.
    """
    mass = atoms.get_masses().sum()
    volume = atoms.get_volume()
    return AU_TO_G_CM3 * mass / volume


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
        temperature_K=atoms.info["exp_temperature"],
        pressure_au=ATM,
        tdamp=50 * units.fs,
        pdamp=500 * units.fs,
        trajectory=output_fname,
        loginterval=LOG_INTERVAL,
        append_trajectory=True,
    )
    dyn.nsteps = nsteps
    dyn.attach(log_md, interval=LOG_INTERVAL, dyn=dyn, start_time=time.time())
    dyn.run(steps=NUM_MD_STEPS - nsteps)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_liquid_densities(mlip: tuple[str, Any], system_id) -> None:
    """
    Run Liquid Densities benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    system_id
        Identifier of the system to run MD on.
    """
    # Download data
    data_path = (
        download_s3_data(
            filename="Liquid_Densities.zip",
            key="inputs/molecular_dynamics/Liquid_Densities/Liquid_Densities.zip",
        )
        / "Liquid_Densities"
    )

    # Get system name
    input_xyz_path = sorted((data_path / "equilibrated_structures_xyz").glob("*.xyz"))[
        system_id
    ]
    system_name = input_xyz_path.stem
    OUT_PATH.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=OUT_PATH / f"{system_name}.log",
        filemode="a",
        force=True,
    )
    model_name, model = mlip
    # Use double precision
    model.default_dtype = "float32"
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    atoms = read(input_xyz_path)
    output_fname = OUT_PATH / f"{system_name}.traj"
    run_npt(atoms, calc, output_fname)
