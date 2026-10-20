"""
Calculate densities using NPT of 25 battery electrolyte systems.

Systems are Na-ion (plus one K-ion reference) battery electrolytes in glyme and
carbonate solvents, together with the corresponding neat solvents, taken from
arXiv:2603.20183. Reference data are experimental densities at 298.2 K.

Configurations were equilibrated under NPT with OPLS-AA and the OMol25-trained
UMA potential, and are distributed at
https://github.com/KMNitesh05/sodium-ion-battery-electrolyte-dataset
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.io import Trajectory, read
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
import pytest

from ml_peg.calcs.molecular_dynamics.battery_electrolyte_densities.protocol import (
    LOG_INTERVAL,
    NUM_MD_STEPS,
    TIMESTEP,
)
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

AU_TO_G_CM3 = 1e24 / units.mol
ATM = 1.01325 * units.bar

# Number of configurations in the dataset.
N_SYSTEMS = 25

# 150 ps of NPT per system: 50 ps equilibration, discarded during analysis, then
# 100 ps of production. This is the protocol proposed in ddmms/ml-peg#358, and is
# shorter than the 1 ns used by liquid_densities because these inputs arrive
# already NPT-equilibrated with a potential of comparable quality.

DATA_ZIP = "battery_electrolyte_densities.zip"
DATA_KEY = (
    "inputs/molecular_dynamics/battery_electrolyte_densities/"
    "battery_electrolyte_densities.zip"
)


def get_density_g_cm3(atoms: Atoms) -> float:
    """
    Get the density of the system in g/cm^3.

    Parameters
    ----------
    atoms
        ASE Atoms object of the periodic system.

    Returns
    -------
    float
        Density in g/cm^3.
    """
    mass = atoms.get_masses().sum()
    volume = atoms.get_volume()
    return AU_TO_G_CM3 * mass / volume


def log_md(dyn, start_time: float) -> None:
    """
    Log molecular dynamics simulation.

    The 15 whitespace-separated fields per line match the liquid_densities
    benchmark, so the same log parsing works for both.

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
            density: {density:.5f} g/cm^3\
        """
    )


def init_velocities(atoms: Atoms, temperature_k: float) -> None:
    """
    Give the system a Maxwell-Boltzmann velocity distribution if it has none.

    Some configurations in the dataset carry momenta from their equilibration
    run and some do not. Starting an NPT run from zero velocities would waste a
    large part of the trajectory on heating, so seed them here when absent.

    Parameters
    ----------
    atoms
        ASE Atoms object to initialise velocities for.
    temperature_k
        Target temperature in K.
    """
    if atoms.has("momenta") and np.any(atoms.get_momenta()):
        return

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)
    Stationary(atoms)


def run_npt(atoms: Atoms, calc, output_fname: Path) -> None:
    """
    Run NPT molecular dynamics using the isotropic MTK barostat.

    Restarts from the last frame of an existing trajectory, so a benchmark run
    that hit a walltime limit can be resumed by rerunning the same command.

    Parameters
    ----------
    atoms
        ASE Atoms of the system.
    calc
        ASE Calculator.
    output_fname
        File name to save the trajectory to.
    """
    output_fname = Path(output_fname)
    status_path = output_fname.with_suffix(".status.json")
    previous_complete = None
    if status_path.exists():
        try:
            previous_complete = json.loads(status_path.read_text())["complete"]
        except (ValueError, KeyError):
            previous_complete = False
    status_path.write_text(json.dumps({"complete": False}))
    nsteps = 0
    if output_fname.exists():
        try:
            with Trajectory(output_fname) as traj:
                restart_atoms = traj[-1]
                restart_steps = (len(traj) - 1) * LOG_INTERVAL
                if restart_steps > NUM_MD_STEPS:
                    raise ValueError("Trajectory exceeds the expected run length")
                if restart_steps == NUM_MD_STEPS and previous_complete is False:
                    raise ValueError("Previous run did not finish successfully")
                if (
                    not np.isfinite(restart_atoms.positions).all()
                    or not np.isfinite(restart_atoms.get_volume())
                    or restart_atoms.get_volume() <= 0
                    or not np.isfinite(restart_atoms.get_potential_energy())
                    or not np.isfinite(restart_atoms.get_forces()).all()
                ):
                    raise ValueError("Invalid restart state")
            atoms = restart_atoms
            nsteps = restart_steps
        except Exception as exc:
            backup = output_fname.with_name(
                f"{output_fname.name}.invalid-{time.time_ns()}"
            )
            output_fname.rename(backup)
            warn(
                f"Cannot resume {output_fname}: {exc}. Saved as {backup}", stacklevel=2
            )

    atoms.calc = calc
    # Set default charge and spin
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)

    temperature_k = atoms.info["exp_temperature"]
    if nsteps == 0:
        init_velocities(atoms, temperature_k)

    dyn = IsotropicMTKNPT(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=temperature_k,
        pressure_au=ATM,
        tdamp=50 * units.fs,
        pdamp=500 * units.fs,
        trajectory=output_fname,
        loginterval=LOG_INTERVAL,
        append_trajectory=nsteps > 0,
    )
    dyn.nsteps = nsteps
    dyn.attach(log_md, interval=LOG_INTERVAL, dyn=dyn, start_time=time.time())
    try:
        dyn.run(steps=NUM_MD_STEPS - nsteps)
    except Exception as exc:
        warn(f"Error running MD: {exc}", stacklevel=2)
        status_path.write_text(json.dumps({"complete": False, "error": str(exc)}))
    else:
        status_path.write_text(
            json.dumps({"complete": dyn.nsteps == NUM_MD_STEPS, "steps": dyn.nsteps})
        )
    finally:
        dyn.close()


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_battery_electrolyte_densities(mlip: tuple[str, Any], system_id: int) -> None:
    """
    Run battery electrolyte densities benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    system_id
        Identifier of the system to run MD on.
    """
    assert system_id in range(0, N_SYSTEMS), (
        f"system_id out of range. Please use a value from 0 to {N_SYSTEMS - 1}"
    )

    data_path = (
        download_s3_data(filename=DATA_ZIP, key=DATA_KEY)
        / "battery_electrolyte_densities"
    )
    structure_paths = sorted((data_path / "structures").glob("*.xyz"))
    if len(structure_paths) != N_SYSTEMS:
        raise ValueError(
            f"Expected {N_SYSTEMS} structures in {data_path / 'structures'}, "
            f"found {len(structure_paths)}"
        )
    input_xyz_path = structure_paths[system_id]

    model_name, model = mlip
    calc = model.get_calculator(precision="low")
    calc = model.add_d3_calculator(calc)

    system_name = input_xyz_path.stem
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=out_dir / f"{system_name}.log",
        filemode="a",
        force=True,
    )

    atoms = read(input_xyz_path)
    output_fname = out_dir / f"{system_name}.traj"
    run_npt(atoms, calc, output_fname)
