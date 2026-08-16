"""
Calculate thermodynamic properties of selected organic liquids.

Liquids include 146 chemicals from  https://doi.org/10.1021/ct200731v
"""

from __future__ import annotations

import logging
from pathlib import Path
from shutil import copy2
import time
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.io import Trajectory, read
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

try:
    from tqdm.auto import tqdm as tqdm
except ModuleNotFoundError:
    tqdm = None


MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"

AU_TO_G_L = 1e27 / units.mol
TIMESTEP = 1 * units.fs
ATM = 1.01325 * units.bar

DEFAULT_SELECTED_CAS = {
    "91-22-5",  # quinoline
    "56-81-5",  # glycerol
    "109-99-9",  # THF
    "67-64-1",  # acetone
    "108-88-3",  # toluene
    "111-87-5",  # 1-octanol
    "106-93-4",  # 1,2-dibromoethane
    "98-08-8",  # benzotrifluoride
    "367-11-3",  # difluorobenzene
}


def get_density_g_liter(atoms: Atoms):
    """
    Get the density of the system in g/L.

    Parameters
    ----------
    atoms
        ASE.Atoms object of the periodic system.

    Returns
    -------
    float
        Density in g/L.
    """
    mass = atoms.get_masses().sum()
    volume = atoms.get_volume()
    return AU_TO_G_L * mass / volume


def truncate_log_to_step(log_file, time_ps):
    """
    Truncate log after the given simulation time.

    Parameters
    ----------
    log_file
        The log file path.
    time_ps
        Last time to keep (in ps, inclusive).
    """
    if not Path(log_file).exists():
        return

    fields = ("t:", "Epot:", "Ekin:", "volume:", "density:")
    last_time = -1.0
    kept = []

    with open(log_file) as f:
        for line in f:
            if not all(field in line for field in fields):
                kept.append(line)
                continue

            t = float(line.split("t:")[1].split()[0])

            if t > time_ps or t <= last_time:
                continue

            kept.append(line)
            last_time = t

    with open(log_file, "w") as f:
        f.writelines(kept)


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
    epot = dyn.atoms.get_potential_energy()
    ekin = dyn.atoms.get_kinetic_energy()
    if not np.all(np.isfinite([epot, ekin])):
        raise ValueError("Energy diverged")
    density = get_density_g_liter(dyn.atoms)
    volume = dyn.atoms.get_volume()
    temperature = dyn.atoms.get_temperature()
    t = dyn.get_time() / (1000 * units.fs)
    logging.info(
        f"""t: {t:>8.3f} ps\
            Walltime: {current_time:>10.3f} s\
            T: {temperature:.5f} K\
            Epot: {epot:.5f} eV\
            Ekin: {ekin:.5f} eV\
            volume: {volume:.5f} A^3\
            density: {density:.5f} g/L\
        """
    )


def run_md(
    atoms,
    temp,
    calc,
    total_md_steps,
    traj_file,
    traj_interval,
    log_file,
    log_interval,
    phase,
):
    """
    Run NPT molecular dynamics using the isotropic MTK barostat.

    Parameters
    ----------
    atoms
        ASE Atoms of the system.
    temp
        Temperature.
    calc
        ASE Calculator.
    total_md_steps
        Total number of steps in the MD simulation.
    traj_file
        File path to save the trajectory to.
    traj_interval
        Trajectory dumping interval.
    log_file
        File path to save the log to.
    log_interval
        Log dumping interval.
    phase
        Phase to simulate ('gas' or 'liq'). Drives the NPT/NVT logics.
    """
    nsteps = 0
    if Path(traj_file).exists():
        try:
            traj = Trajectory(traj_file)
            if traj[0].info.get("md_step", 0) == 0:
                # otherwise the writer will keep appending the zeroth frame
                Path(traj_file).unlink()
                # same treatment for the log
                if Path(log_file).exists():
                    Path(log_file).unlink()
                nsteps = 0
            else:
                atoms = traj[-1]
                nsteps = atoms.info.get("md_step", 0)
                print("Found a trajectory that is", nsteps, "steps long")

        except Exception as exc:
            print(exc)
            nsteps = 0

    truncate_ps = nsteps * TIMESTEP / (1000 * units.fs)

    truncate_log_to_step(log_file, truncate_ps)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=log_file,
        filemode="a",
        force=True,
    )

    # Set default charge and spin
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)

    if phase == "liq":
        atoms.pbc = True
    elif phase == "gas":
        atoms.pbc = False
    else:
        raise ValueError("Wrong phase: " + phase + ", must be either 'liq' or 'gas'.")

    atoms.calc = calc

    if phase == "gas":
        try:  # not all models may support pbc=False
            atoms.get_potential_energy()
        except Exception as exc:
            atoms.set_cell([30.0, 30.0, 30.0])
            atoms.center()
            atoms.pbc = True
            atoms.calc = calc
            warn(
                f"Non-periodic gas-phase calculation failed ({exc}); "
                "falling back to a periodic vacuum box.",
                stacklevel=2,
            )
            atoms.get_potential_energy()

    if phase == "liq":
        dyn = IsotropicMTKNPT(
            atoms=atoms,
            timestep=TIMESTEP,
            temperature_K=temp,
            pressure_au=ATM,
            tdamp=50 * units.fs,
            pdamp=500 * units.fs,
        )
    else:
        dyn = NoseHooverChainNVT(
            atoms=atoms,
            timestep=TIMESTEP,
            temperature_K=temp,
            tdamp=50 * units.fs,
        )

    traj = Trajectory(traj_file, mode="a", atoms=atoms)

    start_step = nsteps

    def write_traj():
        """Define how to write the trajectory."""
        atoms.info["md_step"] = start_step + dyn.nsteps
        traj.write(atoms)

    dyn.attach(log_md, interval=log_interval, dyn=dyn, start_time=time.time())
    dyn.attach(write_traj, interval=traj_interval)

    remaining_steps = total_md_steps - nsteps
    if tqdm is not None:
        pbar = tqdm(total=remaining_steps, desc="MD")

        def update_pbar():
            pbar.update(1)
            pbar.refresh()

        dyn.attach(update_pbar, interval=1)

    try:
        dyn.run(steps=remaining_steps)
    except Exception as exc:
        warn(f"Error running MD: {exc}", stacklevel=2)
        dyn.atoms.info["energy"] = np.nan
    finally:
        if tqdm is not None:
            pbar.close()


@pytest.mark.framework("mace-off-24")
@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_thermodynamic_properties(
    mlip: tuple[str, Any],
    cas: str,
    total_md_steps: int,
    traj_interval: int,
    log_interval: int,
) -> None:
    """
    Run Liquid Densities benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    cas
        CAS number identifier of the system to run MD on.
    total_md_steps
        Total number of steps in the MD simulation.
    traj_interval
        Trajectory dumping interval.
    log_interval
        Log dumping interval.
    """
    # Download data
    data_path = (
        download_s3_data(
            filename="thermodynamic_properties.zip",
            key="inputs/molecular_dynamics/thermodynamic_properties/thermodynamic_properties.zip",
        )
        / "thermodynamic_properties"
    )

    model_name, model = mlip
    calc = model.get_calculator(precision="low")
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(exist_ok=True, parents=True)

    config_files = {}
    available_cas = {}
    system_name = {}
    input_xyz_path = {}

    temp = None
    # Check files and setup some labels
    for phase in ["gas", "liq"]:
        config_files[phase] = sorted(
            (data_path / "equilibrated_structures_xyz").glob("*-" + phase + ".xyz")
        )
        available_cas[phase] = [
            f.name.removesuffix("-" + phase + ".xyz") for f in config_files[phase]
        ]
        assert cas in available_cas[phase], (
            "CAS number not available in " + phase + " database:" + cas
        )
        input_xyz_path[phase] = config_files[phase][
            np.where(np.asarray(available_cas[phase]) == cas)[0][0]
        ]
        system_name[phase] = input_xyz_path[phase].stem
        if phase == "liq":  # temperature is stored only in the liquid phase configs
            temp = float(read(input_xyz_path[phase]).info["exp_temperature"])

        copy2(input_xyz_path[phase], out_dir / f"{system_name[phase]}.xyz")

    if type(temp) is not float:
        raise ValueError("The temperature has not been assigned.")

    # Run the simulations
    for phase in ["gas", "liq"]:
        atoms = read(input_xyz_path[phase])
        traj_file = out_dir / f"{system_name[phase]}.traj"
        log_file = out_dir / f"{system_name[phase]}.log"
        run_md(
            atoms,
            temp,
            calc,
            total_md_steps,
            traj_file,
            traj_interval,
            log_file,
            log_interval,
            phase,
        )
