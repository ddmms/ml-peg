"""Calculate VACFs for metal self-diffusion."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import time
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import Trajectory, read
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Run parameters, timestep and pressure use ASE units.
TIMESTEP = 2.0 * units.fs

MELT_SCALE = 1.5

NPT_MELT = 10000
NPT_EQ = 25000
NPT_PROD = 100000

VACF_SAMPLES = 100
VACF_STEPS = 2500

THERMOSTAT_STEPS = 10
BAROSTAT_STEPS = 100

PRESSURE = 0.0 * units.bar

AU_TO_G_CM3 = 1e24 / units.mol

LOG_INTERVAL = NPT_PROD // VACF_SAMPLES


def init_velocity_recorder(atoms: Atoms) -> tuple[np.ndarray, Callable[[], None]]:
    """
    Allocate a velocity trajectory and its every-step recording callback.

    Parameters
    ----------
    atoms
        Atoms evolved by the NVE dynamics.

    Returns
    -------
    numpy.ndarray
        Velocities in Angstrom/fs, shaped (VACF_STEPS + 1, n_atoms, 3).
        Unrecorded samples remain NaN if the simulation stops early.
    Callable
        Callback recording the current velocities, starting at time zero.
    """
    steps = 0
    velocities = np.full((VACF_STEPS + 1, len(atoms), 3), np.nan)

    def record_velocities() -> None:
        """Record velocities, converting ASE time units to femtoseconds."""
        nonlocal steps
        velocities[steps] = atoms.get_velocities() * units.fs
        steps += 1

    return velocities, record_velocities


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


def run_npt(atoms: Atoms, calc: Calculator, output_fname: str | Path) -> None:
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
    if Path(output_fname).exists():
        try:
            with Trajectory(output_fname) as traj:
                atoms = traj[-1]
                nsteps = (len(traj) - 1) * LOG_INTERVAL
        except Exception as e:
            print(e)
            nsteps = 0
    else:
        nsteps = 0

    atoms.calc = calc
    temperature = atoms.info["exp_temperature"]
    if not atoms.has("momenta"):
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # Melt the crystal, then equilibrate and sample at the target temperature.
    # Recreate MTK to change its thermostat temperature between stages.
    for end_step, stage_temperature in (
        (NPT_MELT, MELT_SCALE * temperature),
        (NPT_MELT + NPT_EQ + NPT_PROD, temperature),
    ):
        if nsteps >= end_step:
            continue
        dyn = IsotropicMTKNPT(
            atoms=atoms,
            timestep=TIMESTEP,
            temperature_K=stage_temperature,
            pressure_au=PRESSURE,
            tdamp=THERMOSTAT_STEPS * TIMESTEP,
            pdamp=BAROSTAT_STEPS * TIMESTEP,
            trajectory=output_fname,
            loginterval=LOG_INTERVAL,
            append_trajectory=True,
        )
        dyn.nsteps = nsteps
        dyn.attach(log_md, interval=LOG_INTERVAL, dyn=dyn, start_time=time.time())
        try:
            dyn.run(steps=end_step - nsteps)
        except Exception as exc:
            warn(f"Error running MD: {exc}", stacklevel=2)
            dyn.atoms.info["energy"] = np.nan
            return
        nsteps = dyn.nsteps


def run_nve(atoms: Atoms, calc: Calculator) -> np.ndarray:
    """
    Run a fresh fixed-cell NVE trajectory with Maxwell-Boltzmann momenta.

    Parameters
    ----------
    atoms
        Sampled structure with target temperature in ``info['exp_temperature']``.
        The input is not modified; inherited momenta are replaced on a copy.
    calc
        ASE calculator.

    Returns
    -------
    numpy.ndarray
        Velocities in Angstrom/fs with shape (n_samples, n_atoms, 3).
        Record ``VACF_STEPS + 1`` samples, including the initial state at t=0.
    """
    state = atoms.copy()
    state.calc = calc
    MaxwellBoltzmannDistribution(state, temperature_K=state.info["exp_temperature"])
    velocities, record_velocities = init_velocity_recorder(state)
    dyn = VelocityVerlet(state, timestep=TIMESTEP)

    dyn.attach(record_velocities, interval=1)
    dyn.attach(log_md, interval=LOG_INTERVAL, dyn=dyn, start_time=time.time())

    try:
        dyn.run(steps=VACF_STEPS)
    except Exception as exc:
        warn(f"Error running MD: {exc}", stacklevel=2)

    return velocities


def compute_vacf(velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the unnormalized, single-origin velocity autocorrelation function.

    Parameters
    ----------
    velocities
        Velocities in Angstrom/fs, shaped (n_samples, n_atoms, 3), recorded
        every integration step.

    Returns
    -------
    numpy.ndarray
        Lag times in fs, starting at zero.
    numpy.ndarray
        Mean over atoms of v(0) dot v(t), in Angstrom squared/fs squared.

    """
    velocities = np.asarray(velocities, dtype=float)
    times = np.arange(len(velocities)) * TIMESTEP / units.fs
    vacf = np.einsum("tij,ij->t", velocities, velocities[0]) / velocities.shape[1]
    return np.vstack((times, vacf))


@pytest.mark.multi_day
@pytest.mark.parametrize("mlip", MODELS.items())
def test_metal_self_diffusion(mlip: tuple[str, Any], system) -> None:
    """
    Run Metal Self Diffusion benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    system
        Elements of the file in the metal input archive. Each crystal supplies paired ``temperatures`` and ``diffusion_constants`` metadata.
    """
    model_name, model = mlip

    calc = model.get_calculator(precision="low")
    
    data_path = (
        download_s3_data(
            filename="metal_self_diffusion.zip",
            key="inputs/molecular_dynamics/metal_self_diffusion/metal_self_diffusion.zip",
        )
        / "metal_self_diffusion"
    )
    input_path = sorted(data_path.glob("*.xyz"))[system]
    crystal = read(input_path)


    for temperature in crystal.info["temperatures"]:
        
        atoms = crystal.copy()
        
        atoms.info["exp_temperature"] = temperature
        
        out_dir = OUT_PATH / model_name / input_path.stem / f"{temperature:g}K"
        out_dir.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            filename=out_dir / "md.log",
            filemode="a",
            force=True,
        )

        output_fname = out_dir / "npt.traj"

        run_npt(atoms, calc, output_fname)

        first_frame = (NPT_MELT + NPT_EQ) // LOG_INTERVAL + 1

        with Trajectory(output_fname) as traj:
            for idx, frame_idx in enumerate(range(first_frame, len(traj))):

                vacf_path = out_dir / f"vacf_{idx}.npy"

                if vacf_path.exists():
                    continue

                velocities = run_nve(traj[frame_idx], calc)
                vacf = compute_vacf(velocities)

                np.save(vacf_path, vacf)
