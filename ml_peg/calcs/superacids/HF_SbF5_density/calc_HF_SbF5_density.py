"""Run calculations for HF/SbF5 density tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.io import Trajectory, read, write
from ase.md.logger import MDLogger
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import FIRE
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Simulation parameters
TEMPERATURE_K = 288.65
PRESSURE_ATM = 1
DT_FS = 0.5  # DT in femtoseconds
N_MIN_STEPS = 300  # maximum minimization steps
N_NPT_STEPS = 200000  # NPT production steps
OUT_FREQ = 200

# Conversions
ATM_TO_GPA = 1.01325e-4  # 1 atm = 0.000101325 GPa
PRESSURE_AU = PRESSURE_ATM * ATM_TO_GPA * units.GPa
DT = 0.5 * units.fs
TDAMP = 100 * DT_FS * units.fs
PDAMP = 1000 * DT_FS * units.fs

# Systems
SYSTEMS = ["X_0", "X_10", "X_100"]


def read_restart(traj_path: Path) -> tuple[Atoms | None, int]:
    """
    Read the last frame of a previous run of this system, if there is one.

    Parameters
    ----------
    traj_path
        Path to the trajectory of the NPT run.

    Returns
    -------
    tuple[Atoms | None, int]
        Last frame written, and the step it was written at, or `(None, 0)` if
        there is no trajectory to restart from.
    """
    if not traj_path.exists():
        return None, 0

    try:
        traj = Trajectory(str(traj_path))
        atoms = traj[-1]
        nsteps = (len(traj) - 1) * OUT_FREQ
    except Exception as exc:
        warn(f"Ignoring unreadable trajectory {traj_path}: {exc}", stacklevel=2)
        return None, 0

    return atoms, nsteps


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items(), ids=lambda x: x[0])
@pytest.mark.parametrize("system", SYSTEMS)
def test_hf_sbf5_density(mlip: tuple[str, Any], system: str) -> None:
    """
    Run HF/SbF5 mixture density test.

    Interrupted runs are resumed from the last frame of the trajectory, so
    minimisation and the initial velocity distribution are only applied when
    starting from scratch.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    system
        System identifier (X_0, X_10, X_100).
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="low")

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    write_dir = OUT_PATH / model_name / system
    write_dir.mkdir(parents=True, exist_ok=True)
    traj_path = write_dir / f"{system}.traj"

    atoms, nsteps = read_restart(traj_path)
    restarting = atoms is not None

    if restarting:
        print(f"Resuming {system} with model {model_name} from step {nsteps}")
    else:
        print(f"Simulating {system} with model {model_name}")

        # Download dataset
        hf_sbf5_density_dir = (
            download_s3_data(
                key="inputs/superacids/HF_SbF5_density/HF_SbF5_density.zip",
                filename="HF_SbF5_density.zip",
            )
            / "HF_SbF5_density"
        )

        atoms = read(hf_sbf5_density_dir / system / "start.xyz")

    atoms.calc = calc

    if not restarting:
        # Minimization
        opt = FIRE(atoms, logfile=str(write_dir / "opt.log"))
        try:
            opt.run(fmax=0.05, steps=N_MIN_STEPS)
        except Exception as exc:
            warn(f"Error minimising {system}: {exc}", stacklevel=2)
        write(write_dir / "minimised.xyz", atoms)

        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
        Stationary(atoms)
        ZeroRotation(atoms)

    dyn = IsotropicMTKNPT(
        atoms=atoms,
        timestep=DT,
        temperature_K=TEMPERATURE_K,
        pressure_au=PRESSURE_AU,
        tdamp=TDAMP,
        pdamp=PDAMP,
    )

    dyn.nsteps = nsteps

    dyn.attach(
        MDLogger(
            dyn,
            atoms,
            str(write_dir / "md.log"),
            header=not restarting,
            mode="a" if restarting else "w",
        ),
        interval=OUT_FREQ,
    )

    traj_file = Trajectory(str(traj_path), "a" if restarting else "w", atoms)
    vol_file = open(write_dir / "volume.dat", "a" if restarting else "w")
    if not restarting:
        vol_file.write("# step  volume_A3\n")

    last_written = nsteps if restarting else -1

    def write_frame(_dyn=dyn, _atoms=atoms) -> None:
        """
        Append the current frame to the trajectory, and its volume to file.

        Parameters
        ----------
        _dyn : IsotropicMTKNPT
            The dynamics object.
        _atoms : Atoms
            The ASE atoms object.
        """
        nonlocal last_written

        step = _dyn.nsteps
        if step <= last_written:
            # Resuming: this frame was already written by the previous run.
            return

        traj_file.write()
        vol_file.write(f"{step}  {_atoms.get_volume():.6f}\n")
        vol_file.flush()
        last_written = step

    write_frame()  # step 0
    dyn.attach(write_frame, interval=OUT_FREQ)

    # Run NPT
    if nsteps < N_NPT_STEPS:
        try:
            dyn.run(N_NPT_STEPS - nsteps)
        except Exception as exc:
            warn(f"Error running MD for {system}: {exc}", stacklevel=2)

    vol_file.close()
    traj_file.close()

    # Save final structure
    atoms.info["system"] = system
    write(write_dir / f"{system}.xyz", atoms)

    print(f"  {system} done")
