"""Run calculations for HF structure factor tests."""

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
TEMPERATURE_K = 296
PRESSURE_BAR = 1.2
DT_FS = 0.5  # femtoseconds
N_MIN_STEPS = 300  # maximum minimization steps
N_NPT_STEPS = 100000  # NPT production steps
OUT_FREQ = 100  # trajectory dump frequency

# Conversions
PRESSURE_AU = PRESSURE_BAR * units.bar
DT = DT_FS * units.fs
TDAMP = 100 * DT_FS * units.fs
PDAMP = 1000 * DT_FS * units.fs


def read_restart(traj_path: Path) -> tuple[Atoms | None, int]:
    """
    Read the last frame of a previous run, if there is one.

    Parameters
    ----------
    traj_path
        Path to the restart trajectory of the NPT run.

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
def test_hf_structure(mlip: tuple[str, Any]) -> None:
    """
    Run HF structure factor NPT simulation.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="low")

    calc = model.add_d3_calculator(calc)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Restart trajectory, written alongside NPT.xyz for post-processing
    restart_path = write_dir / "NPT.traj"
    traj_path = write_dir / "NPT.xyz"

    atoms, nsteps = read_restart(restart_path)
    restarting = atoms is not None

    if restarting:
        print(f"Resuming HF structure with model {model_name} from step {nsteps}")
    else:
        print(f"Simulating HF structure with model {model_name}")

        # Download dataset
        hf_structure_dir = (
            download_s3_data(
                key="inputs/superacids/HF_structure/HF_structure.zip",
                filename="HF_structure.zip",
            )
            / "HF_structure"
        )

        atoms = read(hf_structure_dir / "start.xyz")
        traj_path.unlink(missing_ok=True)

    atoms.calc = calc

    if not restarting:
        # Minimization
        opt = FIRE(atoms, logfile=str(write_dir / "opt.log"))
        try:
            opt.run(fmax=0.05, steps=N_MIN_STEPS)
        except Exception as exc:
            warn(f"Error minimising HF structure: {exc}", stacklevel=2)
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

    restart_file = Trajectory(str(restart_path), "a" if restarting else "w", atoms)

    last_written = nsteps if restarting else -1

    def write_frame(_dyn=dyn, _atoms=atoms, _path=traj_path) -> None:
        """
        Append the current frame to the restart and NPT trajectories.

        Parameters
        ----------
        _dyn : IsotropicMTKNPT
            The dynamics object.
        _atoms : Atoms
            The ASE atoms object.
        _path : Path
            Path to the NPT trajectory file.
        """
        nonlocal last_written

        step = _dyn.nsteps
        if step <= last_written:
            # Resuming: this frame was already written by the previous run.
            return

        restart_file.write()
        write(_path, _atoms, append=True)
        last_written = step

    write_frame()  # step 0
    dyn.attach(write_frame, interval=OUT_FREQ)

    # Run NPT
    if nsteps < N_NPT_STEPS:
        try:
            dyn.run(N_NPT_STEPS - nsteps)
        except Exception as exc:
            warn(f"Error running MD for HF structure: {exc}", stacklevel=2)

    restart_file.close()

    print(f"  HF structure done ({model_name})")
