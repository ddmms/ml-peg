"""Run calculations for HPHT_CH4_H2O tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

from ase import units
from ase.io import read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import LBFGS
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


def get_starting_frames() -> list[Path]:
    """Return the starting structures for the benchmark."""


STARTING_FRAMES = get_starting_frames()


@pytest.mark.parametrize("mlip", MODELS.items(), ids=MODELS.keys())
def test_md(
    mlip: tuple[str, Any],
    structure,
) -> None:
    """
    Run a high pressure high temperature molecular dynamics simulations.

    Run a molecular dynamics simulation at 3000 K for each starting structure
    of the CH4/H2O mixture after a quick geometry optimization.

    Parameters
    ----------
    mlip
        Tuple containing the model name and the corresponding model object.
    structure
        Name of the structure to simulate, corresponding to the input structure
        filename stem.

    Notes
    -----
    Generate a trajectory as a .extxyz file and the associated restart file
    in the output folder, together with two log files, one for the
    molecular dynamics (.log) and one for the optimization (.opt).
    """
    model_name, model = mlip

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    calc = model.get_calculator(precision="low")
    calc = model.add_d3_calculator(calc)

    starting_frames_dir = (
        download_s3_data(
            key="inputs/molecular_reactions/HPHT_CH4_H2O/HPHT_CH4_H2O.zip",
            filename="HPHT_CH4_H2O.zip",
        )
        / "HPHT_CH4_H2O"
    )
    starting_frames_files = sorted(starting_frames_dir.glob("*.extxyz"))

    if structure is None:
        structures = starting_frames_files
    else:
        structures = [path for path in starting_frames_files if path.stem == structure]

    for starting_frame in structures:
        structure_name = starting_frame.stem
        print(f"Actual structure : {structure_name}")
        traj_path = write_dir / f"{structure_name}.extxyz"
        restart_path = write_dir / f"{structure_name}.restart.extxyz"
        log_path = write_dir / f"{structure_name}.log"
        if log_path.exists():
            with open(log_path) as f:
                nsteps_done = sum(1 for _ in f) - 1
            nsteps_done = max(nsteps_done, 0)
        else:
            nsteps_done = 0
        if restart_path.exists():
            atoms = read(restart_path, index=-1, format="extxyz")
            atoms.calc = calc
        else:
            atoms = read(starting_frame, index=-1, format="extxyz")
            atoms.calc = calc
            # OPTIMIZATION
            try:
                opt = LBFGS(atoms, logfile=write_dir / f"{structure_name}.opt")
                opt.run(fmax=0.2)
            except Exception as exc:
                warnings.warn(
                    f"Geometry optimization failed for {structure_name}: {exc}",
                    stacklevel=2,
                )
                continue
            # VELOCITIES INITIALISATION
            rng = np.random.default_rng(seed=13)
            MaxwellBoltzmannDistribution(atoms, temperature_K=3000, rng=rng)
            Stationary(atoms)
            ZeroRotation(atoms)

        # OUTPUT SETUP

        def write_frame(
            traj_path,
            restart_path,
            atoms,
        ):
            try:
                write(traj_path, atoms, format="extxyz", append=True)
                write(restart_path, atoms, format="extxyz", append=False)
            except Exception as exc:
                warnings.warn(f"Writing failed: {exc}", stacklevel=2)

        # MOLECULAR DYNAMICS
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=0.5 * units.fs,
            temperature_K=3000,
            tdamp=100 * units.fs,
            tchain=3,
            tloop=2,
            logfile=str(log_path),
        )

        dyn.attach(
            write_frame,
            1,
            traj_path,
            restart_path,
            atoms,
        )
        remaining_steps = 100000 - nsteps_done
        if remaining_steps <= 0:
            print(f"{structure_name}: trajectory already completed.")
            continue
        print(f"{structure_name}: {remaining_steps} steps remaining, continuing MD.")
        try:
            dyn.run(remaining_steps)
            print("MD finished", flush=True)
        except Exception as exc:
            warnings.warn(f"MD failed for {structure_name}: {exc}", stacklevel=2)
            continue
