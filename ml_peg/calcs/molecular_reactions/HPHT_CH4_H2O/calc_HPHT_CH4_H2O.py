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


@pytest.mark.parametrize("mlip", MODELS.items(), ids=MODELS.keys())
def test_md(mlip: tuple[str, Any]) -> None:
    """
    Run a high pressure high temperature molecular dynamics simulation
    for each starting structure of the CH4/H2O mixture after a quick geometry opitmization.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.

    Results
    -------
    Generate a trajectory as a .extxyz file in the output folder, together with a two log files,
    one for the molecular dynamics (.log) and one for the optimization (.opt).
    """
    model_name, model = mlip

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    calc = model.get_calculator(precision="low")
    calc = model.add_d3_calculator(calc)

    starting_frames_dir = (
        download_s3_data(
            key="inputs/molecular_reactions/HPHT_CH4_H2O/HPHT_CH4_H2O.zip",
            filename="HPHT_CH4_H2O_data.zip",
        )
        / "HPHT_CH4_H2O_data"
    )
    starting_frames_files = sorted(starting_frames_dir.glob("*.extxyz"))
    print(f"starting_frames_dir:{starting_frames_dir}")
    print(f"starting_frames:{starting_frames_files}")
    for starting_frame in starting_frames_files:
        structure_name = starting_frame.stem
        print(f"Actual structure : {structure_name}")
        atoms = read(starting_frame, index=-1, format="extxyz")
        print(f"atoms : {atoms}")
        atoms.calc = calc
        # OPTIMIZATION
        try:
            opt = LBFGS(atoms, logfile=write_dir / f"{structure_name}.opt")
            opt.run(fmax=0.2)
        except Exception as exc:
            warnings.warn(
                f"Geomoetry optimization failed for {structure_name}: {exc}",
                stacklevel=2,
            )
            continue
        # VELOCITIES INITIALISATION
        rng = np.random.default_rng(seed=13)
        MaxwellBoltzmannDistribution(atoms, temperature_K=3000, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)

        # OUTPUT SETUP

        traj_path = write_dir / f"{structure_name}.extxyz"
        log_path = write_dir / f"{structure_name}.log"

        def write_frame():
            try:
                write(traj_path, atoms, format="extxyz", append=True)
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

        dyn.attach(write_frame, interval=1)
        try:
            dyn.run(100000)
        except Exception as exc:
            warnings.warn(
                f"Molecular Dynamics failed for {structure_name}: {exc}", stacklevel=2
            )
            continue
