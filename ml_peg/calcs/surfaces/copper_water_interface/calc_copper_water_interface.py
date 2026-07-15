"""Run calculations for copper water interface benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.constraints import FixAtoms
from ase.io import read
from janus_core.calculations.md import NVT
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"

# MD settings
TEMPERATURE = 330  # Kelvin
FRICTION = 0.05  # Langevin friction coefficient
TIMESTEP = 1  # fs
EQUIL_STEPS = 50  # equilibration steps (not recorded)
RUN_STEPS = 3000  # production steps (recorded)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_copper_water_interface(mlip: tuple[str, Any]) -> None:
    """
    Run Copper-Water Interface test.

    Runs Langevin MD (via janus-core) on the Cu/water interface with deuterated
    hydrogens and the bottom slab layers fixed. The trajectory (positions and
    momenta) is written to ``md-traj.extxyz`` for analysis.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    # Setup calculator with d3 correction
    model_name, model = mlip
    calc = model.get_calculator()
    # calc = model.add_d3_calculator(calc)

    # Get copper water benchmark data
    data_dir = (
        download_s3_data(
            filename="copper_water_interface.zip",
            key="inputs/surfaces/copper_water_interface/copper_water_interface.zip",
        )
        / "copper_water_interface"
    )

    # Load initial structure
    start_config = read(data_dir / "init.xyz")

    # Set deuterium mass for H atoms
    masses = start_config.get_masses()
    masses[np.array(start_config.get_chemical_symbols()) == "H"] = 2.0
    start_config.set_masses(masses)

    # Fix bottom layers of slab
    if "fix_indices" not in start_config.info:
        raise ValueError("Structure missing 'fix_indices' in info dict")
    start_config.set_constraint(FixAtoms(start_config.info["fix_indices"]))

    start_config.pbc = [True, True, False]
    start_config.info["charge"] = 0
    start_config.info["spin"] = 1
    start_config.calc = calc

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Run MD. equil_steps run first (not recorded); the trajectory is written
    # every step from traj_start onwards so velocities are available for VACF/VDOS.
    md = NVT(
        struct=start_config,
        temp=TEMPERATURE,
        steps=EQUIL_STEPS + RUN_STEPS,
        equil_steps=EQUIL_STEPS,
        timestep=TIMESTEP,
        friction=FRICTION,
        stats_every=TIMESTEP,
        traj_every=TIMESTEP,
        traj_start=EQUIL_STEPS,
        file_prefix=write_dir / "md",
        write_kwargs={"columns": ["symbols", "positions", "momenta", "masses"]},
    )

    # try:
    md.run()
    # except Exception as exc:
    #    warn(f"Error during MD: {exc}", stacklevel=2)
