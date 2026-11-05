"""Run calculations for ice benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.calcs.utils.water_md import run_water_md
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"

# MD settings
TEMPERATURE = 250  # Kelvin
FRICTION = 0.05  # Langevin friction coefficient
TIMESTEP = 0.5  # fs, MD integration step
# Frames are written every TRAJ_EVERY steps, so the stored spacing is
# TIMESTEP * TRAJ_EVERY fs -- this must match the reference's spacing.
TRAJ_EVERY = 4
EQUIL_STEPS = 50000  # equilibration steps (not recorded)
RUN_STEPS = 600000  # production steps (recorded)


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_ice(mlip: tuple[str, Any]) -> None:
    """
    Run Ice test.

    Runs Langevin MD (via janus-core) on ice, writing the trajectory
    (positions and momenta) to ``md-traj.extxyz`` for analysis.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    # Setup calculator with d3 correction
    model_name, model = mlip
    calc = model.get_calculator(precision="low")
    calc = model.add_d3_calculator(calc)

    # Get ice benchmark data
    data_dir = (
        download_s3_data(
            filename="ice.zip",
            key="inputs/aqueous_solutions/ice/ice.zip",
        )
        / "ice"
    )

    # Load initial structure
    start_config = read(data_dir / "init.xyz")
    start_config.pbc = [True, True, True]
    start_config.info["charge"] = 0
    start_config.info["spin"] = 1
    start_config.calc = calc

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    run_water_md(
        start_config,
        write_dir,
        equil_steps=EQUIL_STEPS,
        run_steps=RUN_STEPS,
        timestep=TIMESTEP,
        temperature=TEMPERATURE,
        friction=FRICTION,
        traj_every=TRAJ_EVERY,
    )
