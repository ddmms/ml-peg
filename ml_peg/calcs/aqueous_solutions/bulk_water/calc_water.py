"""Run calculations for bulk benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read
from janus_core.calculations.md import NVT
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
EQUIL_STEPS = 25000  # equilibration steps (not recorded)
RUN_STEPS = 300000  # production steps (recorded)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bulk_water(mlip: tuple[str, Any]) -> None:
    """
    Run Bulk Water test.

    Runs Langevin MD (via janus-core) on bulk water, writing the trajectory
    (positions and momenta) to ``md-traj.extxyz`` for analysis.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    # Setup calculator with d3 correction
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    # Get bulk water benchmark data
    data_dir = (
        download_s3_data(
            filename="bulk_water.zip",
            key="inputs/aqueous-solutions/bulk_water/bulk_water.zip",
        )
        / "bulk_water"
    )

    # Load initial structure
    start_config = read(data_dir / "init.xyz")
    start_config.pbc = [True, True, True]
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
        stats_every=100,
        traj_every=1,
        traj_start=EQUIL_STEPS,
        file_prefix=write_dir / "md",
        write_kwargs={"columns": ["symbols", "positions", "momenta", "masses"]},
    )

    try:
        md.run()
    except Exception as exc:
        warn(f"Error during MD: {exc}", stacklevel=2)
