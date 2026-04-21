"""Run calculations for water slab dipole tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read
from janus_core.calculations.md import NVT

# from ase.md.npt import NPT
# from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


@pytest.mark.parametrize("mlip", MODELS.items())
def test_water_dipole(mlip: tuple[str, Any]) -> None:
    """
    Run water dipole test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test (for models where applicable)
    calc = model.add_d3_calculator(calc)

    out_name = "slab"

    # One could consider increasing the length of simulation,
    # so far I only reduced the printing intervals to get more
    # samples.
    md_t = 200000  # number of timesteps, here dt = 1fs
    md_dt = 100  # intervals for printing energy, T, etc
    th_dt = 100  # intervals for printing structures
    temp = 300  # Kelvin

    ttime = 100 * units.fs  # timescale themostat

    data_dir = (
        download_s3_data(
            key="inputs/physicality/water_slab_dipoles/water_slab_dipoles.zip",
            filename="water_slab_dipoles.zip",
        )
        / "water_slab_dipoles"
    )

    start_config = read(data_dir / "init_38A_slab.xyz", "-1")
    start_config.info["charge"] = 0
    start_config.info["spin"] = 1

    start_config.calc = calc

    # Write output structures
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    print("Writing to ", write_dir)

    md = NVT(
        struct=start_config,
        temp=temp,
        steps=md_t,
        stats_every=md_dt,
        traj_every=th_dt,
        friction=1 / ttime,
        file_prefix=write_dir / out_name,
    )

    print("Starting MD")

    md.run()

    print("Finished MD")
