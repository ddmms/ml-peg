"""Run calculation for aqueous FeCl oxidation states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read
from janus_core.calculations.md import NPT
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

IRON_SALTS = ["Fe2Cl", "Fe3Cl"]


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_iron_oxidation_state_md(mlip: tuple[str, Any]) -> None:
    """
    Run MLMD for aqueous FeCl oxidation states tests.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for salt in IRON_SALTS:
        struct_path = DATA_PATH / f"{salt}_start.xyz"
        struct = read(struct_path, "0")
        struct.calc = calc

        npt = NPT(
            struct=struct,
            steps=200,
            timestep=0.5,
            stats_every=100,
            traj_every=200,
            traj_append=True,
            thermostat_time=50,
            bulk_modulus=10,
            barostat_time=None,
            # pressure=pressure,
            file_prefix=OUT_PATH / f"{salt}_{model_name}",
            # restart=True,
            # restart_auto=True,
            # post_process_kwargs={"rdf_compute": True, "rdf_rmax": 6, "rdf_bins": 120},
        )
        npt.run()
