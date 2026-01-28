"""Run calculation for aqueous FeCl oxidation states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.md import NPT
from janus_core.calculations.single_point import SinglePoint
import numpy as np
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

IRON_SALTS = ["Fe2Cl, Fe3Cl"]

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


    for salt in IRON_SALTS:
        struct_path = DATA_PATH / f"{salt}_start.xyz"
        struct = read(struct_path, "0")

        npt = NPT(
            struct=struct,
            steps=40000,
            timestep=0.5,
            stats_every=50,
            traj_every=50,
            traj_append=True,
            thermostat_time=50,
            bulk_modulus=10,
            barostat_time=1500,
            pressure=pressure,
            file_prefix=OUT_PATH / f"{salt}_{mlip}",
            restart=True,
            restart_auto=False,
        )
        npt.run()


@pytest.mark.parametrize("mlip", MODELS.items())
def test_static_md(mlip: tuple[str, Any]) -> None:
    """
    Evaluate an existing AIMD trajectory for the static high-pressure hydrogen tests.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    struct_path = DATA_PATH / "Hpres.xyz"

    model_name, model = mlip
    calc = model.get_calculator()

    structs = read(struct_path, index="::50")
    structs.calc = calc
    for struct in structs:
        struct.info["density"] = (
            np.sum(struct.get_masses()) / struct.get_volume() * DENS_FACT
        )

    SinglePoint(
        struct=structs,
        write_results=True,
        file_prefix=OUT_PATH / f"H-static-{mlip}",
    ).run()
