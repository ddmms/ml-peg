"""Run calculations for mSol23Ec benchmark."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import bulk
from ase.io import write
import pandas as pd
import pytest

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_msol23ec(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the mSol23Ec dataset.

    Surface energy calculations for 23 metals. Reference values are extracted
    from the 23 metals in the Sol54Ec dataset from
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.235162.
    Cite: Phys. Rev. B 93, 235162.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip

    # Use double precision for this test
    model.default_dtype = "float64"
    calc = model.get_calculator()

    data_dir = download_github_data(
        filename="mSol23Ec.zip",
        github_uri="https://github.com/benshi97/ml-peg-transition-metal-data/raw/main",
        force=True,
    )

    msol23ec_reactions_dict = pd.read_csv(
        data_dir / "mSol23Ec.csv", index_col=0
    ).to_dict(orient="index")

    for idx, system in enumerate(msol23ec_reactions_dict.keys()):
        metal = system.split("_")[0]
        structure = system.split("_")[1]
        a_lat = msol23ec_reactions_dict[system]["latpar"]
        bulk_struct = bulk(metal, structure, a=a_lat)
        bulk_struct.calc = copy(calc)
        atom_struct = Atoms(metal, positions=[[0, 0, 0]])
        atom_struct.calc = copy(calc)
        cohesive_energy = (
            bulk_struct.get_potential_energy() / len(bulk_struct)
        ) - atom_struct.get_potential_energy()
        bulk_struct.info["pred_cohesive_energy"] = -cohesive_energy
        bulk_struct.info["PBE_cohesive_energy"] = msol23ec_reactions_dict[system][
            "PBE_Ec"
        ]
        bulk_struct.info["Exp_cohesive_energy"] = msol23ec_reactions_dict[system][
            "Exp_Ec"
        ]
        bulk_struct.calc = None
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        append_mode = True if idx > 0 else False
        write(write_dir / "bulk_structs.extxyz", bulk_struct, append=append_mode)
