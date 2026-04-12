"""Run calculations for mSE21 benchmark."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
import numpy as np
import pandas as pd
import pytest

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_mse21(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the mSE21 dataset.

    Surface energy calculations for 21 metals. Reference values are taken from
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
        filename="mSE21.zip",
        github_uri="https://github.com/benshi97/ml-peg-transition-metal-data/raw/main",
        force=True,
    )

    cmr_structs_list = read(data_dir / "CMRads_structures.extxyz", index=":")
    cmrads_reactions_dict = pd.read_csv(
        data_dir / "reactions_dict.csv", index_col=0
    ).to_dict(orient="index")

    # Add calc to the systems in cmr_structs_list
    for atoms in cmr_structs_list:
        atoms.calc = copy(calc)

    structs_energy_dict = {
        struct.info["sys_formula"]: struct for struct in cmr_structs_list
    }

    # Compute adsorption energies
    for idx, reaction_id in enumerate(cmrads_reactions_dict.keys()):
        systems_list = cmrads_reactions_dict[reaction_id]["Systems"].split(",")
        stoichiometry_list = np.array(
            cmrads_reactions_dict[reaction_id]["Stoichiometry"].split(",")
        ).astype(float)
        systems_energies = np.array(
            [structs_energy_dict[sys].get_potential_energy() for sys in systems_list]
        )
        reaction_energy = np.sum(systems_energies * stoichiometry_list)
        print(
            "Reaction ID: "
            f"{reaction_id}, Predicted Reaction Energy: "
            f"{reaction_energy:.4f} eV, Reference Reaction Energy: "
            f"{cmrads_reactions_dict[reaction_id]['PBE_ads']:.4f} eV"
        )
        mol_surface = structs_energy_dict[systems_list[0]].copy()
        mol_surface.info["pred_adsorption_energy"] = reaction_energy
        mol_surface.info["PBE_adsorption_energy"] = cmrads_reactions_dict[reaction_id][
            "PBE_ads"
        ]
        mol_surface.info["RPA_adsorption_energy"] = cmrads_reactions_dict[reaction_id][
            "RPA_ads"
        ]
        mol_surface.calc = None
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        append_mode = True if idx > 0 else False
        write(
            write_dir / "mol_surface_structs.extxyz",
            mol_surface,
            append=append_mode,
        )
