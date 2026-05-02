"""
Calculate the energy response of linear organic molecules subject to an external 
electric field.

The reference calculations were made using ORCA at the wB97M-V/def2-TZVPD level.
"""

from __future__ import annotations

from ase import units
from ase.io import read, write
import numpy as np
from copy import copy
from pathlib import Path
from typing import Any

import pytest

#from ml_peg.calcs.electric_field.energy_response.energy_response import run_energy_response
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models
from ml_peg.app import APP_ROOT

DATA_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"
MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"



@pytest.mark.parametrize("mlip", MODELS.items())
def test_energy_response(mlip: tuple[str, Any]) -> None:
    """
    Run energy response test for specific model and datasets.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """    
    model_name, model = mlip
    model.default_dtype = "float64"
    clean_calc = model.get_calculator()


    # Download data
    data_path = download_github_data(
        filename="LINEAR_CARBON_wB97M-V.zip",
        github_uri="https://github.com/viktorsvahn/teoroo_ML-PEG/raw/refs/heads/main/data/source",
    )
    datasets = [f.name for f in (data_path/'data').glob("*.xyz")]

    APP_MOLS_NOT_STORED = True
    for dataset in datasets:
        mols_out = []

        if (DATA_PATH/dataset).exists():
            mols = read(DATA_PATH/dataset,':')
        else:
            mols = read(data_path/'data'/dataset,':')
            write(DATA_PATH/dataset, mols)

        for mol in mols:
            mol.calc = copy(clean_calc)
            _ = mol.get_potential_energy()

            mol_name = str(mol.get_chemical_formula())
            mol_dir = APP_ROOT / f"data/electric_field/energy_response/{model_name}"
            mol_dir.mkdir(parents=True, exist_ok=True)
            write(
                mol_dir/f"{mol_name}.xyz",
                mol
            )

            if 'external_field' in mol.info:
                mols_out.append(mol)
        

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir/dataset, mols_out)