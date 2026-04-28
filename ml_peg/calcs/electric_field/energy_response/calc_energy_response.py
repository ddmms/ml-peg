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
    DATA_NOT_DOWNLOADED = True
    
    model_name, model = mlip
    model.default_dtype = "float64"
    clean_calc = model.get_calculator()


    # Organic molecules data sets for electric field response.
    datasets = [
        "ALKANES",
        "CUMULENES",
    ]

    # Download data
    data_path = download_github_data(
        filename="LINEAR_CARBON_wB97M-V.zip",
        github_uri="https://github.com/viktorsvahn/teoroo_ML-PEG/raw/refs/heads/main/data/source",
    )

    for dataset in datasets:
        #mols = read(data_path/f'{dataset}.xyz',':')
        mols_out = []

        if DATA_NOT_DOWNLOADED:
            DATA_PATH.mkdir(parents=True, exist_ok=True)
            mols = read(data_path/ 'data' /f'{dataset}.xyz',':')
            write(DATA_PATH/f'{dataset}.xyz', mols)
            DATA_NOT_DOWNLOADED = False
        else:
            mols = read(DATA_PATH/f'{dataset}.xyz',':')


        for mol in mols:
            mol.calc = copy(clean_calc)
            _ = mol.get_potential_energy()
            if 'external_field' in mol.info:
                mols_out.append(mol)

        # Write output structures
        if len(mols_out) > 0:
            write_dir = OUT_PATH/model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir/f'{dataset}.xyz', mols_out)