"""Functionality for energy response tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import numpy as np
import pytest
import json

from ml_peg.calcs.utils.utils import download_s3_data       # PRODUCTION
from ml_peg.calcs.utils.utils import download_github_data   # TEST
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def run_energy_response(
    mlip: tuple[str, Any],
    datasets: list[str],
    out_path: Path,
) -> None:
    """
    Run energy response test for specific model and datasets.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    datasets
        Datasets to run benchmark for.
    out_path
        Directory to write out data to.
    exclude_elements
        Elements to exclude from calculations. Default is all elements.
    """
    model_name, model = mlip
    model.default_dtype = "float64"
    clean_calc = model.get_calculator()
 
    # Download data
    data_path = download_github_data(
        filename="ORCA_DATA_wB97M-V.zip",
        github_uri="https://github.com/viktorsvahn/teoroo_ML-PEG/raw/refs/heads/main/data/source",
    )

    for dataset in datasets:
        mols = read(data_path/f'{dataset}.xyz',':')
        mol_out = []

        for mol in mols:
            mol.calc = copy(clean_calc)
            _ = mol.get_potential_energy()
            mol_out.append(mol)

        # Write output structures
        write_dir = OUT_PATH/model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir/f'{dataset}.xyz', mol_out)   # This filename needs to be adjusted to whatever's in the zipfile above

