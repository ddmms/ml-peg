"""
Calculate the energy response of linear organic molecules subject to an external
electric field.

The reference calculations were made using ORCA at the wB97M-V/def2-TZVPD level.
"""

from __future__ import annotations

from copy import copy
import gc
from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
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
    clean_calc = model.get_calculator(precision="high")

    data_path = download_github_data(
        filename="LINEAR_CARBON_wB97M-V.zip",
        github_uri="https://github.com/viktorsvahn/teoroo_ML-PEG/raw/refs/heads/main/data/source",
    )
    datasets = sorted(f.name for f in (data_path / "data").glob("*.xyz"))

    for dataset in datasets:
        mols = read(data_path / "data" / dataset, ":")

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        out_file = write_dir / dataset

        if out_file.exists():
            continue

        for i, mol in enumerate(mols):
            mol.info["REF_energy"] = mol.get_potential_energy()
            mol_calc = copy(clean_calc)
            mol.calc = mol_calc
            mol.info["charge"] = 0
            mol.info["spin"] = 1
            try:
                mol.get_potential_energy()
            except Exception as exc:
                warn(f"Error calculating energy: {exc}", stacklevel=2)
                mol.info["energy"] = np.nan
            write(out_file, mol, append=(i > 0))
            mol.calc = None
            del mol_calc

        del mols
        gc.collect()
