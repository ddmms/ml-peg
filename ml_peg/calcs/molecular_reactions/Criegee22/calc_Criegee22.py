"""
Calculate the Criegee22 dataset for reactions involving Crieegee intermediates.

C. D. Smith, A. Karton. J. Comput. Chem. 2020, 41, 328–339.
DOI: 10.1002/jcc.26106
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KJ_TO_EV = units.kJ / units.mol
EV_TO_KJ = 1 / KJ_TO_EV

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_criegee22(mlip: tuple[str, Any]) -> None:
    """
    Run Criegee22 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="Criegee22.zip",
            key="inputs/molecular_reactions/Criegee22/Criegee22.zip",
        )
        / "Criegee22"
    )

    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    with open(data_path / "reference.txt") as lines:
        for i, line in tqdm(enumerate(lines)):
            if i == 0:
                continue
            items = line.strip().split()
            label = items[0]
            bh_ref = float(items[8]) * KJ_TO_EV
            atoms_reac = read(data_path / "structures" / f"{label}-reac.xyz")
            atoms_reac.calc = calc
            atoms_reac.info["charge"] = 0
            atoms_reac.info["spin"] = 1
            atoms_reac.info["model_energy"] = atoms_reac.get_potential_energy()
            atoms_reac.info["ref_energy"] = 0

            atoms_ts = read(data_path / "structures" / f"{label}-TS.xyz")
            atoms_ts.calc = calc
            atoms_ts.info["charge"] = 0
            atoms_ts.info["spin"] = 1
            atoms_ts.info["model_energy"] = atoms_ts.get_potential_energy()
            atoms_ts.info["ref_energy"] = bh_ref

            write_dir = OUT_PATH / model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{label}_rct.xyz", atoms_reac)
            write(write_dir / f"{label}_ts.xyz", atoms_ts)
