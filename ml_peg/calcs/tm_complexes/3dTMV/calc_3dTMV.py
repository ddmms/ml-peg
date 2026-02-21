"""
Compute the 3dTMV dataset for transition metal complex vertical ionization energies.

Journal of Chemical Theory and Computation 2023 19 (18), 6208-6225
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

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"

# Molecular data from Main Paper Table 1
MOLECULAR_DATA = {
    1: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    2: {"charge_ox": 1, "charge_in": 0, "mult_ox": 1, "mult_in": 2, "subset": "SR"},
    3: {"charge_ox": 1, "charge_in": 0, "mult_ox": 4, "mult_in": 3, "subset": "SR"},
    4: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    5: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    6: {"charge_ox": 2, "charge_in": 1, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    7: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    8: {"charge_ox": 2, "charge_in": 1, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    9: {"charge_ox": 2, "charge_in": 1, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    10: {"charge_ox": 2, "charge_in": 1, "mult_ox": 2, "mult_in": 1, "subset": "SR"},
    11: {"charge_ox": 2, "charge_in": 1, "mult_ox": 1, "mult_in": 2, "subset": "SR"},
    12: {"charge_ox": 2, "charge_in": 1, "mult_ox": 1, "mult_in": 2, "subset": "SR"},
    13: {"charge_ox": 1, "charge_in": 0, "mult_ox": 1, "mult_in": 2, "subset": "SR/MR"},
    14: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 3, "subset": "SR/MR"},
    15: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR/MR"},
    16: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR/MR"},
    17: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR/MR"},
    18: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR/MR"},
    19: {"charge_ox": 2, "charge_in": 1, "mult_ox": 2, "mult_in": 1, "subset": "SR/MR"},
    20: {"charge_ox": 1, "charge_in": 0, "mult_ox": 3, "mult_in": 2, "subset": "SR/MR"},
    21: {"charge_ox": 1, "charge_in": 0, "mult_ox": 3, "mult_in": 2, "subset": "SR/MR"},
    22: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "SR/MR"},
    23: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "MR"},
    24: {"charge_ox": 1, "charge_in": 0, "mult_ox": 3, "mult_in": 4, "subset": "MR"},
    25: {"charge_ox": 1, "charge_in": 0, "mult_ox": 3, "mult_in": 6, "subset": "MR"},
    26: {"charge_ox": 1, "charge_in": 0, "mult_ox": 2, "mult_in": 1, "subset": "MR"},
    27: {"charge_ox": 0, "charge_in": -1, "mult_ox": 2, "mult_in": 3, "subset": "MR"},
    28: {"charge_ox": 0, "charge_in": -1, "mult_ox": 1, "mult_in": 2, "subset": "MR"},
}

# ph-AFQMC reference IPs from SM Table S9 (kcal/mol)
REFERENCE_IES = {
    # SR subset (1-12)
    1: 188.4,
    2: 158.3,
    3: 119.6,
    4: 152.3,
    5: 142.2,
    6: 315.9,
    7: 191.1,
    8: 259.6,
    9: 276.2,
    10: 284.1,
    11: 198.5,
    12: 230.3,
    # SR/MR subset (13-22)
    13: 120.9,
    14: 148.1,
    15: 140.4,
    16: 164.1,
    17: 130.9,
    18: 136.3,
    19: 300.7,
    20: 186.4,
    21: 125.3,
    22: 161.2,
    # MR subset (23-28)
    23: 198.9,
    24: 166.0,
    25: 215.8,
    26: 192.9,
    27: 68.6,
    28: 43.6,
}


def get_atoms(data_path, complex_id: int):
    """
    Get the atoms object with charge and spin.

    Parameters
    ----------
    data_path
        Path to the data.
    complex_id
        Identifier of the complex, from 1 to 28.

    Returns
    -------
    Atoms
        Atoms object of the system.
    """
    return read(data_path / str(complex_id) / "struc.xyz")


@pytest.mark.parametrize("mlip", MODELS.items())
def test_3dtmv(mlip: tuple[str, Any]) -> None:
    """
    Run 3dTMV benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="3dTMV.zip",
            key="inputs/tm_complexes/3dTMV/3dTMV.zip",
        )
        / "3dTMV"
    )
    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for complex_id in tqdm(range(1, 29)):
        atoms = get_atoms(data_path, complex_id)
        model_ion_energy = 0
        # Get oxidized complex energy
        atoms.info["charge"] = MOLECULAR_DATA[complex_id]["charge_ox"]
        atoms.info["spin"] = MOLECULAR_DATA[complex_id]["mult_ox"]
        atoms.calc = calc
        model_ion_energy += atoms.get_potential_energy()
        # Get initial complex energy
        atoms.info["charge"] = MOLECULAR_DATA[complex_id]["charge_in"]
        atoms.info["spin"] = MOLECULAR_DATA[complex_id]["mult_in"]
        atoms.calc = calc
        model_ion_energy -= atoms.get_potential_energy()

        atoms.info.update(
            {
                "model_ionization_energy": model_ion_energy,
                "ref_ionization_energy": REFERENCE_IES[complex_id] * KCAL_TO_EV,
            }
        )
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{complex_id}.xyz", atoms)
