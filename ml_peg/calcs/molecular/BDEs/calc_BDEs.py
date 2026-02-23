"""
Calculate bond dissociation energies of drug-like molecules.

Journal of Chemical Theory and Computation 2024 20 (1), 164-177
DOI: 10.1021/acs.jctc.3c00710
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bond_dissociation_energy(mlip: tuple[str, Any]) -> None:
    """
    Calculate C-H bond dissociation energy of drug-like molecules.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test (for models where applicable)
    calc = model.add_d3_calculator(calc)

    bde_dir = (
        download_s3_data(
            key="inputs/molecular/BDEs/BDEs.zip",
            filename="BDEs.zip",
        )
        / "BDEs"
    )

    structures_filename = "cytochrome_p450_substrates.dft_opt.xyz"
    mols_rads = read(Path(bde_dir) / structures_filename, ":")

    ats_out = []
    for at in mols_rads:
        at.calc = copy(calc)
        at.info["pred_energy"] = at.get_potential_energy()
        at.arrays["pred_forces"] = at.get_forces()
        ats_out.append(at)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / structures_filename, ats_out)
