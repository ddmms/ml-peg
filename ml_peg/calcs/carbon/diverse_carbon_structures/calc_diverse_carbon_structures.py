"""
Compute single-point energies and forces for the ACE carbon training dataset.

Reference: Bochkarev et al., J. Chem. Theory Comput. 2022.
17 293 carbon structures covering sp2, sp3, amorphous/liquid, general bulk,
and general cluster environments. Reference energies and forces from DFT/PBE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_diverse_carbon_structures(mlip: tuple[str, Any]) -> None:
    """
    Run single-point energy and force calculations for all ACE carbon structures.

    Parameters
    ----------
    mlip
        Model name and model object to use as calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_dir = (
        download_s3_data(
            key="inputs/carbon/diverse_carbon_structures/diverse_carbon_structures.zip",
            filename="diverse_carbon_structures.zip",
        )
        / "diverse_carbon_structures"
    )
    atoms_list = read(data_dir / "ACE_PBE_dataset.xyz", ":")

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for atoms in tqdm(atoms_list, desc=model_name):
        # Reference energy and forces stored by ASE's SinglePointCalculator on read
        ref_energy = atoms.get_potential_energy()
        ref_forces = atoms.get_forces()

        atoms.calc = calc
        pred_energy = atoms.get_potential_energy()
        pred_forces = atoms.get_forces()

        atoms.info["ref_energy"] = float(ref_energy)
        atoms.info["pred_energy"] = float(pred_energy)
        atoms.arrays["ref_forces"] = ref_forces
        atoms.arrays["pred_forces"] = pred_forces
        atoms.calc = None

        results.append(atoms)

    write(out_dir / "results.xyz", results)
