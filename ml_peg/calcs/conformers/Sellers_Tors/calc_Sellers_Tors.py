"""
Calculate the Sellers torsion scan benchmark for drug-like fragments.

A Comparison of Quantum and Molecular Mechanical Methods to Estimate Strain Energy in
Druglike Fragments
Benjamin D. Sellers, Natalie C. James, Alberto Gobbi
Journal of Chemical Information and Modeling 2017 57 (6), 1265-1275
DOI: 10.1021/acs.jcim.6b00614
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_sellers_tors(mlip: tuple[str, Any]) -> None:
    """
    Run Sellers torsion scan benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    data_path = (
        download_s3_data(
            filename="Sellers_Tors.zip",
            key="inputs/conformers/Sellers_Tors/Sellers_Tors.zip",
        )
        / "Sellers_Tors"
    )

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    for scan_file in tqdm(sorted(data_path.glob("*.extxyz"))):
        # Each file contains the full torsion scan of one fragment
        scan = read(scan_file, index=":")
        for atoms in scan:
            atoms.info["charge"] = int(atoms.info["charge"])
            atoms.info["spin"] = int(atoms.info["spin"])
            atoms.calc = calc
            try:
                atoms.info["model_energy"] = atoms.get_potential_energy()
            except Exception as exc:
                warn(
                    f"Error calculating energy for {model_name}, fragment "
                    f"{atoms.info['mol_id']}, dihedral {atoms.info['dihedral']}: {exc}",
                    stacklevel=2,
                )
                atoms.info["model_energy"] = np.nan
            atoms.calc = None

        write(write_dir / scan_file.name, scan)
