"""
Compute the ACONFL dataset for molecular conformer relative energies.

Conformational Energy Benchmark for Longer n-Alkane Chains
Sebastian Ehlert, Stefan Grimme, and Andreas Hansen
The Journal of Physical Chemistry A 2022 126 (22), 3521-3535
DOI: 10.1021/acs.jpca.2c02439
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_aconfl_conformer_energies(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the ACONFL dataset.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="ACONFL.zip",
            key="inputs/conformers/ACONFL/ACONFL.zip",
        )
        / "ACONFL"
    )

    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    with open(data_path / ".res") as lines:
        for line in lines:
            if "$tmer" in line:
                items = line.strip().split()
                zero_atoms_label = items[1].replace("/$f", "")
                atoms_label = items[2].replace("/$f", "")
                ref_rel_energy = float(items[7]) * KCAL_TO_EV
                atoms = read(data_path / atoms_label / "struc.xyz")
                atoms.calc = calc
                atoms.info.update({"charge": 0, "spin": 1})
                zero_atoms = read(data_path / zero_atoms_label / "struc.xyz")
                zero_atoms.calc = calc
                zero_atoms.info.update({"charge": 0, "spin": 1})
                atoms.info["model_rel_energy"] = (
                    atoms.get_potential_energy() - zero_atoms.get_potential_energy()
                )
                atoms.info["ref_rel_energy"] = ref_rel_energy

                write_dir = OUT_PATH / model_name
                write_dir.mkdir(parents=True, exist_ok=True)

                write(write_dir / f"{atoms_label}.xyz", atoms)
