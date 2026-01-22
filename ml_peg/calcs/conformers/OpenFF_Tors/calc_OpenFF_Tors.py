"""
Calculate the OpenFF-Tors benchmark dataset for torsional angles.

The Journal of Physical Chemistry B 2024 128 (32), 7888-7902.
DOI: 10.1021/acs.jpcb.4c03167.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import write
import numpy as np
import pytest
from rdkit import Chem
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_openff_tors(mlip: tuple[str, Any]) -> None:
    """
    Run OpenFF-Tors benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="OpenFF-Tors.zip",
            key="inputs/conformers/OpenFF-Tors/OpenFF-Tors.zip",
        )
        / "OpenFF-Tors"
    )
    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)
    with open(data_path / "MP2_heavy-aug-cc-pVTZ_torsiondrive_data.json") as file:
        data = json.load(file)

    for molecule_id, conf in tqdm(data.items()):
        charge = int(conf["metadata"]["mol_charge"])
        spin = int(conf["metadata"]["mol_multiplicity"])
        smiles = conf["metadata"]["mapped_smiles"]
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smiles, params)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_map = {
            atom.GetIntProp("molAtomMapNumber"): idx
            for idx, atom in enumerate(mol.GetAtoms())
            if atom.HasProp("molAtomMapNumber")
        }
        remapped_symbols = [symbols[atom_map[i]] for i in range(1, len(symbols) + 1)]

        for i, (ref_energy, positions) in enumerate(
            zip(conf["final_energies"], conf["final_geometries"], strict=True)
        ):
            label = f"{molecule_id}_{i}"
            atoms = Atoms(
                symbols=remapped_symbols, positions=np.array(positions) * units.Bohr
            )
            atoms.info["charge"] = charge
            atoms.info["spin"] = spin
            atoms.calc = calc

            if i == 0:
                e_ref_zero_conf = ref_energy * units.Hartree
                e_model_zero_conf = atoms.get_potential_energy()
            else:
                atoms.info["ref_rel_energy"] = (
                    ref_energy * units.Hartree - e_ref_zero_conf
                )
                atoms.info["model_rel_energy"] = (
                    atoms.get_potential_energy() - e_model_zero_conf
                )
                write_dir = OUT_PATH / model_name
                write_dir.mkdir(parents=True, exist_ok=True)
                write(write_dir / f"{label}.xyz", atoms)
