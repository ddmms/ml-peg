"""
Calculate the OpenFF-Tors benchmark dataset for torsional angles.

The Journal of Physical Chemistry B 2024 128 (32), 7888-7902.
DOI: 10.1021/acs.jpcb.4c03167.
"""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms, units
from ase.io import write
import mlipx
from mlipx.abc import NodeWithCalculator
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import chdir, download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV

OUT_PATH = Path(__file__).parent / "outputs"


class OpenFFTorsBenchmark(zntrack.Node):
    """Compute the benchmark."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def run(self):
        """Run the benchmark."""
        data_path = (
            download_s3_data(
                filename="OpenFF-Tors.zip",
                key="inputs/conformers/OpenFF-Tors/OpenFF-Tors.zip",
            )
            / "OpenFF-Tors"
        )
        # Read in data and attach calculator
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)
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
            remapped_symbols = [
                symbols[atom_map[i]] for i in range(1, len(symbols) + 1)
            ]

            for i, (ref_energy, positions) in enumerate(
                zip(conf["final_energies"], conf["final_geometries"], strict=False)
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
                    write_dir = OUT_PATH / self.model_name
                    write_dir.mkdir(parents=True, exist_ok=True)
                    write(write_dir / f"{label}.xyz", atoms)


def build_project(repro: bool = False) -> None:
    """
    Build mlipx project.

    Parameters
    ----------
    repro
        Whether to call dvc repro -f after building.
    """
    project = mlipx.Project()
    benchmark_node_dict = {}

    for model_name, model in MODELS.items():
        with project.group(model_name):
            benchmark = OpenFFTorsBenchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_openff_tors():
    """Run OpenFF-Tors benchmark via pytest."""
    build_project(repro=True)
