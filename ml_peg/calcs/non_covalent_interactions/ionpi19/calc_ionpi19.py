"""
Compute the IONPI19 dataset for ion-pi system interactions.

Phys. Chem. Chem. Phys., 2021,23, 11635-11648.
"""

from __future__ import annotations

import os
from pathlib import Path

from ase import units
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import chdir, download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV

OUT_PATH = Path(__file__).parent / "outputs"


class IONPI19Benchmark(zntrack.Node):
    """Benchmark the IONPI19 dataset."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    species = {
        1: ["1_AB", "1_A", "1_B"],
        2: ["2_AB", "2_A", "2_B"],
        3: ["3_AB", "3_A", "3_B"],
        4: ["4_AB", "4_A", "4_B"],
        5: ["5_AB", "5_A", "5_B"],
        6: ["6_AB", "6_A", "6_B"],
        7: ["7_AB", "7_A", "7_B"],
        8: ["8_AB", "8_A", "8_B"],
        9: ["9_AB", "9_A", "9_B"],
        10: ["10_AB", "10_A", "10_B"],
        11: ["11_AB", "11_A", "11_B"],
        12: ["12_AB", "12_A", "12_B"],
        13: ["13_AB", "13_A", "13_B"],
        14: ["14_AB", "14_A", "14_B"],
        15: ["15_AB", "15_A", "15_B"],
        16: ["16_AB", "15_A", "16_B"],
        17: ["17_AB", "17_A", "17_B"],
        18: ["18_A", "18_B"],
        19: ["19_A", "19_B"],
    }

    stoich = {
        1: ["1", "-1", "-1"],
        2: ["1", "-1", "-1"],
        3: ["1", "-1", "-1"],
        4: ["1", "-1", "-1"],
        5: ["1", "-1", "-1"],
        6: ["1", "-1", "-1"],
        7: ["1", "-1", "-1"],
        8: ["1", "-1", "-1"],
        9: ["1", "-1", "-1"],
        10: ["1", "-1", "-1"],
        11: ["1", "-1", "-1"],
        12: ["1", "-1", "-1"],
        13: ["1", "-1", "-1"],
        14: ["1", "-1", "-1"],
        15: ["1", "-1", "-1"],
        16: ["1", "-1", "-1"],
        17: ["1", "-1", "-1"],
        18: ["1", "-1"],
        19: ["1", "-1"],
    }

    @staticmethod
    def get_atoms(path):
        """
        Get the atoms object with charge and spin.

        Parameters
        ----------
        path
            Path to atoms.

        Returns
        -------
        ASE.Atoms
            Atoms object of the system.
        """
        chrg = path / "CHRG"
        mol_fname = path / "mol.xyz"
        atoms = read(mol_fname)
        atoms.info["spin"] = 1
        if not os.path.exists(chrg):
            charge = 0
        else:
            with open(chrg) as lines:
                for line in lines:
                    items = line.strip().split()
                    charge = int(items[0])
        atoms.info["charge"] = charge
        return atoms

    def get_ref_energies(self, data_path):
        """
        Extract the reference energies.

        Parameters
        ----------
        data_path
            Path to data.
        """
        self.ref_energies = {}
        with open(data_path / "energies.txt") as lines:
            for i, line in enumerate(lines):
                if i < 2:
                    continue
                items = line.strip().split()
                system_id = int(items[0])
                self.ref_energies[system_id] = float(items[1]) * KCAL_TO_EV

    def run(self):
        """Run new benchmark."""
        data_path = (
            download_s3_data(
                filename="IONPI19.zip",
                key="inputs/non_covalent_interactions/IONPI19/IONPI19.zip",
            )
            / "ionpi19"
        )
        self.get_ref_energies(data_path)
        # Read in data and attach calculator
        calc = self.model.get_calculator()

        for system_id in tqdm(range(1, 20)):
            for config in self.species[system_id]:
                label = f"{config}"
                atoms = self.get_atoms(data_path / label)
                atoms.info["ref_int_energy"] = self.ref_energies[system_id]
                atoms.calc = calc
                atoms.info["model_energy"] = atoms.get_potential_energy()

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
            benchmark = IONPI19Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_ionpi19():
    """Run IONPI19 benchmark via pytest."""
    build_project(repro=True)
