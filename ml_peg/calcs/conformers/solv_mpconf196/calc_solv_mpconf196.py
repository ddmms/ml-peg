"""
Calculate the solvMPCONF196 dataset of solvated biomolecule conformers.

J. Comput. Chem. 2024, 45(7), 419.
https://doi.org/10.1002/jcc.27248.
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
import numpy as np
import pandas as pd
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import chdir, download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV

OUT_PATH = Path(__file__).parent / "outputs"

MOLECULES = [
    "FGG",
    "GFA",
    "GGF",
    "WG",
    "WGG",
    "CAMVES",
    "CHPSAR",
    "COHVAW",
    "GS464992",
    "GS557577",
    "POXTRD",
    "SANGLI",
    "YIVNOG",
]


class SolvMPCONF196Benchmark(zntrack.Node):
    """Benchmark the solvMPCONF196 dataset."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def get_atoms(atoms_path):
        """
        Read atoms object and add charge and spin.

        Parameters
        ----------
        atoms_path
            Path to atoms object.

        Returns
        -------
        atoms
            ASE atoms object.
        """
        atoms = read(atoms_path)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        return atoms

    def get_ref_energies(self, data_path):
        """
        Get reference conformer energies.

        Parameters
        ----------
        data_path
            Path to the structure.
        """
        df = pd.read_excel(
            data_path / "Energies_CCSD(T).xlsx",
            sheet_name="Total PNO-CCS(T) Energies solvM",
            header=1,
        )
        self.ref_energies = {}
        for row in df.iterrows():
            label = row[1][0]
            e_ref = float(row[1][1]) * units.Hartree
            self.ref_energies[label] = e_ref

    def run(self):
        """Run new benchmark."""
        data_path = (
            download_s3_data(
                filename="SolvMPCONF196.zip",
                key="inputs/conformers/SolvMPCONF196/SolvMPCONF196.zip",
            )
            / "SolvMPCONF196"
        )
        self.get_ref_energies(data_path)
        # Read in data and attach calculator
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)
        for molecule in tqdm(MOLECULES):
            model_abs_energies = []
            ref_abs_energies = []
            current_molecule_labels = []
            for label, e_ref in self.ref_energies.items():
                molecule_label = label.split("_")[0]
                conformer_label = label.split("_")[1]
                if label[-1].isnumeric():
                    xyz_label = f"{molecule_label}{conformer_label}"
                else:
                    xyz_label = f"{molecule_label}_{conformer_label}"
                if molecule != molecule_label:
                    continue
                atoms = self.get_atoms(
                    data_path
                    / "solvMPCONF196_geometries/solvMPCONF196"
                    / xyz_label
                    / "struc.xyz"
                )
                atoms.translate(-atoms.get_center_of_mass())
                atoms.calc = calc
                model_abs_energies.append(atoms.get_potential_energy())
                ref_abs_energies.append(e_ref)
                current_molecule_labels.append(label)

            for label, e_model in zip(
                current_molecule_labels, model_abs_energies, strict=False
            ):
                molecule_label = label.split("_")[0]
                conformer_label = label.split("_")[1]
                if label[-1].isnumeric():
                    xyz_label = f"{molecule_label}{conformer_label}"
                else:
                    xyz_label = f"{molecule_label}_{conformer_label}"
                if molecule != molecule_label:
                    continue
                atoms = self.get_atoms(
                    data_path
                    / "solvMPCONF196_geometries/solvMPCONF196"
                    / xyz_label
                    / "struc.xyz"
                )
                atoms.translate(-atoms.get_center_of_mass())
                atoms.info["ref_rel_energy"] = self.ref_energies[label] - np.mean(
                    ref_abs_energies
                )
                atoms.info["model_rel_energy"] = e_model - np.mean(model_abs_energies)

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
            benchmark = SolvMPCONF196Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_solv_mpconf196():
    """Run solvMPCONF196 benchmark via pytest."""
    build_project(repro=True)
