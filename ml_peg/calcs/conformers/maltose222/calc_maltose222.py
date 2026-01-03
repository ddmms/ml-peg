"""
Calculate the Maltose conformer energy dataset.

Journal of Chemical Theory and Computation,
2016 12 (12), 6157-6168.
DOI: 10.1021/acs.jctc.6b00876
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
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


class Maltose222Benchmark(zntrack.Node):
    """Benchmark the Maltose222 dataset."""

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

    def get_labels(self, data_path):
        """
        Get system labels.

        Parameters
        ----------
        data_path
            Path to the structure.
        """
        self.labels = []
        for system_path in sorted((data_path / "Maltose_structures").glob("*.xyz")):
            self.labels.append(system_path.stem)

    def get_ref_energies(self, data_path):
        """
        Get reference conformer energies.

        Parameters
        ----------
        data_path
            Path to the structure.
        """
        df = pd.read_csv(data_path / "maltose.csv")
        self.get_labels(data_path)
        self.ref_energies = {}
        for i, label in enumerate(self.labels):
            self.ref_energies[label] = df[" dlpno/cbs(3-4)"][i] * KCAL_TO_EV

    def run(self):
        """Run new benchmark."""
        data_path = (
            download_s3_data(
                filename="Maltose222.zip",
                key="inputs/conformers/Maltose222/Maltose222.zip",
            )
            / "Maltose222"
        )
        self.get_ref_energies(data_path)
        # Read in data and attach calculator
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        lowest_conf_label = "maltose_001"

        conf_lowest = self.get_atoms(
            data_path / "Maltose_structures" / f"{lowest_conf_label}.xyz"
        )
        conf_lowest.calc = calc
        e_conf_lowest_model = conf_lowest.get_potential_energy()

        for label, e_ref in tqdm(self.ref_energies.items()):
            # Skip the reference conformer for which the error is automatically zero
            if label == lowest_conf_label:
                continue

            atoms = self.get_atoms(data_path / "Maltose_structures" / f"{label}.xyz")
            atoms.calc = calc
            atoms.info["model_rel_energy"] = (
                atoms.get_potential_energy() - e_conf_lowest_model
            )
            atoms.info["ref_energy"] = e_ref

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
            benchmark = Maltose222Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_maltose222():
    """Run Maltose222 benchmark via pytest."""
    build_project(repro=True)
