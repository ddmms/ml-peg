"""
Calculate the UPU46 benchmark dataset for RNA backbone conformations.

Journal of Chemical Theory and Computation,
2015 11 (10), 4972-4991.
DOI: 10.1021/acs.jctc.5b00515.
"""

from __future__ import annotations

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


class UPU46Benchmark(zntrack.Node):
    """Compute the benchmark."""

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
        atoms.info["charge"] = -1
        atoms.info["spin"] = 1
        if atoms.calc is not None:
            if "energy" in atoms.calc.results:
                del atoms.calc.results["energy"]
        return atoms

    def get_ref_energies(self, data_path):
        """
        Get reference conformer energies.

        Parameters
        ----------
        data_path
            Path to the structure.
        """
        self.ref_energies = {}
        with open(data_path / "references") as lines:
            for i, line in enumerate(lines):
                # Skip the comment lines
                if i < 5:
                    continue
                items = line.strip().split()
                label = items[2]
                self.ref_energies[label] = float(items[7]) * KCAL_TO_EV

    def run(self):
        """Run new benchmark."""
        data_path = (
            download_s3_data(
                filename="UPU46.zip",
                key="inputs/conformers/UPU46/UPU46.zip",
            )
            / "UPU46"
        )
        zero_conf_label = "2p"
        self.get_ref_energies(data_path)
        # Read in data and attach calculator
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        conf_lowest = self.get_atoms(data_path / f"{zero_conf_label}.xyz")
        conf_lowest.calc = calc
        e_conf_lowest_model = conf_lowest.get_potential_energy()

        for label, e_ref in tqdm(self.ref_energies.items()):
            # Skip the reference conformer for
            # which the error is automatically zero
            if label == zero_conf_label:
                continue

            atoms = self.get_atoms(data_path / f"{label}.xyz")
            atoms.calc = calc
            atoms.info["model_rel_energy"] = (
                atoms.get_potential_energy() - e_conf_lowest_model
            )
            atoms.info["ref_energy"] = e_ref
            atoms.calc = None

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
            benchmark = UPU46Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_upu46():
    """Run UPU46 benchmark via pytest."""
    build_project(repro=True)
