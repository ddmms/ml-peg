"""
Compute the NCIA SH250x10 dataset for sigma-hole interactions.

Phys. Chem. Chem. Phys., 2022,24, 14794-14804.
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


class NCIASH250x10Benchmark(zntrack.Node):
    """Benchmark the NCIA_SH250x10 dataset."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def get_ref_energies(self, data_path):
        """
        Get reference energies.

        Parameters
        ----------
        data_path
            Path to data.
        """
        self.ref_energies = {}
        with open(data_path / "NCIA_SH250x10_benchmark.txt") as lines:
            for i, line in enumerate(lines):
                if i < 2:
                    continue
                items = line.strip().split()
                label = items[0]
                ref_energy = float(items[1]) * KCAL_TO_EV
                self.ref_energies[label] = ref_energy

    @staticmethod
    def get_monomers(atoms):
        """
        Get ASE atoms objects of the monomers.

        Parameters
        ----------
        atoms
            ASE atoms object of the structure.

        Returns
        -------
        tuple[ASE.Atoms, ASE.Atoms]
            Tuple containing the two monomers.
        """
        if isinstance(atoms.info["selection_a"], str):
            a_ids = [int(id) for id in atoms.info["selection_a"].split("-")]
            a_ids[0] -= 1
        else:
            a_ids = [int(atoms.info["selection_a"]) - 1, int(atoms.info["selection_a"])]

        if isinstance(atoms.info["selection_b"], str):
            b_ids = [int(id) for id in atoms.info["selection_b"].split("-")]
            b_ids[0] -= 1
        else:
            b_ids = [int(atoms.info["selection_b"]) - 1, int(atoms.info["selection_b"])]

        atoms_a = atoms[a_ids[0] : a_ids[1]]
        atoms_b = atoms[b_ids[0] : b_ids[1]]
        assert len(atoms_a) + len(atoms_b) == len(atoms)

        atoms_a.info["charge"] = int(atoms.info["charge_a"])
        atoms_a.info["spin"] = 1

        atoms_b.info["charge"] = int(atoms.info["charge_b"])
        atoms_b.info["spin"] = 1
        return (atoms_a, atoms_b)

    def run(self):
        """Run new benchmark."""
        # Read in data and attach calculator
        data_path = (
            download_s3_data(
                filename="NCIA_SH250x10.zip",
                key="inputs/non_covalent_interactions/NCIA_SH250x10/NCIA_SH250x10.zip",
            )
            / "NCIA_SH250x10"
        )
        self.get_ref_energies(data_path)

        calc = self.model.get_calculator()

        for label, ref_energy in tqdm(self.ref_energies.items()):
            xyz_fname = f"{label}.xyz"
            atoms = read(data_path / "geometries" / xyz_fname)
            atoms_a, atoms_b = self.get_monomers(atoms)
            atoms.info["spin"] = 1
            atoms.info["charge"] = int(atoms_a.info["charge"] + atoms_b.info["charge"])
            atoms.calc = calc
            atoms_a.calc = calc
            atoms_b.calc = calc

            atoms.info["model_int_energy"] = (
                atoms.get_potential_energy()
                - atoms_a.get_potential_energy()
                - atoms_b.get_potential_energy()
            )
            atoms.info["ref_int_energy"] = ref_energy
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
            benchmark = NCIASH250x10Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_ncia_sh250x10():
    """Run NCIA_SH250x10 barriers benchmark via pytest."""
    build_project(repro=True)
