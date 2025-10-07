"""Run calculations for S24 benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import chdir, get_benchmark_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"


class S24Benchmark(zntrack.Node):
    """
    Benchmark model for S24 dataset.

    Adsorption energy calculations for molecule-surface systems.
    - 24 molecule-surface combinations
    - Each system consists of surface, molecule+surface, and isolated molecule
    - Computes adsorption energy = E(molecule+surface) - E(surface) - E(molecule)
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def compute_adsorption_energy(
        surface_e: float, mol_surf_e: float, molecule_e: float
    ) -> float:
        """
        Compute adsorption energy.

        Parameters
        ----------
        surface_e
            Energy of the clean surface.
        mol_surf_e
            Energy of the molecule+surface system.
        molecule_e
            Energy of the isolated molecule.

        Returns
        -------
        float
            Adsorption energy.
        """
        return mol_surf_e - (surface_e + molecule_e)

    @staticmethod
    def evaluate_energies(atoms_list: list[Atoms], calc: Calculator) -> None:
        """
        Evaluate energies for S24 structures.

        Parameters
        ----------
        atoms_list
            List of Atoms structures to calculate energies for.
        calc
            Calculator to use to evaluate structure energy.
        """
        for atoms in atoms_list:
            atoms.calc = deepcopy(calc)
            atoms.get_potential_energy()

    def run(self):
        """Run S24 energy calculations."""
        calc = self.model.get_calculator()
        data = get_benchmark_data("s24.zip") / "s24/s24_data.extxyz"
        atoms_list = read(data, ":")

        for i in tqdm(
            range(0, len(atoms_list), 3),
            desc=f"Processing Triplets for model: {self.model_name}",
        ):
            surface = atoms_list[i]
            mol_surface = atoms_list[i + 1]
            molecule = atoms_list[i + 2]

            # Create sys_id
            sys_id = f"{(i // 3) + 1:03d}"

            # Store reference energies
            surface.info["ref_energy"] = surface.get_potential_energy()
            mol_surface.info["ref_energy"] = mol_surface.get_potential_energy()
            molecule.info["ref_energy"] = molecule.get_potential_energy()

            # Store system information
            surface_formula = surface.get_chemical_formula()
            molecule_formula = molecule.get_chemical_formula()
            system_name = f"{surface_formula}-{molecule_formula}"

            mol_surface.info["sys_id"] = sys_id
            mol_surface.info["system_name"] = system_name

            # Evaluate with the model
            triplet = [surface, mol_surface, molecule]
            self.evaluate_energies(triplet, calc)

            # Calculate and store adsorption energies
            surface_e = surface.get_potential_energy()
            mol_surf_e = mol_surface.get_potential_energy()
            molecule_e = molecule.get_potential_energy()
            pred_ads_energy = self.compute_adsorption_energy(
                surface_e, mol_surf_e, molecule_e
            )

            ref_surface_e = surface.info["ref_energy"]
            ref_mol_surf_e = mol_surface.info["ref_energy"]
            ref_molecule_e = molecule.info["ref_energy"]
            ref_ads_energy = self.compute_adsorption_energy(
                ref_surface_e, ref_mol_surf_e, ref_molecule_e
            )

            # Store adsorption energies in mol_surface
            mol_surface.info["adsorption_energy"] = pred_ads_energy
            mol_surface.info["ref_adsorption_energy"] = ref_ads_energy

            # Save individual molecule-surface structure using sys_id
            write_dir = OUT_PATH / self.model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{sys_id}.xyz", mol_surface)


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
            benchmark = S24Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_s24():
    """Run S24 benchmark via pytest."""
    build_project(repro=True)
