"""Run calculations for elemental slab oxygen adsorption benchmark."""

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


class ElementalSlabOxygenAdsorptionBenchmark(zntrack.Node):
    """
    Benchmark model predictions of adsorption energies for oxygen on elemental slabs.

    The elemental slabs are obtained using the Materials Project API.
    For each element, the most bulk crystal is obtained. Then, the most stable surface
    from this bulk crystal is obtained.
    The details of how these surfaces are obtained are in Tran et al. [1].

    The single-point energy calculations are performed for the following systems:
    - Isolated slab
    - Isolated oxygen atom
    - Oxygen+slab: the oxygen atom is place on-top of the atom furthest along the
    surface normal direction,
    at a height which minimizes the energy according to MACE-MATPES-r2SCAN.

    The adsorption energy (E(molecule+surface) - E(surface) - E(molecule)) is calculated
    for elements.

    The reference energies are calculated using `pymatgen.io.vasp.sets.MatPESStaticSet`,
    with `user_incar_setting`. The following modifications to the default settings are
    made.

    {
    "ENCUT", 520
    "LREAL": "Auto",
    "LDIPOL: True",
    "IDIPOL": 3,
    "DIPOL": <slab-center-of-mass>,
    }

    [1] Tran, Richard, et al. "Surface energies of elemental crystals." Scientific data
    3.1 (2016): 1-13 (https://doi.org/10.1038/sdata.2016.80).
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
        Evaluate energies for elemental slab oxygen adsorption structures.

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
        """Run oxygen adsorption energy calculations."""
        calc = self.model.get_calculator()
        data_dir = (
            get_benchmark_data("elemental_slab_oxygen_adsorption_pbe.zip")
            / "elemental_slab_oxygen_adsorption_pbe/"
        )
        triplets_list = [read(xyz_path, ":") for xyz_path in data_dir.glob("*.xyz")]

        for triplet in tqdm(
            triplets_list,
            desc=f"Processing Triplets for model: {self.model_name}",
        ):
            self.evaluate_energies(triplet, calc)

        # Write all structures organized by system
        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        for triplet in triplets_list:
            system_name = triplet[0].info["system_name"]
            write(write_dir / f"{system_name}.xyz", triplet)


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
            benchmark = ElementalSlabOxygenAdsorptionBenchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_elemental_slab_oxygen_adsorption():
    """Run elemental slab oxygen adsorption benchmark via pytest."""
    build_project(repro=True)
