"""Run calculations for Wiggle150 benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ase import Atoms, units
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

# Unit conversion
KCAL_PER_MOL_TO_EV = units.kcal / units.mol
EV_TO_KCAL_PER_MOL = 1.0 / KCAL_PER_MOL_TO_EV

# Dataset constants
DATA_ARCHIVE = "wiggle150-structures.zip"
DATA_SUBDIR = "wiggle150-structures"
DATA_FILENAME = "ct5c00015_si_003.xyz"
MOLECULE_ORDER = ("ado", "bpn", "efa")


class Wiggle150Benchmark(zntrack.Node):
    """
    Benchmark model on Wiggle150 molecular conformer dataset.

    Wiggle150 contains 150 strained conformers (50 each for ado, bpn, efa).
    Relative energies are evaluated against DLPNO-CCSD(T)/CBS references.
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def parse_structure_info(atoms: Atoms) -> tuple[str, float]:
        """
        Extract structure identifier and reference energy from Atoms.info.

        Parameters
        ----------
        atoms
            Input structure with metadata stored in info keys.

        Returns
        -------
        tuple[str, float]
            Structure label and absolute reference energy.
        """
        info_keys = list(atoms.info.keys())
        if len(info_keys) < 2:
            raise ValueError("Structure metadata missing required keys.")

        label = info_keys[0]
        ref_energy = float(info_keys[1])
        return label, ref_energy

    @staticmethod
    def load_structures(base_dir: Path) -> dict[str, dict[str, Iterable[Atoms]]]:
        """
        Load Wiggle150 structures and organise by molecule.

        Parameters
        ----------
        base_dir
            Directory containing the extracted dataset.

        Returns
        -------
        dict
            Mapping of molecule id to ground state and conformers.
        """
        dataset_path = base_dir / DATA_FILENAME
        structures = read(dataset_path, ":")

        molecules: dict[str, dict[str, list[Atoms]]] = {
            mol: {"ground": None, "conformers": []} for mol in MOLECULE_ORDER
        }

        for atoms in structures:
            label, _ = Wiggle150Benchmark.parse_structure_info(atoms)
            molecule = label.split("_")[0]
            if molecule not in molecules:
                continue

            if label.endswith("_00"):
                molecules[molecule]["ground"] = atoms
            else:
                molecules[molecule]["conformers"].append(atoms)

        for mol, entries in molecules.items():
            if entries["ground"] is None:
                raise FileNotFoundError(f"Missing ground-state structure for {mol}.")

        return molecules

    @staticmethod
    def get_energy(atoms: Atoms, calc: Calculator) -> float:
        """
        Evaluate potential energy for a structure.

        Parameters
        ----------
        atoms
            Structure to evaluate.
        calc
            ASE calculator instance.

        Returns
        -------
        float
            Potential energy in eV.
        """
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        energy = atoms_copy.get_potential_energy()
        return float(energy)

    @staticmethod
    def benchmark_wiggle150(
        calc: Calculator, model_name: str, base_dir: Path
    ) -> list[Atoms]:
        """
        Run Wiggle150 benchmark for a given calculator.

        Parameters
        ----------
        calc
            ASE calculator for predictions.
        model_name
            Name of the model.
        base_dir
            Path to extracted Wiggle150 data.

        Returns
        -------
        list[Atoms]
            Conformer structures annotated with prediction metadata.
        """
        molecules = Wiggle150Benchmark.load_structures(base_dir)
        conformer_atoms: list[Atoms] = []

        for molecule in MOLECULE_ORDER:
            entries = molecules[molecule]
            ground = entries["ground"]
            ground_label, ground_ref_energy = Wiggle150Benchmark.parse_structure_info(
                ground
            )
            ground_energy_model = Wiggle150Benchmark.get_energy(ground, calc)

            for atoms in tqdm(
                entries["conformers"], desc=f"Wiggle150 {molecule.upper()}"
            ):
                label, ref_energy = Wiggle150Benchmark.parse_structure_info(atoms)
                model_energy = Wiggle150Benchmark.get_energy(atoms, calc)

                rel_energy_model_kcal = (
                    model_energy - ground_energy_model
                ) * EV_TO_KCAL_PER_MOL
                rel_energy_ref_kcal = ref_energy - ground_ref_energy
                error_kcal = rel_energy_model_kcal - rel_energy_ref_kcal

                annotated = atoms.copy()
                annotated.calc = None
                annotated.info.clear()
                annotated.info.update(
                    {
                        "structure": label,
                        "molecule": molecule,
                        "ground_state": ground_label,
                        "relative_energy_ref_kcal": rel_energy_ref_kcal,
                        "relative_energy_pred_kcal": rel_energy_model_kcal,
                        "relative_energy_error_kcal": error_kcal,
                        "model": model_name,
                    }
                )
                conformer_atoms.append(annotated)

        return conformer_atoms

    def run(self) -> None:
        """Execute Wiggle150 benchmark calculations."""
        calc = self.model.get_calculator()

        base_dir = get_benchmark_data(DATA_ARCHIVE) / DATA_SUBDIR

        conformers = self.benchmark_wiggle150(calc, self.model_name, base_dir)

        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        for index, atoms in enumerate(conformers):
            atoms.info["index"] = index
            write(write_dir / f"{index}.xyz", atoms, format="extxyz")


def build_project(repro: bool = False) -> None:
    """
    Build mlipx project for Wiggle150 benchmark.

    Parameters
    ----------
    repro
        Whether to trigger ``dvc repro -f`` after building.
    """
    project = mlipx.Project()

    for model_name, model in MODELS.items():
        with project.group(model_name):
            Wiggle150Benchmark(
                model=model,
                model_name=model_name,
            )

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_wiggle150() -> None:
    """Run Wiggle150 benchmark via pytest."""
    build_project(repro=True)
