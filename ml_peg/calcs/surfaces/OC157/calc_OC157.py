"""Run calculations for OC157 benchmark."""

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


# OC157 benchmark node
class OC157Benchmark(zntrack.Node):
    """
    Benchmark model for OC157 dataset.

    Prediction of the most stable structures for a molecule-surface system
    - relative energies between 3 structures and 157 molecule surface combinations
    - identification of the most stable structure

    reference: MPRelaxSet DFT (Becke-Johnson damped D3 dispersion correction)
    - surfaces taken from the Open Catalyst Challenge 2023
    - 200 refs but excludes those with Hubbard U -> 157
    - 3 structures per system (triplet)
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def _incar_has_hubbard_u(incar_path: Path) -> bool:
        """
        Check whether Hubbard U correction was used.

        Parameters
        ----------
        incar_path
            Path to reference input file.

        Returns
        -------
        bool
            Whether Hubbard U correction was used.
        """
        if not incar_path.is_file():
            return False
        return "LDAU" in incar_path.read_text()

    @staticmethod
    def _find_energy(outcar, key="energy  without entropy=") -> float:
        """
        Find reference energy.

        Parameters
        ----------
        outcar
            Reference output file.
        key
            Key to access energy result.

        Returns
        -------
        float
            Reference energy.
        """
        with open(outcar, encoding="ISO-8859-1") as fh:
            hits = [line for line in fh if key in line]
        if not hits:
            raise RuntimeError(f"No energy found in {outcar}")
        return float(hits[-1].split()[-1])

    @staticmethod
    def _read_structure(folder: Path) -> Atoms:
        """
        Read reference structure.

        Parameters
        ----------
        folder
            Folder containing reference files.

        Returns
        -------
        Atoms
            Reference structure as an Atoms object.
        """
        for fname in ("CONTCAR", "POSCAR"):
            fpath = folder / fname
            if fpath.is_file():
                return read(fpath, format="vasp")
        raise FileNotFoundError(f"No CONTCAR/POSCAR in {folder}")

    @staticmethod
    def evaluate_energies(triplet: list[Atoms], calc: Calculator) -> None:
        """
        Evaluate energies for each triplet.

        Parameters
        ----------
        triplet
            Triplet of Atoms structures to calculate energies for.
        calc
            Calculator to use to evaluate structure energy.
        """
        for atoms in triplet:
            atoms.calc = deepcopy(calc)
            atoms.get_potential_energy()

    def run(self):
        """Run OC157 energy calculations."""
        calc = self.model.get_calculator()
        base_dir = get_benchmark_data("OC_Dataset.zip") / "OC_Dataset"
        n_systems = 200
        skip_hubbard_u = True

        triplets = []
        system_ids = []
        system_compositions = []

        for idx in tqdm(range(1, n_systems + 1), desc="Loading OC157 systems"):
            sys_id = f"{idx:03d}"
            sys_dir = base_dir / sys_id

            if skip_hubbard_u and self._incar_has_hubbard_u(sys_dir / "1" / "INCAR"):
                continue

            poscar = (sys_dir / "1" / "POSCAR").read_text().splitlines()[0].strip()
            system_ids.append(sys_id)
            system_compositions.append(poscar)

            trio_atoms = []
            for member in (1, 2, 3):
                subdir = sys_dir / str(member)
                atoms = self._read_structure(subdir)
                energy = self._find_energy(subdir / "OUTCAR")
                atoms.info["ref_energy"] = energy
                atoms.info["composition"] = poscar
                atoms.info["sys_id"] = sys_id
                trio_atoms.append(atoms)
            triplets.append(trio_atoms)

        for trio in tqdm(triplets, desc="Evaluating model on triplets"):
            self.evaluate_energies(trio, calc)
            write_dir = OUT_PATH / self.model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{trio[-1].info['sys_id']}.xyz", trio)


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
            benchmark = OC157Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_oc157():
    """Run OC157 benchmark via pytest."""
    build_project(repro=True)
