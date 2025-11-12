"""Run calculations for S30L benchmark."""

from __future__ import annotations

from pathlib import Path
import warnings

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import chdir, download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"

# Constants
KCAL_PER_MOL_TO_EV = units.kcal / units.mol
EV_TO_KCAL_PER_MOL = 1.0 / KCAL_PER_MOL_TO_EV


class S30LBenchmark(zntrack.Node):
    """
    Benchmark model for S30L dataset.

    Evaluates interaction energies for 30 host-guest supramolecular complexes.
    Each complex consists of host (A), guest (B), and complex (AB) structures.
    Computes interaction energy = E(AB) - E(A) - E(B)
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def read_charge(folder: Path) -> int:
        """
        Read charge from .CHRG file.

        Parameters
        ----------
        folder : Path
            Folder containing charge file.

        Returns
        -------
        int
            Charge value, 0 if file doesn't exist.
        """
        for f in folder.iterdir():
            if f.name.upper() == ".CHRG":
                try:
                    return int(f.read_text().strip())
                except ValueError:
                    warnings.warn(
                        f"Invalid charge in {f} - assuming neutral.", stacklevel=2
                    )
        return 0

    def read_atoms(self, folder: Path, ident: str) -> Atoms:
        """
        Read Turbomole format structure from folder.

        Parameters
        ----------
        folder : Path
            Folder containing coord file.
        ident : str
            Identifier for the structure.

        Returns
        -------
        Atoms
            ASE Atoms object with charge and identifier info.
        """
        coord = next(
            (p for p in folder.iterdir() if p.name.lower().startswith("coord")), None
        )
        if coord is None:
            raise FileNotFoundError(f"No coord file in {folder}")
        atoms = read(coord, format="turbomole")
        atoms.info.update(
            {"identifier": ident, "charge": self.read_charge(folder), "spin": 1}
        )
        return atoms

    def load_complex(self, index: int, root: Path) -> dict[str, Atoms]:
        """
        Load host, guest, and complex structures for a S30L system.

        Parameters
        ----------
        index : int
            System index (1-30).
        root : Path
            Root directory containing system data.

        Returns
        -------
        Dict[str, Atoms]
            Dictionary with 'host', 'guest', and 'complex' Atoms objects.
        """
        base = root / f"{index}"
        if not base.exists():
            raise FileNotFoundError(base)
        return {
            "host": self.read_atoms(base / "A", f"{index}_host"),
            "guest": self.read_atoms(base / "B", f"{index}_guest"),
            "complex": self.read_atoms(base / "AB", f"{index}_complex"),
        }

    @staticmethod
    def interaction_energy(frags: dict[str, Atoms], calc: Calculator) -> float:
        """
        Calculate interaction energy from fragments.

        Parameters
        ----------
        frags : Dict[str, Atoms]
            Dictionary containing 'host', 'guest', and 'complex' fragments.
        calc : Calculator
            ASE calculator for energy calculations.

        Returns
        -------
        float
            Interaction energy in eV.
        """
        frags["complex"].calc = calc
        e_complex = frags["complex"].get_potential_energy()
        frags["host"].calc = calc
        e_host = frags["host"].get_potential_energy()
        frags["guest"].calc = calc
        e_guest = frags["guest"].get_potential_energy()
        return e_complex - e_host - e_guest

    @staticmethod
    def parse_references(path: Path) -> dict[int, float]:
        """
        Parse reference energies from S30L reference file.

        Parameters
        ----------
        path : Path
            Path to reference file.

        Returns
        -------
        Dict[int, float]
            Dictionary mapping system index to reference energy in eV.
        """
        refs: dict[int, float] = {}
        for idx, ln in enumerate(path.read_text().splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            kcal = float(ln.split()[0])
            refs[idx + 1] = kcal * KCAL_PER_MOL_TO_EV
        return refs

    def benchmark_s30l(
        self, calc: Calculator, model_name: str, base_dir: Path
    ) -> list[Atoms]:
        """
        Benchmark S30L dataset.

        Parameters
        ----------
        calc : Calculator
            ASE calculator for energy calculations.
        model_name : str
            Name of the model being benchmarked.
        base_dir : Path
            Base directory containing S30L data.

        Returns
        -------
        list[Atoms]
            List of complex structures.
        """
        print(f"Benchmarking S30L with {model_name}...")

        ref_file = base_dir / "references_s30.txt"
        refs = self.parse_references(ref_file)

        complex_atoms_list = []

        for idx in tqdm(range(1, 31), desc="S30L"):
            try:
                # Load system structures
                fragments = self.load_complex(idx, base_dir)
                complex_atoms = fragments["complex"]
                host_atoms = fragments["host"]
                guest_atoms = fragments["guest"]

                # Compute interaction energy
                e_int_model = self.interaction_energy(fragments, calc)

                # Reference energy in eV
                e_int_ref = refs[idx]

                # Store additional info in complex atoms
                complex_atoms.info["model"] = model_name
                complex_atoms.info["E_int_model"] = e_int_model
                complex_atoms.info["E_int_ref"] = e_int_ref
                complex_atoms.info["system_index"] = idx
                complex_atoms.info["n_atoms"] = len(complex_atoms)
                complex_atoms.info["host_charge"] = host_atoms.info["charge"]
                complex_atoms.info["guest_charge"] = guest_atoms.info["charge"]
                complex_atoms.info["complex_charge"] = complex_atoms.info["charge"]

                complex_atoms_list.append(complex_atoms)

            except Exception as e:
                print(f"Error processing system {idx}: {e}")
                continue

        return complex_atoms_list

    def run(self):
        """Run S30L benchmark calculations."""
        calc = self.model.get_calculator()

        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        # Get benchmark data
        base_dir = (
            download_s3_data(
                filename="S30L.zip", key="inputs/supramolecular/S30L/S30L.zip"
            )
            / "S30L/s30l_test_set"
        )

        # Run benchmark
        complex_atoms = self.benchmark_s30l(calc, self.model_name, base_dir)

        # Write output structures
        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        # Save individual complex atoms files for each system
        for i, atoms in enumerate(complex_atoms):
            atoms_copy = atoms.copy()

            # Write each system to its own file
            system_file = write_dir / f"{i}.xyz"
            write(system_file, atoms_copy, format="extxyz")


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
            benchmark = S30LBenchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_s30l():
    """Run S30L benchmark via pytest."""
    build_project(repro=True)
