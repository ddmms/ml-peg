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

from mlip_testing.calcs.models.models import MODELS
from mlip_testing.calcs.utils.utils import chdir, get_benchmark_data

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
    def read_charge(folder: Path) -> float:
        """
        Read charge from .CHRG file.

        Parameters
        ----------
        folder : Path
            Folder containing charge file.

        Returns
        -------
        float
            Charge value, 0.0 if file doesn't exist.
        """
        for f in folder.iterdir():
            if f.name.upper() == ".CHRG":
                try:
                    return float(f.read_text().strip())
                except ValueError:
                    warnings.warn(
                        f"Invalid charge in {f} - assuming neutral.", stacklevel=2
                    )
        return 0.0

    @staticmethod
    def read_atoms(folder: Path, ident: str) -> Atoms:
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
            {"identifier": ident, "charge": int(S30LBenchmark.read_charge(folder))}
        )
        return atoms

    @staticmethod
    def load_complex(index: int, root: Path) -> dict[str, Atoms]:
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
            "host": S30LBenchmark.read_atoms(base / "A", f"{index}_host"),
            "guest": S30LBenchmark.read_atoms(base / "B", f"{index}_guest"),
            "complex": S30LBenchmark.read_atoms(base / "AB", f"{index}_complex"),
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

    @staticmethod
    def benchmark_s30l(
        calc: Calculator, model_name: str, base_dir: Path
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
        refs = S30LBenchmark.parse_references(ref_file)

        complex_atoms_list = []

        for idx in tqdm(range(1, 31), desc="S30L"):
            try:
                # Load system structures
                fragments = S30LBenchmark.load_complex(idx, base_dir)
                complex_atoms = fragments["complex"]
                host_atoms = fragments["host"]
                guest_atoms = fragments["guest"]

                # Compute interaction energy
                e_int_model = S30LBenchmark.interaction_energy(fragments, calc)

                # Reference energy in eV
                e_int_ref = refs[idx]

                # Calculate errors
                error_ev = e_int_model - e_int_ref
                error_kcal = error_ev * EV_TO_KCAL_PER_MOL

                # Store additional info in complex atoms
                complex_atoms.info["model"] = model_name
                complex_atoms.info["E_int_model_kcal"] = (
                    e_int_model * EV_TO_KCAL_PER_MOL
                )
                complex_atoms.info["E_int_ref_kcal"] = e_int_ref * EV_TO_KCAL_PER_MOL
                complex_atoms.info["E_int_model_ev"] = e_int_model
                complex_atoms.info["E_int_ref_ev"] = e_int_ref
                complex_atoms.info["error_kcal"] = error_kcal
                complex_atoms.info["error_ev"] = error_ev
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

        # Get benchmark data
        base_dir = get_benchmark_data("S30L.zip") / "S30L/s30l_test_set"

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
