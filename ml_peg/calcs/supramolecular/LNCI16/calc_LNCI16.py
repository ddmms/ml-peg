"""Run calculations for LNCI16 benchmark."""

from __future__ import annotations

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

# Constants
KCAL_PER_MOL_TO_EV = units.kcal / units.mol
EV_TO_KCAL_PER_MOL = 1.0 / KCAL_PER_MOL_TO_EV

# Reference energies and charges from LNCI16 dataset
LNCI16_REFERENCE_ENERGIES = {
    "BpocBenz": -6.81,
    "BpocMeOH": -6.19,
    "BNTube": -14.32,
    "GramA": -36.30,
    "DHComplex": -57.57,
    "DNA": -363.30,
    "SH3": -25.65,
    "TYK2": -72.31,
    "FXa": -70.73,
    "2xHB238": -74.92,
    "FullGraph": -74.13,
    "DithBrCap": -45.63,
    "BrCap": -21.12,
    "MolMus": -62.58,
    "Rotax": -55.89,
    "Nylon": -566.23,
}

LNCI16_CHARGES = {
    "BpocBenz": 0,
    "BpocMeOH": 0,
    "BNTube": 0,
    "GramA": 0,
    "DHComplex": 0,
    "DNA": 0,
    "SH3": 0,
    "TYK2": +1,
    "FXa": -2,
    "2xHB238": 0,
    "FullGraph": 0,
    "DithBrCap": 0,
    "BrCap": 0,
    "MolMus": 0,
    "Rotax": 0,
    "Nylon": 0,
}


class LNCI16Benchmark(zntrack.Node):
    """
    Benchmark model for LNCI16 dataset.

    Evaluates interaction energies for large non-covalent complexes.
    16 host-guest systems including proteins, DNA, and supramolecular assemblies.

    Each system consists of complex, host, and guest structures.
    Computes interaction energy = E(complex) - E(host) - E(guest)
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def read_charge_file(filepath: Path) -> float:
        """
        Read charge from .CHRG file.

        Parameters
        ----------
        filepath : Path
            Path to the charge file.

        Returns
        -------
        float
            Charge value, 0.0 if file doesn't exist.
        """
        try:
            with open(filepath) as f:
                charge_str = f.read().strip()
                return int(float(charge_str))
        except (FileNotFoundError, OSError):
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def load_lnci16_system(system_name: str, base_dir: Path) -> dict[str, Atoms]:
        """
        Load complex, host, and guest structures for a LNCI16 system.

        Parameters
        ----------
        system_name : str
            Name of the LNCI16 system.
        base_dir : Path
            Base directory containing system data.

        Returns
        -------
        dict[str, Atoms]
            Dictionary with 'complex', 'host', and 'guest' Atoms objects.
        """
        system_dir = base_dir / system_name
        if not system_dir.exists():
            raise FileNotFoundError(f"System directory not found: {system_dir}")

        # Load structures
        complex_atoms = read(system_dir / "complex" / "struc.xyz", format="xyz")
        host_atoms = read(system_dir / "host" / "struc.xyz", format="xyz")
        guest_atoms = read(system_dir / "guest" / "struc.xyz", format="xyz")

        # Read charges
        complex_charge = LNCI16Benchmark.read_charge_file(
            system_dir / "complex" / ".CHRG"
        )
        host_charge = LNCI16Benchmark.read_charge_file(system_dir / "host" / ".CHRG")
        guest_charge = LNCI16Benchmark.read_charge_file(system_dir / "guest" / ".CHRG")

        # Set charges and system info in atoms
        # orb requires spin as well as charge
        complex_atoms.info.update(
            {"charge": complex_charge, "system": system_name, "spin": 1}
        )
        host_atoms.info.update(
            {"charge": host_charge, "system": system_name, "spin": 1}
        )
        guest_atoms.info.update(
            {"charge": guest_charge, "system": system_name, "spin": 1}
        )

        return {
            "complex": complex_atoms,
            "host": host_atoms,
            "guest": guest_atoms,
        }

    @staticmethod
    def interaction_energy(frags: dict[str, Atoms], calc: Calculator) -> float:
        """
        Calculate interaction energy from fragments.

        Parameters
        ----------
        frags : dict[str, Atoms]
            Dictionary containing 'complex', 'host', and 'guest' fragments.
        calc : Calculator
            ASE calculator for energy calculations.

        Returns
        -------
        float
            Interaction energy in eV.
        """
        # Use copies to avoid potential caching issues between fragments
        complex_copy = frags["complex"].copy()
        host_copy = frags["host"].copy()
        guest_copy = frags["guest"].copy()

        complex_copy.calc = calc
        e_complex = complex_copy.get_potential_energy()
        host_copy.calc = calc
        e_host = host_copy.get_potential_energy()
        guest_copy.calc = calc
        e_guest = guest_copy.get_potential_energy()
        return e_complex - e_host - e_guest

    @staticmethod
    def benchmark_lnci16(
        calc: Calculator, model_name: str, base_dir: Path
    ) -> list[Atoms]:
        """
        Benchmark LNCI16 dataset.

        Parameters
        ----------
        calc : Calculator
            ASE calculator for energy calculations.
        model_name : str
            Name of the model being benchmarked.
        base_dir : Path
            Base directory containing LNCI16 data.

        Returns
        -------
        list[Atoms]
            List of complex structures.
        """
        print(f"Benchmarking LNCI16 with {model_name}...")

        # Check if calculator supports charges
        supports_charges = any(
            hasattr(calc, attr) for attr in ["set_charge", "charge", "total_charge_key"]
        )
        if supports_charges:
            print(f"  Calculator {model_name} supports charge handling")
        else:
            print(f"  Calculator {model_name} may not support charge handling")

        complex_atoms_list = []

        for system_name in tqdm(LNCI16_REFERENCE_ENERGIES.keys(), desc="LNCI16"):
            # Load system structures
            frags = LNCI16Benchmark.load_lnci16_system(system_name, base_dir)
            complex_atoms = frags["complex"]
            host_atoms = frags["host"]
            guest_atoms = frags["guest"]

            # Log charge information for charged systems
            if complex_atoms.info["charge"] != 0:
                print(
                    f"  Processing charged system {system_name} "
                    f"(charge = {complex_atoms.info['charge']:+.0f})"
                )

            # Compute interaction energy
            e_int_model = LNCI16Benchmark.interaction_energy(frags, calc)

            # Reference energy in kcal/mol, convert to eV
            e_int_ref_kcal = LNCI16_REFERENCE_ENERGIES[system_name]
            e_int_ref_ev = e_int_ref_kcal * KCAL_PER_MOL_TO_EV

            # Calculate errors
            error_ev = e_int_model - e_int_ref_ev
            error_kcal = error_ev * EV_TO_KCAL_PER_MOL

            # Store additional info in complex atoms
            complex_atoms.info["model"] = model_name
            complex_atoms.info["E_int_model_kcal"] = e_int_model * EV_TO_KCAL_PER_MOL
            complex_atoms.info["E_int_ref_kcal"] = e_int_ref_kcal
            complex_atoms.info["error_kcal"] = error_kcal
            complex_atoms.info["system"] = system_name
            complex_atoms.info["charged"] = complex_atoms.info["charge"] != 0
            complex_atoms.info["complex_charge"] = complex_atoms.info["charge"]
            complex_atoms.info["host_charge"] = host_atoms.info["charge"]
            complex_atoms.info["guest_charge"] = guest_atoms.info["charge"]

            complex_atoms_list.append(complex_atoms)

        return complex_atoms_list

    def run(self):
        """Run LNCI16 benchmark calculations."""
        calc = self.model.get_calculator()

        # Get benchmark data
        base_dir = (
            get_benchmark_data("LNCI16_data.zip") / "LNCI16_data/benchmark-LNCI16-main"
        )

        # Run benchmark
        complex_atoms = self.benchmark_lnci16(calc, self.model_name, base_dir)

        # Write output structures
        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        # Save individual complex atoms files for each system
        for i, atoms in enumerate(complex_atoms):
            # Temp fix: Clear calculator to avoid array broadcasting issues
            atoms_copy = atoms.copy()
            atoms_copy.calc = None

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
            benchmark = LNCI16Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_lnci16():
    """Run LNCI16 benchmark via pytest."""
    build_project(repro=True)
