"""Run calculations for PLA15 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import write
import mlipx
from mlipx.abc import NodeWithCalculator
import numpy as np
from tqdm import tqdm
import zntrack

from mlip_testing.calcs.models.models import MODELS
from mlip_testing.calcs.utils.utils import chdir, get_benchmark_data

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"

# Constants
KCAL_PER_MOL_TO_EV = units.kcal / units.mol
EV_TO_KCAL_PER_MOL = 1.0 / KCAL_PER_MOL_TO_EV


class PLA15Benchmark(zntrack.Node):
    """
    Benchmark model for PLA15 dataset.

    Evaluates protein-ligand interaction energies for 15 complete active site complexes.
    Each complex consists of protein, ligand, and complex structures from PDB files.
    Computes interaction energy = E(complex) - E(protein) - E(ligand)
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def extract_charge_and_selections(
        pdb_path: Path,
    ) -> tuple[float, float, float, str, str]:
        """
        Extract charge and selection information from PDB REMARK lines.

        Parameters
        ----------
        pdb_path : Path
            Path to PDB file.

        Returns
        -------
        Tuple[float, float, float, str, str]
            Total charge, charge A, charge B, selection A, selection B.
        """
        total_charge = qa = qb = 0.0
        selection_a = selection_b = ""

        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("REMARK"):
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        break
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                tag = parts[1].lower()

                if tag == "charge":
                    total_charge = float(parts[2])
                elif tag == "charge_a":
                    qa = float(parts[2])
                elif tag == "charge_b":
                    qb = float(parts[2])
                elif tag == "selection_a":
                    selection_a = " ".join(parts[2:])
                elif tag == "selection_b":
                    selection_b = " ".join(parts[2:])

        return total_charge, qa, qb, selection_a, selection_b

    @staticmethod
    def separate_protein_ligand_simple(pdb_path: Path):
        """
        Separate protein and ligand based on residue names.

        Parameters
        ----------
        pdb_path : Path
            Path to PDB file.

        Returns
        -------
        Tuple
            All atoms, protein atoms, ligand atoms.
        """
        import MDAnalysis as mda  # noqa: N813

        # Load with MDAnalysis
        u = mda.Universe(str(pdb_path))

        # Simple separation: ligand = UNK residues, protein = everything else
        protein_atoms = []
        ligand_atoms = []

        for atom in u.atoms:
            if atom.resname.strip().upper() in ["UNK", "LIG", "MOL"]:
                ligand_atoms.append(atom)
            else:
                protein_atoms.append(atom)

        return u.atoms, protein_atoms, ligand_atoms

    @staticmethod
    def mda_atoms_to_ase(atom_list, charge: float, identifier: str) -> Atoms:
        """
        Convert MDAnalysis atoms to ASE Atoms object.

        Parameters
        ----------
        atom_list
            List of MDAnalysis atoms.
        charge : float
            Charge of the fragment.
        identifier : str
            Identifier for the structure.

        Returns
        -------
        Atoms
            ASE Atoms object.
        """
        if not atom_list:
            atoms = Atoms()
            atoms.info.update({"charge": charge, "identifier": identifier})
            return atoms

        symbols = []
        positions = []

        for atom in atom_list:
            # Get element symbol
            try:
                elem = (atom.element or "").strip().title()
            except (AttributeError, TypeError):
                elem = ""

            if not elem:
                # Fallback: first letter of atom name
                elem = "".join([c for c in atom.name if c.isalpha()])[:1].title() or "C"

            symbols.append(elem)
            positions.append(atom.position)

        atoms = Atoms(symbols=symbols, positions=np.array(positions))
        atoms.info.update({"charge": int(round(charge)), "identifier": identifier})
        return atoms

    @staticmethod
    def process_pdb_file(pdb_path: Path) -> dict[str, Atoms]:
        """
        Parse PDB file and return complex with separated fragments.

        Parameters
        ----------
        pdb_path : Path
            Path to PDB file.

        Returns
        -------
        Dict[str, Atoms]
            Dictionary with 'complex', 'protein', and 'ligand' Atoms objects.
        """
        total_charge, charge_a, charge_b, _, _ = (
            PLA15Benchmark.extract_charge_and_selections(pdb_path)
        )

        try:
            all_atoms, protein_atoms, ligand_atoms = (
                PLA15Benchmark.separate_protein_ligand_simple(pdb_path)
            )

            if len(ligand_atoms) == 0:
                print(f"Warning: No ligand atoms found in {pdb_path.name}")
                return {}

            if len(protein_atoms) == 0:
                print(f"Warning: No protein atoms found in {pdb_path.name}")
                return {}

            base_id = pdb_path.stem

            complex_atoms = PLA15Benchmark.mda_atoms_to_ase(
                list(all_atoms), total_charge, base_id
            )
            protein_frag = PLA15Benchmark.mda_atoms_to_ase(
                protein_atoms, charge_a, base_id
            )
            ligand = PLA15Benchmark.mda_atoms_to_ase(ligand_atoms, charge_b, base_id)

            return {"complex": complex_atoms, "protein": protein_frag, "ligand": ligand}

        except (ImportError, AttributeError, ValueError, OSError) as e:
            print(f"Warning: Error processing {pdb_path}: {e}")
            return {}

    @staticmethod
    def parse_pla15_references(path: Path) -> dict[str, float]:
        """
        Parse PLA15 reference interaction energies from file.

        Parameters
        ----------
        path : Path
            Path to reference file.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping system identifier to reference energy in eV.
        """
        ref: dict[str, float] = {}

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.lower().startswith("no.") or line.startswith("-"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                energy_kcal = float(parts[-1])
            except ValueError:
                continue

            # Extract full identifier with residue type
            full_identifier = parts[1].replace(".pdb", "")

            # TODO: review this part + error handling
            # Extract base identifier by removing residue type suffix
            # Format: "1ABC_15_lys" -> "1ABC_15"
            identifier_parts = full_identifier.split("_")
            if len(identifier_parts) >= 3:
                # Last part is residue type (lys, arg, asp, etc.)
                base_identifier = "_".join(identifier_parts[:-1])
            else:
                # Fallback: use full identifier if format is unexpected
                base_identifier = full_identifier

            energy_ev = energy_kcal * KCAL_PER_MOL_TO_EV  # Convert to eV
            ref[base_identifier] = energy_ev

        return ref

    @staticmethod
    def interaction_energy(fragments: dict[str, Atoms], calc: Calculator) -> float:
        """
        Calculate interaction energy from fragments.

        Parameters
        ----------
        fragments : Dict[str, Atoms]
            Dictionary containing 'complex', 'protein', and 'ligand' fragments.
        calc : Calculator
            ASE calculator for energy calculations.

        Returns
        -------
        float
            Interaction energy in eV.
        """
        fragments["complex"].calc = calc
        e_complex = fragments["complex"].get_potential_energy()
        fragments["protein"].calc = calc
        e_protein = fragments["protein"].get_potential_energy()
        fragments["ligand"].calc = calc
        e_ligand = fragments["ligand"].get_potential_energy()
        return e_complex - e_protein - e_ligand

    @staticmethod
    def benchmark_pla15(
        calc: Calculator, model_name: str, base_dir: Path
    ) -> list[Atoms]:
        """
        Benchmark PLA15 dataset.

        Parameters
        ----------
        calc : Calculator
            ASE calculator for energy calculations.
        model_name : str
            Name of the model being benchmarked.
        base_dir : Path
            Base directory containing PLA15 data.

        Returns
        -------
        list[Atoms]
            List of complex structures.
        """
        print(f"Benchmarking PLA15 with {model_name}...")

        pla15_dir = base_dir / "PLA15_pdbs"
        pla15_ref_file = pla15_dir / "reference_energies.txt"

        pla15_refs = PLA15Benchmark.parse_pla15_references(pla15_ref_file)
        pdb_files = list(pla15_dir.glob("*.pdb"))

        complex_atoms_list = []

        for pdb_file in tqdm(pdb_files, desc="PLA15"):
            identifier = pdb_file.stem
            if identifier not in pla15_refs:
                continue

            fragments = PLA15Benchmark.process_pdb_file(pdb_file)
            if not fragments:
                continue

            try:
                # Calculate interaction energy
                e_int_model = PLA15Benchmark.interaction_energy(fragments, calc)
                e_int_ref = pla15_refs[identifier]

                # Calculate errors
                error_ev = e_int_model - e_int_ref
                error_kcal = error_ev * EV_TO_KCAL_PER_MOL

                # Store additional info in complex atoms
                complex_atoms = fragments["complex"]
                complex_atoms.info["model"] = model_name
                complex_atoms.info["E_int_model_kcal"] = (
                    e_int_model * EV_TO_KCAL_PER_MOL
                )
                complex_atoms.info["E_int_ref_kcal"] = e_int_ref * EV_TO_KCAL_PER_MOL
                complex_atoms.info["E_int_model_ev"] = e_int_model
                complex_atoms.info["E_int_ref_ev"] = e_int_ref
                complex_atoms.info["error_kcal"] = error_kcal
                complex_atoms.info["error_ev"] = error_ev
                complex_atoms.info["identifier"] = identifier
                complex_atoms.info["dataset"] = "PLA15"
                complex_atoms.info["complex_atoms"] = len(complex_atoms)
                complex_atoms.info["protein_atoms"] = len(fragments["protein"])
                complex_atoms.info["ligand_atoms"] = len(fragments["ligand"])
                complex_atoms.info["complex_charge"] = complex_atoms.info["charge"]
                complex_atoms.info["protein_charge"] = fragments["protein"].info[
                    "charge"
                ]
                complex_atoms.info["ligand_charge"] = fragments["ligand"].info["charge"]

                complex_atoms_list.append(complex_atoms)

                # print(
                #     f"  {identifier}: E_int = {e_int_model:.6f} eV "
                #     f"(ref: {e_int_ref:.6f} eV, error: {error_kcal:.2f} kcal/mol)"
                # )

            except (KeyError, ValueError, RuntimeError) as e:
                print(f"Error processing {identifier}: {e}")
                continue

        return complex_atoms_list

    def run(self):
        """Run PLA15 benchmark calculations."""
        calc = self.model.get_calculator()

        # Get benchmark data
        base_dir = (
            get_benchmark_data("protein-ligand-data_PLA15_PLF547.zip")
            / "protein-ligand-data_PLA15_PLF547"
        )

        # Run benchmark
        complex_atoms = self.benchmark_pla15(calc, self.model_name, base_dir)

        # Write output structures
        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        # Save individual complex atoms files for each system
        for i, atoms in enumerate(complex_atoms):
            atoms_copy = atoms.copy()
            # atoms_copy.calc = None

            # Write each system to its own file
            system_file = write_dir / f"{i}.xyz"
            write(system_file, atoms_copy, format="extxyz")

        # Calculate and save MAE if we have results
        # if complex_atoms:
        #     errors = [atoms.info["error_kcal"] for atoms in complex_atoms]
        #     mae = sum(abs(error) for error in errors) / len(errors)
        #     mae_data = {"MAE_kcal": float(mae)}

        #     with open(write_dir / "mae_results.json", "w") as f:
        #         json.dump(mae_data, f, indent=2)

        #     print(f"MAE for {self.model_name} on PLA15: {mae:.2f} kcal/mol")


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
            benchmark = PLA15Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_pla15():
    """Run PLA15 benchmark via pytest."""
    build_project(repro=True)
