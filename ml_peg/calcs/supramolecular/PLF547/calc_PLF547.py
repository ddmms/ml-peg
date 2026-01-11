"""Calculate PLF547 benchmark. 10.1021/acs.jcim.9b01171."""

from __future__ import annotations

import logging
from pathlib import Path

import ase
from ase import units
from ase.io import write
import mlipx
from mlipx.abc import NodeWithCalculator
import numpy as np
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import chdir, download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"


class PLF547Benchmark(zntrack.Node):
    """Benchmarking PLF547 subset ionic hydrogen bonds benchmark dataset."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    # ------------------------------------------------------------
    # PDB processing functions
    # ------------------------------------------------------------
    @staticmethod
    def extract_charge_and_selections(pdb_path: Path) -> tuple[int, int, int]:
        """
        Extract charge information from PDB REMARK lines.

        Parameters
        ----------
        pdb_path
            Path to pdb files.

        Returns
        -------
        tuple[int, int, int]
            A tuple containing:

            - total_charge: Total charge of the system.
            - qa: Charge of selection a.
            - qb: Charge of selection b.
        """
        total_charge = qa = qb = 0

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
                    total_charge = int(parts[2])
                elif tag == "charge_a":
                    qa = int(parts[2])
                elif tag == "charge_b":
                    qb = int(parts[2])

        return total_charge, qa, qb

    @staticmethod
    def separate_protein_ligand_simple(pdb_path: Path):
        """
        Separate based on residue names.

        Parameters
        ----------
        pdb_path
            Path to pdb files.

        Returns
        -------
        tuple[Mda.universe.atoms, Mda.universe.atoms, Mda.universe.atoms]:

            - u.atoms: MDA atoms of the full system.
            - protein_atoms: MDA atoms of the protein.
            - ligand_atoms: MDA atoms of the ligand.
        """
        import MDAnalysis as Mda

        # Load with MDAnalysis
        u = Mda.Universe(str(pdb_path))

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
    def mda_atoms_to_ase(atom_list, charge: float, identifier: str) -> ase.Atoms:
        """
        Convert MDAnalysis atoms to ASE Atoms object.

        Parameters
        ----------
        atom_list
            List of MDA atoms.
        charge
            Charge of the system.
        identifier
            Identifier of the system.

        Returns
        -------
        ASE.Atoms
            ASE atoms object of the system.
        """
        from ase import Atoms

        if not atom_list:
            atoms = Atoms()
            atoms.info.update({"charge": charge, "spin": 1, "identifier": identifier})
            return atoms

        symbols = []
        positions = []

        for atom in atom_list:
            # Get element symbol
            try:
                elem = (atom.element or "").strip().title()
            except Exception:
                elem = ""

            if not elem:
                # Fallback: first letter of atom name
                elem = "".join([c for c in atom.name if c.isalpha()])[:1].title() or "C"

            symbols.append(elem)
            positions.append(atom.position)

        atoms = Atoms(symbols=symbols, positions=np.array(positions))
        atoms.info.update({"charge": charge, "spin": 1, "identifier": identifier})
        return atoms.copy()

    def process_pdb_file(self, pdb_path: Path) -> dict[str, ase.Atoms]:
        """
        Return complex + separated fragments from PDB file.

        Parameters
        ----------
        pdb_path
            Path to pdb files.

        Returns
        -------
        dict[str, ASE.Atoms]
            Dictionary containing the ASE atoms objects
            of the complex, protein, and ligand.
        """
        total_charge, charge_a, charge_b = self.extract_charge_and_selections(pdb_path)

        try:
            all_atoms, protein_atoms, ligand_atoms = (
                self.separate_protein_ligand_simple(pdb_path)
            )

            if len(ligand_atoms) == 0:
                logging.warning(f"No ligand atoms found in {pdb_path.name}")
                return {}

            if len(protein_atoms) == 0:
                logging.warning(f"No protein atoms found in {pdb_path.name}")
                return {}

            base_id = pdb_path.stem

            # Create ASE objects
            complex_atoms = self.mda_atoms_to_ase(
                list(all_atoms), total_charge, base_id
            )
            protein_frag = self.mda_atoms_to_ase(protein_atoms, charge_a, base_id)
            ligand = self.mda_atoms_to_ase(ligand_atoms, charge_b, base_id)

            return {"complex": complex_atoms, "protein": protein_frag, "ligand": ligand}

        except Exception as e:
            logging.warning(f"Error processing {pdb_path}: {e}")
            return {}

    # ------------------------------------------------------------
    # Reference energy parsing
    # ------------------------------------------------------------
    @staticmethod
    def parse_plf547_references(path: Path) -> dict[str, float]:
        """
        Parse PLF547 reference interaction energies (kcal/mol -> eV).

        Parameters
        ----------
        path
            Path to PLF547 references.

        Returns
        -------
        dict[str, float]
            Dictionary containing the reference interaction energies.
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

            # Extract base identifier by removing residue type suffix
            # Format: "2P4Y_28_met" -> "2P4Y_28"
            identifier_parts = full_identifier.split("_")
            if len(identifier_parts) >= 3:
                # Assume last part is residue type (met, arg, bbn, etc.)
                base_identifier = "_".join(identifier_parts[:-1])
            else:
                # Fallback: use full identifier if format is unexpected
                base_identifier = full_identifier

            energy_ev = energy_kcal * KCAL_TO_EV  # Convert to eV
            ref[base_identifier] = energy_ev

        return ref

    def run(self):
        """Run new benchmark."""
        # Read in data and attach calculator
        data_dir = (
            download_s3_data(
                filename="PLF547.zip", key="inputs/supramolecular/PLF547/PLF547.zip"
            )
            / "PLF547"
        )

        ref_energies = self.parse_plf547_references(data_dir / "reference_energies.txt")

        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        for label, ref_energy in tqdm(ref_energies.items()):
            pdb_fname = data_dir / f"{label}.pdb"

            fragments = self.process_pdb_file(pdb_fname)
            if not fragments:
                continue

            complex_atoms = fragments["complex"]
            complex_atoms.calc = calc
            complex_energy = complex_atoms.get_potential_energy()

            protein_atoms = fragments["protein"]
            protein_atoms.calc = calc
            protein_energy = protein_atoms.get_potential_energy()

            ligand_atoms = fragments["ligand"]
            ligand_atoms.calc = calc
            ligand_energy = ligand_atoms.get_potential_energy()

            complex_atoms.info["model_int_energy"] = (
                complex_energy - protein_energy - ligand_energy
            )
            complex_atoms.info["ref_int_energy"] = ref_energy

            complex_atoms.calc = None
            ligand_atoms.calc = None
            protein_atoms.calc = None

            write_dir = OUT_PATH / self.model_name
            write_dir.mkdir(parents=True, exist_ok=True)

            write(
                write_dir / f"{label}_complex.xyz",
                [complex_atoms, ligand_atoms, protein_atoms],
            )


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
            benchmark = PLF547Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_plf547():
    """Run PLF547 conformation energies benchmark via pytest."""
    build_project(repro=True)
