"""Shared utility functions for PLF547 and PLA15 benchmarks."""

from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import write
import numpy as np
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import download_s3_data

KCAL_TO_EV = units.kcal / units.mol


def extract_charge_and_selections(pdb_path: Path) -> tuple[float, float, float]:
    """
    Extract charge and selection information from PDB REMARK lines.

    Parameters
    ----------
    pdb_path : Path
        Path to PDB file.

    Returns
    -------
    Tuple[float, float, float]
        Total charge, charge A, charge B.
    """
    total_charge = qa = qb = 0.0

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

    return total_charge, qa, qb


def parse_references(path: Path) -> dict[str, float]:
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
    ref = {}

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
        # Format: "1ABC_15_lys" -> "1ABC_15"
        identifier_parts = full_identifier.split("_")
        if len(identifier_parts) >= 3:
            # Last part is residue type (lys, arg, asp, etc.)
            base_identifier = "_".join(identifier_parts[:-1])
        else:
            # Fallback: use full identifier if format is unexpected
            base_identifier = full_identifier

        energy_ev = energy_kcal * KCAL_TO_EV  # Convert to eV
        ref[base_identifier] = energy_ev

    return ref


def get_interaction_energy(fragments: dict[str, Atoms], calc: Calculator) -> float:
    """
    Calculate interaction energy from fragments.

    Parameters
    ----------
    fragments
        Dictionary containing 'complex', 'protein', and 'ligand' fragments.
    calc
        ASE calculator for energy calculations.

    Returns
    -------
    float
        Interaction energy in eV.
    """
    fragments["complex"].calc = calc
    e_complex = fragments["complex"].get_potential_energy()
    fragments["complex"].calc = None

    fragments["protein"].calc = calc
    e_protein = fragments["protein"].get_potential_energy()
    fragments["protein"].calc = None

    fragments["ligand"].calc = calc
    e_ligand = fragments["ligand"].get_potential_energy()
    fragments["ligand"].calc = None

    return e_complex - e_protein - e_ligand


def mda_atoms_to_ase(atom_list, charge: float, identifier: str) -> Atoms:
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
        atoms.info.update(
            {"charge": int(round(charge)), "spin": 1, "identifier": identifier}
        )
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
    atoms.info.update(
        {"charge": int(round(charge)), "spin": 1, "identifier": identifier}
    )
    return atoms.copy()


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


def process_pdb_file(pdb_path: Path) -> dict[str, Atoms]:
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
    total_charge, charge_a, charge_b = extract_charge_and_selections(pdb_path)

    try:
        all_atoms, protein_atoms, ligand_atoms = separate_protein_ligand_simple(
            pdb_path
        )

        if len(ligand_atoms) == 0:
            logging.warning(f"No ligand atoms found in {pdb_path.name}")
            return {}

        if len(protein_atoms) == 0:
            logging.warning(f"No protein atoms found in {pdb_path.name}")
            return {}

        base_id = pdb_path.stem

        # Create ASE objects
        complex_atoms = mda_atoms_to_ase(list(all_atoms), total_charge, base_id)
        protein_frag = mda_atoms_to_ase(protein_atoms, charge_a, base_id)
        ligand = mda_atoms_to_ase(ligand_atoms, charge_b, base_id)

        return {"complex": complex_atoms, "protein": protein_frag, "ligand": ligand}

    except Exception as e:
        logging.warning(f"Error processing {pdb_path}: {e}")
        return {}


def run_benchmark(benchmark: zntrack.Node, name: str, out_path: Path) -> None:
    """
    Run calculations for benchmark.

    Parameters
    ----------
    benchmark
        Zntrack benchmark class.
    name
        Name of benchmark.
    out_path
        Name of path to write outputs to.
    """
    data_dir = (
        download_s3_data(
            filename=f"{name}.zip", key=f"inputs/supramolecular/{name}/{name}.zip"
        )
        / name
    )

    benchmark.model.default_dtype = "float64"
    calc = benchmark.model.get_calculator()
    # Add D3 calculator for this test
    calc = benchmark.model.add_d3_calculator(calc)

    ref_energies = parse_references(data_dir / "reference_energies.txt")

    for label, ref_energy in tqdm(ref_energies.items()):
        pdb_fname = data_dir / f"{label}.pdb"

        fragments = process_pdb_file(pdb_fname)
        if not fragments:
            continue

        complex_atoms = fragments["complex"]
        complex_atoms.info["model_int_energy"] = get_interaction_energy(fragments, calc)
        complex_atoms.info["ref_int_energy"] = ref_energy
        complex_atoms.info["model"] = benchmark.model_name
        complex_atoms.info["identifier"] = label

        protein_charge = fragments["protein"].info["charge"]
        ligand_charge = fragments["ligand"].info["charge"]

        if protein_charge != 0 and ligand_charge != 0:
            interaction_type = "ion-ion"
        elif protein_charge != 0 or ligand_charge != 0:
            interaction_type = "ion-neutral"
        else:
            interaction_type = "neutral-neutral"

        complex_atoms.info["interaction_type"] = interaction_type

        write_dir = out_path / benchmark.model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        write(
            write_dir / f"{label}_complex.xyz",
            [complex_atoms, fragments["ligand"], fragments["protein"]],
        )
