"""Run calculations for Defectstab benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


def get_ref_energy(poscar_path: Path) -> float:
    """
    Extract reference energy from POSCAR/structure file header.

    Parameters
    ----------
    poscar_path
        Path to the POSCAR file.

    Returns
    -------
    float
        Reference energy parsed from the file header.

    Raises
    ------
    ValueError
        If the reference energy cannot be parsed from the file header.
    """
    with open(poscar_path) as f:
        line = f.readline().strip()
        # Handle 'E0= -1.23' format from rewrite_poscar.py
        if line.startswith("E0="):
            return float(line.split("=")[1].strip())
        # Handle legacy format '... E ...' (tokens[4])
        tokens = line.split()
        if len(tokens) >= 5:
            try:
                return float(tokens[4])
            except ValueError:
                pass
    raise ValueError(
        f"Could not parse reference energy from header of {poscar_path}: '{line}'"
    )


@pytest.mark.parametrize("mlip", MODELS.items())
def test_defectstab(mlip: tuple[str, Any]) -> None:
    """
    Run Defectstab test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = download_s3_data(
        key="inputs/defect/Defectstab/Defectstab.zip",
        filename="Defectstab.zip",
    )
    formation_energy_dir = data_path / "Defectstab"

    # Define subsets and their processing logic
    subsets = {
        "fe_sia": list(formation_energy_dir.glob("fe_sia/*.poscar")),
        "mapi_tetragonal": list(formation_energy_dir.glob("mapi_tetragonal/*.poscar")),
        "boroncarbide_stoichiometry": list(
            formation_energy_dir.glob("boroncarbide_stoichiometry/*.poscar")
        ),
        "boroncarbide_defects": list(
            formation_energy_dir.glob("boroncarbide_defects/*.poscar")
        ),
    }

    # Output directory
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # 1. FE_SIA (Legacy)
    # Ref: ref.poscar in fe_sia folder
    fe_sia_files = subsets["fe_sia"]
    ref_file = formation_energy_dir / "fe_sia" / "ref.poscar"

    e_ref_bulk_dft = 0.0
    if ref_file.exists():
        e_ref_bulk_dft = get_ref_energy(ref_file)
        atoms_ref = read(ref_file)
        # Set default charge and spin
        atoms_ref.info.setdefault("charge", 0)
        atoms_ref.info.setdefault("spin", 1)
        # Assuming existing logic for fe_sia where ref.poscar header has E_bulk
        # And we need to calculate E_bulk_pred
        atoms_ref.calc = calc
        e_ref_bulk_pred = atoms_ref.get_potential_energy()
        n_bulk = len(atoms_ref)

        # Save ref calculation
        atoms_ref.info["energy_dft"] = e_ref_bulk_dft
        atoms_ref.info["energy_pred"] = e_ref_bulk_pred
        # Store system name for analysis lookup
        atoms_ref.info["system"] = "fe_sia_ref"
        write(write_dir / "fe_sia_ref.xyz", atoms_ref)

        for poscar_path in fe_sia_files:
            if poscar_path.name == "ref.poscar":
                continue

            atoms = read(poscar_path)
            # Set default charge and spin
            atoms.info.setdefault("charge", 0)
            atoms.info.setdefault("spin", 1)
            e_dft = get_ref_energy(poscar_path)

            atoms.calc = calc
            e_pred = atoms.get_potential_energy()

            atoms.info["energy_dft"] = e_dft
            atoms.info["energy_pred"] = e_pred

            # Formation energy: E_f = E_config - (N_config / N_bulk) * E_bulk
            n_config = len(atoms)
            ref_fe = e_dft - (n_config / n_bulk) * e_ref_bulk_dft
            atoms.info["ref"] = ref_fe

            # Store bulk reference for predicted FE calculation in analysis
            atoms.info["e_bulk_pred"] = e_ref_bulk_pred
            atoms.info["n_bulk"] = n_bulk

            # Store subset info
            atoms.info["subset"] = "fe_sia"
            atoms.info["system"] = poscar_path.stem

            write(write_dir / f"fe_sia_{poscar_path.stem}.xyz", atoms)

    # 2. Boron Carbide (Stoichiometry & Defects)
    # Both need RefBC calculations: Graphite and AlphaBoron
    ref_bc_files = {}

    # Find graphite
    graphite_files = list(
        formation_energy_dir.glob("**/boroncarbide_refbc_graphiteLDA.poscar")
    )
    if graphite_files:
        ref_bc_files["C"] = graphite_files[0]

    # Find alpha Boron
    alpha_files = list(
        formation_energy_dir.glob("**/boroncarbide_refbc_alphaB_LDA.poscar")
    )
    if alpha_files:
        ref_bc_files["B"] = alpha_files[0]

    ref_energies = {}  # DFT per atom energies

    # Pre-calculate DFT reference energies for Boron (B) and Carbon (C)
    if "C" in ref_bc_files and "B" in ref_bc_files:
        for el, p_path in ref_bc_files.items():
            atoms = read(p_path)
            # Set default charge and spin
            atoms.info.setdefault("charge", 0)
            atoms.info.setdefault("spin", 1)
            e_dft = get_ref_energy(p_path)
            atoms.calc = calc
            e_pred = atoms.get_potential_energy()  # Just to verify/store if needed

            n_atoms = len(atoms)
            ref_energies[f"E_{el}_dft"] = e_dft / n_atoms
            ref_energies[f"E_{el}_pred"] = e_pred / n_atoms

            atoms.info["subset"] = "boroncarbide_ref"
            atoms.info["energy_dft"] = e_dft
            atoms.info["energy_pred"] = e_pred
            write(write_dir / f"boroncarbide_ref_{el}.xyz", atoms)

    # Pre-calculate NoDefects energy for defect formation
    e_no_defect_dft = 0.0
    e_no_defect_pred = None
    no_defect_file = None
    for f in subsets["boroncarbide_defects"]:
        if "NoDefects" in f.name:
            no_defect_file = f
            e_no_defect_dft = get_ref_energy(f)
            # Also compute predicted energy for NoDefects
            atoms_nd = read(f)
            # Set default charge and spin
            atoms_nd.info.setdefault("charge", 0)
            atoms_nd.info.setdefault("spin", 1)
            atoms_nd.calc = calc
            e_no_defect_pred = atoms_nd.get_potential_energy()
            break

    # Process Boron Carbide Subsets
    for subset in ["boroncarbide_stoichiometry", "boroncarbide_defects"]:
        files = subsets[subset]
        for poscar_path in files:
            if "refbc" in poscar_path.name:
                continue  # Skip ref files

            atoms = read(poscar_path)
            # Set default charge and spin
            atoms.info.setdefault("charge", 0)
            atoms.info.setdefault("spin", 1)
            e_dft = get_ref_energy(poscar_path)

            atoms.calc = calc
            e_pred = atoms.get_potential_energy()

            atoms.info["energy_dft"] = e_dft
            atoms.info["energy_pred"] = e_pred
            atoms.info["subset"] = subset
            atoms.info["system"] = poscar_path.stem

            # --- Stoichiometry Ref Logic ---
            if subset == "boroncarbide_stoichiometry" and ref_energies:
                syms = atoms.get_chemical_symbols()
                n_b = syms.count("B")
                n_c = syms.count("C")
                ref_val = (
                    e_dft
                    - n_b * ref_energies["E_B_dft"]
                    - n_c * ref_energies["E_C_dft"]
                )
                atoms.info["ref"] = ref_val

            # --- Defects Ref Logic ---
            if subset == "boroncarbide_defects" and no_defect_file and ref_energies:
                # Bipolar: E - E0
                if "Bipolar" in poscar_path.stem:
                    atoms.info["ref"] = e_dft - e_no_defect_dft
                # Vacancy (Brich): E - E0 + muB (muB = E_B_bulk)
                elif "VB0" in poscar_path.stem:
                    atoms.info["ref"] = (
                        e_dft - e_no_defect_dft + ref_energies["E_B_dft"]
                    )

            # Store component reference energies for analysis
            if ref_energies:
                atoms.info["E_B_dft"] = ref_energies["E_B_dft"]
                atoms.info["E_C_dft"] = ref_energies["E_C_dft"]
                atoms.info["E_B_pred"] = ref_energies["E_B_pred"]
                atoms.info["E_C_pred"] = ref_energies["E_C_pred"]
            if subset == "boroncarbide_defects" and e_no_defect_pred is not None:
                atoms.info["e_nodefect_dft"] = e_no_defect_dft
                atoms.info["e_nodefect_pred"] = e_no_defect_pred

            write(write_dir / f"{subset}_{poscar_path.stem}.xyz", atoms)

    # 3. MAPI Tetragonal
    mapi_files = subsets["mapi_tetragonal"]

    # Pre-fetch dft energies and compute predicted energies for MAI and pristine
    e_mai_dft = 0.0
    e_pris_dft = 0.0
    e_mai_pred = 0.0
    e_pris_pred = 0.0

    for p in mapi_files:
        if p.name.endswith("_MAI.poscar") or p.name == "MAI.poscar":
            e_mai_dft = get_ref_energy(p)
            atoms_mai = read(p)
            # Set default charge and spin
            atoms_mai.info.setdefault("charge", 0)
            atoms_mai.info.setdefault("spin", 1)
            atoms_mai.calc = calc
            e_mai_pred = atoms_mai.get_potential_energy()
        elif "pristine" in p.name:
            e_pris_dft = get_ref_energy(p)
            atoms_pris = read(p)
            # Set default charge and spin
            atoms_pris.info.setdefault("charge", 0)
            atoms_pris.info.setdefault("spin", 1)
            atoms_pris.calc = calc
            e_pris_pred = atoms_pris.get_potential_energy()

    for poscar_path in mapi_files:
        atoms = read(poscar_path)
        # Set default charge and spin
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)
        e_dft = get_ref_energy(poscar_path)

        atoms.calc = calc
        e_pred = atoms.get_potential_energy()

        atoms.info["energy_dft"] = e_dft
        atoms.info["energy_pred"] = e_pred
        atoms.info["subset"] = "mapi_tetragonal"
        atoms.info["system"] = poscar_path.stem

        # Calculate Ref for VMAI
        # Ref = E(VMAI) + 0.5*E(MAI) - E(pristine)
        if "VMAI" in poscar_path.stem and e_mai_dft and e_pris_dft:
            atoms.info["ref"] = e_dft + 0.5 * e_mai_dft - e_pris_dft
            atoms.info["e_mai_pred"] = e_mai_pred
            atoms.info["e_pris_pred"] = e_pris_pred
            atoms.info["system"] = "MAPI_Tetragonal_VMAI_Formation"

        write(write_dir / f"mapi_tetragonal_{poscar_path.stem}.xyz", atoms)
