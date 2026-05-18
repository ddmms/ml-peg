"""
Compute single-point energies for the graphene oxide dataset.

3813 graphene oxide structures (C/H/O) with varying oxidation coverage.
Config types encode O/C ratio, OH/O ratio, and optionally edge R/(R+H) ratio.
Reference energies from DFT/PBE. Energies are made relative to isolated
C, H, and O atom energies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
DATA_PATH = Path(__file__).parent / "prepared_data" / "total.xyz"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_graphene_oxide(mlip: tuple[str, Any]) -> None:
    """
    Run single-point energy calculations for all graphene oxide structures.

    Parameters
    ----------
    mlip
        Model name and model object to use as calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    atoms_list = read(DATA_PATH, ":")

    # Separate isolated atoms from structures
    structures = [a for a in atoms_list if a.info.get("config_type") != "IsolatedAtom"]

    # Subtract the first structure's energy per atom from all others so that
    # DFT and MLIP are compared on the same relative scale.
    ref_atoms = structures[0]
    ref_dft_per_atom = ref_atoms.info["QM_energy"] / len(ref_atoms)
    ref_atoms.calc = calc
    ref_mlip_per_atom = float(ref_atoms.get_potential_energy()) / len(ref_atoms)
    ref_atoms.calc = None

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for atoms in tqdm(structures, desc=model_name):
        n = len(atoms)
        ref_energy = atoms.info["QM_energy"]
        atoms.calc = calc
        pred_energy = float(atoms.get_potential_energy())
        atoms.calc = None

        atoms.info["ref_energy"] = ref_energy
        atoms.info["pred_energy"] = pred_energy
        atoms.info["ref_energy_rel"] = ref_energy / n - ref_dft_per_atom
        atoms.info["pred_energy_rel"] = pred_energy / n - ref_mlip_per_atom

        results.append(atoms)

    write(out_dir / "results.xyz", results)
