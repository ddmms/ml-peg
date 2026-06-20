"""
Compute single-point energies and forces for the graphene oxide dataset.

3813 graphene oxide structures (C/H/O) with varying oxidation coverage.
Config types encode O/C ratio, OH/O ratio, and optionally edge R/(R+H) ratio.
Reference energies from DFT/PBE. Formation energies are computed by subtracting
isolated C, H, and O atom energies (composition-weighted) from each structure,
separately for DFT and MLIP, to place both on the same relative scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DATA_PATH = Path(__file__).parent / "prepared_data" / "total.xyz"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_graphene_oxide(mlip: tuple[str, Any]) -> None:
    """
    Run single-point energy and force calculations for all graphene oxide structures.

    Parameters
    ----------
    mlip
        Model name and model object to use as calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    atoms_list = read(DATA_PATH, ":")

    isolated = [a for a in atoms_list if a.info.get("config_type") == "IsolatedAtom"]
    structures = [a for a in atoms_list if a.info.get("config_type") != "IsolatedAtom"]

    # Build per-element isolated-atom energy dicts for DFT and MLIP
    e0_dft: dict[str, float] = {}
    e0_mlip: dict[str, float] = {}
    for iso in isolated:
        sym = iso.get_chemical_symbols()[0]
        e0_dft[sym] = iso.info["QM_energy"]
        iso_copy = iso.copy()
        iso_copy.calc = calc
        e0_mlip[sym] = float(iso_copy.get_potential_energy())

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for atoms in tqdm(structures, desc=model_name):
        n = len(atoms)
        syms = atoms.get_chemical_symbols()
        ref_energy = atoms.info["QM_energy"]
        ref_forces = atoms.arrays["QM_forces"]
        atoms.calc = calc
        pred_energy = float(atoms.get_potential_energy())
        pred_forces = atoms.get_forces()
        atoms.calc = None

        e_offset_dft = sum(e0_dft[s] for s in syms)
        e_offset_mlip = sum(e0_mlip[s] for s in syms)

        atoms.info["ref_energy"] = ref_energy
        atoms.info["pred_energy"] = pred_energy
        atoms.info["ref_energy_rel"] = (ref_energy - e_offset_dft) / n
        atoms.info["pred_energy_rel"] = (pred_energy - e_offset_mlip) / n
        atoms.arrays["ref_forces"] = ref_forces
        atoms.arrays["pred_forces"] = pred_forces

        results.append(atoms)

    write(out_dir / "results.xyz", results)
