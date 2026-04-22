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

    # Separate isolated atoms (C, H, O) from structures
    ref_iso: dict[str, float] = {}
    pred_iso: dict[str, float] = {}
    structures = []
    for a in atoms_list[:10]:
        if a.info.get("config_type") == "IsolatedAtom":
            sym = a.get_chemical_symbols()[0]
            ref_iso[sym] = a.info["QM_energy"]
            a.calc = calc
            pred_iso[sym] = float(a.get_potential_energy())
            a.calc = None
        else:
            structures.append(a)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for atoms in tqdm(structures, desc=model_name):
        syms = atoms.get_chemical_symbols()
        n = len(atoms)
        n_c = syms.count("C")
        n_h = syms.count("H")
        n_o = syms.count("O")

        ref_energy = atoms.info["QM_energy"]
        atoms.calc = calc
        pred_energy = float(atoms.get_potential_energy())
        atoms.calc = None

        ref_energy_rel = (
            ref_energy - n_c * ref_iso["C"] - n_h * ref_iso["H"] - n_o * ref_iso["O"]
        ) / n
        pred_energy_rel = (
            pred_energy
            - n_c * pred_iso["C"]
            - n_h * pred_iso["H"]
            - n_o * pred_iso["O"]
        ) / n

        atoms.info["ref_energy"] = ref_energy
        atoms.info["pred_energy"] = pred_energy
        atoms.info["ref_energy_rel"] = ref_energy_rel
        atoms.info["pred_energy_rel"] = pred_energy_rel

        results.append(atoms)

    write(out_dir / "results.xyz", results)
