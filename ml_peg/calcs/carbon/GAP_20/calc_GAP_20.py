"""
Compute single-point energies for the GAP-20 carbon dataset.

Reference: Deringer et al., Carbon 227, 119510 (2024). 6088 carbon structures
covering sp2, sp3, amorphous/liquid, surfaces, and defects.
Reference energies from DFT/optB88vdW. Energies are made relative to the
isolated carbon atom energy.
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
DATA_PATH = Path(__file__).parent / "prepared_data" / "FPS_stress" / "total.xyz"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_gap_20(mlip: tuple[str, Any]) -> None:
    """
    Run single-point energy calculations for all GAP-20 carbon structures.

    Parameters
    ----------
    mlip
        Model name and model object to use as calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    atoms_list = read(DATA_PATH, ":")

    # Separate isolated atoms from structures
    structures = [a for a in atoms_list if a.info.get("config_type") != "IsolatedAtom"]

    # Subtract the first structure's energy per atom from all others so that
    # DFT and MLIP are compared on the same relative scale. Use a copy for the
    # MLIP call so the original's SinglePointCalculator stays intact for the loop.
    _ref = structures[0].copy()
    _ref.calc = calc
    ref_dft_per_atom = structures[0].get_potential_energy() / len(structures[0])
    ref_mlip_per_atom = _ref.get_potential_energy() / len(_ref)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for atoms in tqdm(structures, desc=model_name):
        n = len(atoms)
        ref_energy = atoms.get_potential_energy()

        atoms.calc = calc
        pred_energy = float(atoms.get_potential_energy())
        atoms.calc = None

        atoms.info["ref_energy"] = ref_energy
        atoms.info["pred_energy"] = pred_energy
        atoms.info["ref_energy_rel"] = ref_energy / n - ref_dft_per_atom
        atoms.info["pred_energy_rel"] = pred_energy / n - ref_mlip_per_atom

        results.append(atoms)

    write(out_dir / "results.xyz", results)
