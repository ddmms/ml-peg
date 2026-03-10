"""
Compute the Folmsbee dataset of molecular conformers.

Assessing conformer energies using electronic structure and
machine learning methods

Dakota Folmsbee, Geoffrey Hutchinson
International Journal of Quantum Chemistry 2020 121 (1) e26381
DOI: 10.1002/qua.26381
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import write
import pytest
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV

OUT_PATH = Path(__file__).parent / "outputs"


def get_relative_energies(energies: list[float], ref_idx: int) -> list[float]:
    """
    Get energies relative to reference.

    Parameters
    ----------
    energies
        List of energy values.
    ref_idx
        Index of reference energy.

    Returns
    -------
    list[float]
        Energies relative to the reference conformer.
    """
    return [x - energies[ref_idx] for x in energies]


@pytest.mark.parametrize("mlip", MODELS.items())
def test_folmsbee(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the Folmsbee dataset.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    # Use double precision
    model.default_dtype = "float64"
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    data_path = Path(__file__).parent / "data" / "folmsbee_dataset.json"
    out_path = OUT_PATH / model_name

    with open(data_path) as f:
        data = json.load(f)
    progress = tqdm(total=len(data))
    for structure_data in data:
        structure_name = structure_data["molecule_name"]
        conformers = []
        model_energies = []
        raw_energies = structure_data["dft_energy_profile"]
        ref_min_conformer_idx = raw_energies.index(min(raw_energies))
        ref_energies = get_relative_energies(raw_energies, ref_min_conformer_idx)

        for i, conf_positions in enumerate(structure_data["conformer_coordinates"]):
            conf_atoms = Atoms(
                positions=conf_positions, symbols=structure_data["atom_symbols"]
            )
            conf_atoms.calc = calc
            conf_atoms.info.update({"charge": 0, "spin": 1})
            conf_atoms.info["ref_rel_energy"] = ref_energies[i]

            conformers.append(conf_atoms)
            model_energies.append(conf_atoms.get_potential_energy())

        model_energies = get_relative_energies(model_energies, ref_min_conformer_idx)
        out_path.mkdir(parents=True, exist_ok=True)
        for i, conf_atoms in enumerate(conformers):
            conf_atoms.info["model_rel_energy"] = model_energies[i]
            write(out_path / f"{structure_name}_conf{i}.xyz", conf_atoms)

        progress.update()
