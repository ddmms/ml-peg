"""
Calculate the CRBH20 reaction barrier dataset.

Reference barriers from Appendix B.5 of arXiv:2401.00096.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import units
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

EV_TO_KCAL = units.mol / units.kcal


@pytest.mark.parametrize("mlip", MODELS.items())
def test_crbh20_barrier_calculation(mlip: tuple[str, Any]) -> None:
    """
    Run calculations of the reaction energy barriers for the 20 systems in CRBH20.

    This function will be run automatically for every model in models.yml.

    Parameters
    ----------
    mlip
        Tuple containing (model_name, model_object).
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    # Add D3 calculator for this test (for models where applicable)
    calc = model.add_d3_calculator(calc)

    data_dir = (
        download_s3_data(
            key="inputs/molecular/CRBH20/CRBH20.zip",
            filename="CRBH20.zip",
        )
        / "CRBH20"
    )

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(1, 21)):
        rxn_id = str(i)
        rxn_path = data_dir / rxn_id

        energies = {}
        atoms_dict = {}

        # Calculate energy for reactant and transition state
        for state in ("react", "ts"):
            poscar_path = rxn_path / state / "POSCAR"

            atoms = read(poscar_path)
            atoms.calc = calc

            atoms.info.setdefault("charge", 0)
            atoms.info.setdefault("spin", 1)

            try:
                e_pot = atoms.get_potential_energy()
            except Exception as exc:
                warn(
                    f"Error calculating energy for {poscar_path}: {exc}",
                    stacklevel=2,
                )
                e_pot = np.nan

            energies[state] = e_pot
            atoms_dict[state] = atoms

            atoms.info["rxn_id"] = rxn_id
            atoms.info["state"] = state
            atoms.info["energy_ev"] = e_pot
            atoms.info["model"] = model_name

        # Compute barrier and tag both structures with it
        barrier_ev = energies["ts"] - energies["react"]
        barrier_kcal = barrier_ev * EV_TO_KCAL

        for state in ("react", "ts"):
            atoms_dict[state].info["barrier_ev"] = barrier_ev
            atoms_dict[state].info["barrier_kcal"] = barrier_kcal

        # Write combined XYZ file (reactant + TS) for this reaction
        write(
            write_dir / f"crbh20_{rxn_id}.xyz",
            [atoms_dict["react"], atoms_dict["ts"]],
        )
