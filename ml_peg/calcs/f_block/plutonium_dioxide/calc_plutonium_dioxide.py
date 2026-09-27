"""Calculate plutonium dioxide metrics."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_plutonium_dioxide(mlip: tuple[str, Any]) -> None:
    """
    Run single point calculations for plutonium dioxide structures.

    Parameters
    ----------
    mlip : tuple[str, Any]
        A tuple of (model_name, model_object).
    """
    model_name, model = mlip

    # Download data.
    puo2_data_dir = (
        download_s3_data(
            key="inputs/f_block/plutonium_dioxide/plutonium_dioxide.zip",
            filename="plutonium_dioxide.zip",
        )
        / "plutonium_dioxide"
    )

    ref_file = puo2_data_dir / "dft_ref_data.xyz"
    ref_structures = read(ref_file, ":")

    calculator = model.get_calculator()

    for atoms in tqdm(ref_structures, desc="Evaluating energy, forces and stress"):
        atoms.calc = copy(calculator)
        # Set default charge and spin
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)

        try:
            atoms.get_potential_energy()
        except Exception as exc:
            warn(f"Error calculating energy: {exc}", stacklevel=2)
            atoms.calc.results["energy"] = np.nan

        try:
            atoms.get_forces()
        except Exception as exc:
            warn(f"Error calculating forces: {exc}", stacklevel=2)
            n_atoms = len(atoms)
            atoms.calc.results["forces"] = np.full((n_atoms, 3), np.nan)

        try:
            atoms.get_stress()
        except Exception as exc:
            warn(f"Error calculating stress: {exc}", stacklevel=2)
            atoms.calc.results["stress"] = np.full(6, np.nan)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / "puo2_results.xyz", ref_structures, format="extxyz")
