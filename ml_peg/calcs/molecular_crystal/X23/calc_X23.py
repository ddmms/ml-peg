"""Run calculations for X23 tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import get_benchmark_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


@pytest.mark.parametrize("mlip", MODELS.items())
def test_lattice_energy(mlip: tuple[str, Any]) -> None:
    """
    Run X23 lattice energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # download X23 dataset
    lattice_energy_dir = get_benchmark_data("lattice_energy.zip") / "lattice_energy"

    with open(lattice_energy_dir / "list") as f:
        systems = f.read().splitlines()

    for system in systems:
        mol_path = lattice_energy_dir / system / "POSCAR_molecule"
        sol_path = lattice_energy_dir / system / "POSCAR_solid"
        ref_path = lattice_energy_dir / system / "lattice_energy_DMC"
        nmol_path = lattice_energy_dir / system / "nmol"

        mol = read(mol_path, index=0, format="vasp")
        mol.calc = calc
        mol.get_potential_energy()

        sol = read(sol_path, index=0, format="vasp")
        sol.calc = copy(calc)
        sol.get_potential_energy()

        ref = np.loadtxt(ref_path)[0]
        nmol = np.loadtxt(nmol_path)

        sol.info["ref"] = ref
        sol.info["nmol"] = nmol
        sol.info["system"] = system
        mol.info["ref"] = ref
        mol.info["nmol"] = nmol
        mol.info["system"] = system

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{system}.xyz", [sol, mol])
