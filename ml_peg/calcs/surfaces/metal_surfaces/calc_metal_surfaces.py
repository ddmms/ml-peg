
"""Run calculations for metal surface tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms

import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
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

    # Add D3 calculator for this test (for models where applicable)
    # calc = model.add_d3_calculator(calc)

    # Download X23 dataset
    #lattice_energy_dir = (
    #    download_s3_data(
    #        key="inputs/molecular_crystal/X23/X23.zip",
    #        filename="lattice_energy.zip",
    #    )
    #    / "lattice_energy"
    #)
    surface_configurations = Path(__file__).parent / "surface_configurations"



    with open(surface_configurations / "list") as f:
        systems = f.read().splitlines()

    for system in systems:
        slab = read(surface_configurations / f"{system}.xyz")
        slab.info["system"] = system
        z_min = np.min(slab.positions[:,2])
        c = FixAtoms(indices=[at.index for at in slab if at.position[2] < (z_min+0.1)])
        slab.set_constraint(c)

        slab.calc = calc
        opt = BFGS(slab)
        opt.run(fmax=0.03)
        slab.get_potential_energy()

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{system}.xyz", slab)
