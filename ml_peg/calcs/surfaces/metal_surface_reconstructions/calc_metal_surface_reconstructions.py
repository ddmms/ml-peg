"""Run calculations for metal surface reconstructions tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_lattice_energy(mlip: tuple[str, Any]) -> None:
    """
    Run Metal surface reconstructions lattice energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Download metal_surface_reconstructions dataset
    surface_configurations = (
        download_s3_data(
            key="inputs/surfaces/metal_surface_reconstructions/metal_surface_reconstructions.zip",
            filename="metal_surface_reconstructions.zip",
        )
        / "metal_surface_reconstructions"
    )

    with open(surface_configurations / "list") as f:
        systems = f.read().splitlines()

    for system in systems:
        slab = read(surface_configurations / f"{system}.xyz")
        slab.info["system"] = system
        slab.calc = calc

        if not (system.startswith("gas_phase") and system.startswith("bulk")):
            z_min = np.min(slab.positions[:, 2])
            c = FixAtoms(
                indices=[at.index for at in slab if at.position[2] < (z_min + 0.1)]
            )
            slab.set_constraint(c)

        if system.startswith("gas_phase"):
            opt = BFGS(slab)
            opt.run(fmax=0.01)
        elif not system.startswith("bulk"):
            opt = BFGS(slab)
            opt.run(fmax=0.05, steps=500)

        slab.get_potential_energy()

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{system}.xyz", slab)
