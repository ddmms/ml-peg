"""Run calculations for extensivity tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import bulk, surface
from ase.io import write
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


def make_slab(
    symbol: str, layers: int, vacuum_z: float, size_xy: tuple[int, int]
) -> Atoms:
    """
    Prepare slab ASE structure.

    Parameters
    ----------
    symbol
        Element to build bulk surface for.
    layers
        Number of layers of slab to build.
    vacuum_z
        Amount of vaccum to add on both sides of slab, in Å.
    size_xy
        Number of times to repeated slab in the x and y directions.

    Returns
    -------
    Atoms
        Prepared slab structure.
    """
    slab = surface(bulk(symbol, "fcc"), (1, 1, 1), layers, vacuum=vacuum_z)
    slab = slab.repeat((size_xy[0], size_xy[1], 1))
    slab.center(axis=2, vacuum=vacuum_z)
    slab.pbc = True
    slab.info["charge"] = 0
    slab.info["spin"] = 1
    return slab


@pytest.mark.parametrize("mlip", MODELS.items())
def test_extensivity(mlip: tuple[str, Any]) -> None:
    """
    Run extensivity test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    # Use double precision
    model.default_dtype = "float64"
    calc = model.get_calculator()

    sym1, sym2 = "Al", "Ni"  # element of slab-1 and slab-2
    layers = 8
    vacuum_z = 100.0  # Å vacuum on isolated slab
    size_xy = (4, 4)
    gap = 100.0  # Å gap between slabs in combined cell

    # Prepare slabs
    slab1 = make_slab(sym1, layers, vacuum_z, size_xy)
    slab2 = make_slab(sym2, layers, vacuum_z, size_xy)
    slab2.translate([0, 0, gap])

    combined = slab1 + slab2
    tall_cell = slab1.cell.copy()
    tall_cell[2, 2] += gap
    combined.set_cell(tall_cell)
    combined.pbc = True

    # Put isolated slabs in the same tall cell
    slab1_big = slab1.copy()
    slab1_big.set_cell(tall_cell, scale_atoms=False)
    slab2_big = slab2.copy()
    slab2_big.set_cell(tall_cell, scale_atoms=False)

    slab1_big.calc = calc
    slab1_big.get_potential_energy()

    slab2_big.calc = copy(calc)
    slab2_big.get_potential_energy()

    combined.calc = copy(calc)
    combined.get_potential_energy()

    # Write output structures
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / "slabs.xyz", [slab1_big, slab2_big, combined])
