"""Run calculations for DMC-ICE13 benchmark."""

from __future__ import annotations

from copy import copy
import json
from pathlib import Path
from typing import Any

from ase.io import read, write
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
    Run DMC-ICE13 lattice energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # Download data
    data_dir = (
        download_s3_data(
            key="inputs/molecular_crystal/DMC_ICE13/DMC_ICE13.zip",
            filename="DMC_ICE13.zip",
        )
        / "dmc-ice13-main/INPUT/VASP"
    )

    with open(data_dir / "../../ice_polymorph_ref_PBE_D3.json", encoding="utf8") as f:
        ice_ref = json.load(f)

    polymorphs = [
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and path.name != "water"
    ]
    water = read(data_dir / "water/POSCAR", "0")
    water.calc = calc
    # Set default charge and spin
    water.info.setdefault("charge", 0)
    water.info.setdefault("spin", 1)
    water.get_potential_energy()

    for polymorph in polymorphs:
        polymorph_path = data_dir / polymorph / "POSCAR"
        struct = read(polymorph_path, "0")
        ref = ice_ref[polymorph]

        struct.calc = copy(calc)
        # Set default charge and spin
        struct.info.setdefault("charge", 0)
        struct.info.setdefault("spin", 1)
        struct.get_potential_energy()
        struct.info["ref"] = ref
        struct.info["polymorph"] = polymorph

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{polymorph}_polymorph.xyz", struct)

    write(write_dir / "water.xyz", water)
