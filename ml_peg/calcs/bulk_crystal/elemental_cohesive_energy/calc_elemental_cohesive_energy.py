"""Run calculations for elemental cohesive energy benchmark."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.build import bulk
from ase.io import write
from janus_core.calculations.geom_opt import GeomOpt
import pytest
import yaml

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"
REF_PATH = Path(__file__).with_name("reference.yml")


@pytest.mark.parametrize("mlip", MODELS.items())
def test_elemental_cohesive_energy(mlip: tuple[str, Any]) -> None:
    """
    Run elemental cohesive energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    with open(REF_PATH, encoding="utf8") as file:
        reference = yaml.safe_load(file)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    for symbol, ref in reference.items():
        crystal = bulk(symbol, ref["structure"], a=ref.get("a"))
        # Isolated atom in a large periodic box
        atom = Atoms(symbol, cell=[30.0, 30.0, 30.0], pbc=True)
        for struct in (crystal, atom):
            struct.info.update({"charge": 0, "spin": 1, "name": symbol})
            struct.calc = copy(calc)

        try:
            # Relax the bulk cell, keeping it cubic
            geom_opt = GeomOpt(
                struct=crystal, fmax=0.01, filter_kwargs={"hydrostatic_strain": True}
            )
            geom_opt.run()
            crystal = geom_opt.struct
            crystal.get_potential_energy()
            atom.get_potential_energy()
        except Exception as exc:
            warn(f"Error calculating {symbol}: {exc}", stacklevel=2)
            continue

        write(write_dir / f"{symbol}.xyz", [crystal, atom])
