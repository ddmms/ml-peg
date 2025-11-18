"""Run calculations for lattice constants benchmark."""

from __future__ import annotations

from copy import copy
import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import bulk
from janus_core.calculations.geom_opt import GeomOpt
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


def build_crystal(
    symbols: str,
    lattice_type: str,
    lattice_a: float | None = None,
    lattice_c: float | None = None,
    **kwargs,
) -> Atoms:
    """
    Build a bulk crystal.

    Parameters
    ----------
    symbols
        The chemical symbol or symbols to build.
    lattice_type
        Lattice type, e.g. "bcc", "fcc", "hcp".
    lattice_a
        Lattice constant a.
    lattice_c
        Lattice constant c.
    **kwargs
        Key word arguments to pass to ase.build.bulk.

    Returns
    -------
    Atoms
        Built bulk crystal.
    """
    if lattice_type == "2H-SiC":
        cell = [
            [lattice_a, 0, 0],
            [-lattice_a / 2, lattice_a * 3**0.5 / 2, 0],
            [0, 0, lattice_c],
        ]
        scaled_positions = [
            (1 / 3, 2 / 3, 0.0),
            (2 / 3, 1 / 3, 0.5),
            (1 / 3, 2 / 3, 0.25),
            (2 / 3, 1 / 3, 0.75),
        ]
        symbols = ["C", "C", "Si", "Si"]
        atoms = Atoms(
            symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True
        )
    else:
        atoms = bulk(symbols, lattice_type, a=lattice_a, c=lattice_c, **kwargs)

    atoms.info["lattice_type"] = lattice_type
    return atoms


@pytest.mark.parametrize("mlip", MODELS.items())
def test_lattice_consts(mlip: tuple[str, Any]) -> None:
    """
    Run lattice constant test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_dir = (
        download_s3_data(
            key="inputs/bulk_crystal/lattice_constants.zip",
            filename="lattice_constants.zip",
        )
        / "lattice_constants"
    )

    lattice_consts_exp_path = data_dir / "lattice_constants_ref_exp.json"
    lattice_consts_dft_path = data_dir / "lattice_constants_ref.json"
    lattice_types_path = data_dir / "lattice_constants_ref_types.json"

    with open(lattice_consts_exp_path) as file:
        lattice_consts_exp = json.load(file)
    with open(lattice_consts_dft_path) as file:
        lattice_consts_dft = json.load(file)
    with open(lattice_types_path) as file:
        lattice_types = json.load(file)

    crystals = {}
    for name, lattice_type in lattice_types.items():
        # Get lattice constants from data
        if name == "SiC":
            lattice_a_exp = lattice_consts_exp["SiC(a)"]
            lattice_c_exp = lattice_consts_exp["SiC(c)"]
            lattice_a_dft = lattice_consts_dft["SiC(a)"]
            lattice_c_dft = lattice_consts_dft["SiC(c)"]
        else:
            lattice_a_exp = lattice_consts_exp[name]
            lattice_a_dft = lattice_consts_dft[name]
            lattice_c_exp = None
            lattice_c_dft = None

        crystal = build_crystal(
            symbols=name,
            lattice_type=lattice_type,
            lattice_a=lattice_a_exp,
            lattice_c=lattice_c_exp,
        )

        # Save reference data to structure
        crystal.info["name"] = name
        crystal.info["a_exp"] = lattice_a_exp
        crystal.info["c_exp"] = lattice_c_exp
        crystal.info["a_dft"] = lattice_a_dft
        crystal.info["c_dft"] = lattice_c_dft
        crystals[name] = crystal

    for name, crystal in crystals.items():
        crystal.calc = copy(calc)
        GeomOpt(
            struct=crystal,
            fmax=0.03,
            write_traj=True,
            file_prefix=OUT_PATH / model_name / name,
        ).run()
