"""Run calculations for SBH17 tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

@pytest.mark.parametrize("mlip", MODELS.items())
def test_surface_barrier(mlip: tuple[str, Any]) -> None:
    """
    Run SBH17 dissociative chemisorption barrier test.

    Note the data directory currently excludes one data point
    from the paper, H2Cu111, because the PBE value
    is not available.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    # Do not want D3 as references here are dispersionless PBE
    calc = model.get_calculator()

    # Download SBH17 dataset
    SBH17_dir = (
        download_s3_data(
            key="inputs/surfaces/SBH17/SBH17.zip",
            filename="SBH17.zip",
        )
        / "SBH17"
    )

    with open(SBH17_dir / "list") as f:
        systems = f.read().splitlines()

    for system in tqdm(systems, desc="Evaluating models on SBH17 structures"):
        gp_path = SBH17_dir / system / "POSCAR-gp"
        ts_path = SBH17_dir / system / "POSCAR-ts"
        ref_path = SBH17_dir / system / "barrier_pbe"

        gp = read(gp_path, index=0, format="vasp")
        gp.calc = calc
        gp.get_potential_energy()

        ts = read(ts_path, index=0, format="vasp")
        ts.calc = copy(calc)
        ts.get_potential_energy()

        ref = np.loadtxt(ref_path)

        gp.info["ref"] = ref
        gp.info["system"] = system
        ts.info["ref"] = ref
        ts.info["system"] = system

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{system}.xyz", [gp, ts])
