"""Run calculations for SBH17 tests."""

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
    calc = model.get_calculator(precision="high")

    # Download SBH17 dataset
    sbh17_dir = (
        download_s3_data(
            key="inputs/surfaces/SBH17/SBH17.zip",
            filename="SBH17.zip",
        )
        / "SBH17"
    )

    with open(sbh17_dir / "list") as f:
        systems = f.read().splitlines()

    for system in tqdm(systems, desc="Evaluating models on SBH17 structures"):
        gp_path = sbh17_dir / system / "gp.xyz"
        ts_path = sbh17_dir / system / "ts.xyz"
        ref_path = sbh17_dir / system / "barrier_pbe"

        gp = read(gp_path, index=0)
        gp.info.setdefault("charge", 0)
        gp.info.setdefault("spin", 1)
        gp.info["spin"] = int(round(gp.info["spin"]))
        gp.calc = calc
        try:
            gp.get_potential_energy()
        except Exception as exc:
            warn(f"Error calculating energy for {system}: {exc}", stacklevel=2)
            gp.info["energy"] = np.nan

        ts = read(ts_path, index=0)
        ts.info.setdefault("charge", 0)
        ts.info.setdefault("spin", 1)
        ts.info["spin"] = int(round(ts.info["spin"]))
        ts.calc = copy(calc)
        try:
            ts.get_potential_energy()
        except Exception as exc:
            warn(f"Error calculating energy for {system}: {exc}", stacklevel=2)
            ts.info["energy"] = np.nan

        ref = np.loadtxt(ref_path)

        gp.info["ref"] = ref
        gp.info["system"] = system
        ts.info["ref"] = ref
        ts.info["system"] = system

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{system}.xyz", [gp, ts])
