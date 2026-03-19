"""Calculate ethanol-water density curves."""

from __future__ import annotations

import logging
import os
from typing import Any

import pytest

from ml_peg.calcs.liquids.ethanol_water_density._compositions import (
    BENCH_ROOT,
    DATA_PATH,
    load_compositions,
)
from ml_peg.calcs.liquids.ethanol_water_density.md_code import run_one_case
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

OUT_PATH = BENCH_ROOT / "outputs"

MODELS = load_models(current_models)
MODEL_INDEX = {name: i for i, name in enumerate(MODELS)}
FAKE_DATA = os.getenv("FAKE_DENSITY_DATA", "") == "1"
CONTINUE_RUNNING = os.getenv("CONTINUE_RUNNING", "") == "1"

# IMPORTANT: create the list once for parametrization
COMPOSITIONS = load_compositions()


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items(), ids=list(MODELS.keys()))
def test_water_ethanol_density_curves(mlip: tuple[str, Any]) -> None:
    """
    Generate one density-curve case for a model and composition.

    Parameters
    ----------
    mlip : tuple[str, Any]
        Pair of model name and model object.

    Returns
    -------
    None
        This test writes output files for a single case.
    """
    for case in COMPOSITIONS:
        water_ethanol_density_curve_one_case(mlip, case)


def water_ethanol_density_curve_one_case(mlip: tuple[str, Any], case) -> None:
    """
    Run one MD simulation case and write its density time series.

    Parameters
    ----------
    mlip : tuple[str, Any]
        Pair of model name and model object.
    case : Any
        Composition case containing ``x_ethanol`` and ``filename``.

    Returns
    -------
    None
        This function writes outputs for one composition.
    """
    model_name, model = mlip

    model_out = OUT_PATH / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    # TODO: the data downloading thing here

    struct_path = DATA_PATH / case.filename
    if not struct_path.exists():
        raise FileNotFoundError(
            f"Missing structure for x={case.x_ethanol}: {struct_path}"
        )

    case_dir = model_out / f"x_ethanol_{case.x_ethanol:.2f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=case_dir / f"{model_name}.log",
        filemode="a",
        force=True,
    )
    run_one_case(struct_path, calc, case_dir / f"{model_name}.traj")


if __name__ == "__main__":  # TODO: delete this
    # run a very small simulation to see if it does something reasonable
    from ase import units
    from mace.calculators import mace_mp

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    calc = mace_mp(
        "data_old/mace-omat-0-small.model",
        dispersion=True,
        dispersion_cutoff=25 * units.Bohr,
    )
    run_one_case(
        "data/mix_xe_0.00.extxyz",
        calc,
        output_fname="debug/whatever.traj",
    )
