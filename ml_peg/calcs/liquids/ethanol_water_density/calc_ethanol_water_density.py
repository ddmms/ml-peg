"""Calculate ethanol-water density curves."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_peg.calcs.liquids.ethanol_water_density._compositions import (
    BENCH_ROOT,
    DATA_PATH,
    load_compositions,
)
from ml_peg.calcs.liquids.ethanol_water_density._fake_data import (
    make_fake_curve,
    make_fake_density_timeseries,
)
from ml_peg.calcs.liquids.ethanol_water_density._io_tools import (
    write_density_timeseries_checkpointed,
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


def add_shorter_d3_calculator(model, calcs):
    """
    Add D3 dispersion to calculator(s).

    Parameters
    ----------
    model
        Model to add the dispersion.
    calcs
        Calculator, or list of calculators, to add D3 dispersion to via a
        SumCalculator.

    Returns
    -------
    SumCalculator | Calculator
        Calculator(s) with D3 dispersion added, or the original calculator when
        the model is already trained with D3 corrections.
    """
    if model.trained_on_d3:
        return calcs
    from ase import units
    from ase.calculators.mixing import SumCalculator
    import torch
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    if not isinstance(calcs, list):
        calcs = [calcs]

    d3_calc = TorchDFTD3Calculator(
        device=model.d3_kwargs.get("device", "cpu"),
        damping=model.d3_kwargs.get("damping", "bj"),
        xc=model.d3_kwargs.get("xc", "pbe"),
        dtype=getattr(torch, model.d3_kwargs.get("dtype", "float32")),
        cutoff=model.d3_kwargs.get(
            "cutoff", 25.0 * units.Bohr
        ),  # shortened to make run more manageable.
    )
    calcs.append(d3_calc)

    return SumCalculator(calcs)


def _case_id(composition) -> str:
    """
    Build a readable test identifier for a composition case.

    Parameters
    ----------
    composition : Any
        Composition object with an ``x_ethanol`` attribute.

    Returns
    -------
    str
        Case identifier shown in pytest output.
    """
    # nicer test ids in `pytest -vv`
    return f"x={composition.x_ethanol:.2f}"


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items(), ids=list(MODELS.keys()))
@pytest.mark.parametrize(
    "composition", COMPOSITIONS, ids=[_case_id(c) for c in COMPOSITIONS]
)
def test_water_ethanol_density_curve(mlip: tuple[str, Any], composition) -> None:
    """
    Generate one density-curve case for a model and composition.

    Parameters
    ----------
    mlip : tuple[str, Any]
        Pair of model name and model object.
    composition : Any
        Composition case input.

    Returns
    -------
    None
        This test writes output files for a single case.
    """
    if not FAKE_DATA:
        water_ethanol_density_curve_one_case(mlip, composition)
    else:
        water_ethanol_density_dummy_data_one_case(mlip, composition)


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

    struct_path = DATA_PATH / case.filename
    if not struct_path.exists():
        raise FileNotFoundError(
            f"Missing structure for x={case.x_ethanol}: {struct_path}"
        )

    case_dir = model_out / f"x_ethanol_{case.x_ethanol:.2f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    rho_series = run_one_case(
        struct_path, calc, workdir=case_dir, continue_running=CONTINUE_RUNNING
    )

    ts_path = case_dir / "density_timeseries.csv"
    write_density_timeseries_checkpointed(ts_path, rho_series)


def water_ethanol_density_dummy_data_one_case(mlip: tuple[str, Any], case) -> None:
    """
    Generate one synthetic density time series for debugging.

    Parameters
    ----------
    mlip : tuple[str, Any]
        Pair of model name and model object.
    case : Any
        Composition case containing ``x_ethanol``.

    Returns
    -------
    None
        This function writes a fake density time series.
    """
    model_name, model = mlip

    model_out = OUT_PATH / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    model_kind = MODEL_INDEX[model_name] % 4
    xs_curve, ys_curve = make_fake_curve(model_kind, seed=MODEL_INDEX[model_name])
    xs_curve = np.asarray(xs_curve, dtype=float)
    ys_curve = np.asarray(ys_curve, dtype=float)

    case_dir = model_out / f"x_ethanol_{case.x_ethanol:.2f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    rho_eq = float(np.interp(case.x_ethanol, xs_curve, ys_curve))
    n_steps = 200  # fixed for dummy data

    seed = (hash(model_name) ^ hash(round(case.x_ethanol, 4))) & 0xFFFFFFFF
    rho_series = make_fake_density_timeseries(
        rho_eq, n_steps, seed=seed, start_offset=0.01, tau=0.10, noise_sigma=0.0005
    )

    ts_path = case_dir / "density_timeseries.csv"
    write_density_timeseries_checkpointed(ts_path, rho_series, do_not_raise=True)


if __name__ == "__main__":  # TODO: delete this
    # run a very small simulation to see if it does something reasonable
    from ase import units
    from mace.calculators import mace_mp

    calc = mace_mp(
        "data_old/mace-omat-0-small.model",
        dispersion=True,
        dispersion_cutoff=25 * units.Bohr,
    )
    rho = run_one_case(
        "data/mix_xe_0.00.extxyz",
        calc,
        nvt_steps=1000,
        npt_steps=1000,
        log_every=50,
        workdir=Path("debug"),
        continue_running=False,
    )
    print(rho)
