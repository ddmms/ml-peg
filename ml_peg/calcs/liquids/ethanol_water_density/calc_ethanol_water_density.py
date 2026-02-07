import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_peg.calcs.liquids.ethanol_water_density.compositions import BENCH_ROOT, DATA_PATH, load_compositions
from ml_peg.calcs.liquids.ethanol_water_density.fake_data import make_fake_curve, make_fake_density_timeseries
from ml_peg.calcs.liquids.ethanol_water_density.md_code import run_one_case
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

# Local paths
OUT_PATH = BENCH_ROOT / "outputs"

MODELS = load_models(current_models)
MODEL_INDEX = {name: i for i, name in enumerate(MODELS)}
FAKE_DATA = os.getenv("FAKE_DENSITY_DATA", "") == "1"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_water_ethanol_density_curve(mlip: tuple[str, Any]) -> None:
    if not FAKE_DATA:
        water_ethanol_density_curve(mlip)
    else:
        water_ethanol_density_dummy_data(mlip)

def water_ethanol_density_curve(mlip: tuple[str, Any]) -> None:
    """
    Run water–ethanol density curve benchmark for a single MLIP.

    Writes:
      - per-composition density time series (raw data)
      - a summary CSV derived from those time series
    """
    model_name, model = mlip  # TODO: dispersion ???
    cases = load_compositions()

    # Where this model writes outputs
    model_out = OUT_PATH / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    # Get calculator
    calc = model.get_calculator()

    for case in cases:
        struct_path = DATA_PATH / case.filename
        if not struct_path.exists():
            raise FileNotFoundError(
                f"Missing structure for x={case.x_ethanol}: {struct_path}"
            )

        case_dir = model_out / f"x_ethanol_{case.x_ethanol:.2f}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # --- run simulation ---
        rho_series = run_one_case(
            struct_path,
            calc,
            workdir=case_dir,
        )

        # --- write raw density time series ---
        ts_path = case_dir / "density_timeseries.csv"
        with ts_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "rho_g_cm3"])
            for i, rho in enumerate(rho_series):
                w.writerow([i, f"{rho:.8f}"])


def water_ethanol_density_dummy_data(mlip: tuple[str, Any]) -> None:
    model_name, model = mlip
    cases = load_compositions()

    model_out = OUT_PATH / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    # one curve per model
    model_kind = MODEL_INDEX[model_name] % 4
    xs_curve, ys_curve = make_fake_curve(model_kind, seed=MODEL_INDEX[model_name])
    xs_curve = np.asarray(xs_curve, dtype=float)
    ys_curve = np.asarray(ys_curve, dtype=float)

    for case in cases:
        case_dir = model_out / f"x_ethanol_{case.x_ethanol:.2f}"
        case_dir.mkdir(parents=True, exist_ok=True)

        rho_eq = float(np.interp(case.x_ethanol, xs_curve, ys_curve))
        n_steps = 200  # fixed for dummy data

        seed = (hash(model_name) ^ hash(round(case.x_ethanol, 4))) & 0xFFFFFFFF
        rho_series = make_fake_density_timeseries(
            rho_eq, n_steps, seed=seed, start_offset=0.01, tau=0.10, noise_sigma=0.0005
        )

        ts_path = case_dir / "density_timeseries.csv"
        with ts_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "rho_g_cm3"])
            for i, rho in enumerate(rho_series):
                w.writerow([i, f"{rho:.8f}"])


if __name__ == "__main__":  # TODO: delete this
    # run a very small simulation to see if it does something reasonable
    from mace.calculators import mace_mp
    calc = mace_mp("data_old/mace-omat-0-small.model")
    rho = run_one_case("data/mix_xe_0.10.extxyz", calc, nvt_stabilise_steps=250, npt_settle_steps=1000, nvt_thermalise_steps=250, npt_equil_steps=1000, npt_prod_steps=1000, log_every=50, workdir=Path("debug"))
    print(rho)