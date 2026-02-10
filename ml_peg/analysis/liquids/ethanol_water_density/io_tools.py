"""i/o tools for analysis of ethanol-water densities."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ml_peg.analysis.liquids.ethanol_water_density._analysis import (
    weight_to_mole_fraction,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CATEGORY = "liquids"
BENCHMARK = "ethanol_water_density"
CALC_PATH = CALCS_ROOT / CATEGORY / BENCHMARK / "outputs"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK
DATA_PATH = CALCS_ROOT / CATEGORY / BENCHMARK / "data"


def _debug_plot_enabled() -> bool:
    # Turn on plots by: DEBUG_PLOTS=1 pytest ...
    return os.environ.get("DEBUG_PLOTS", "0") not in ("0", "", "false", "False")


def _savefig(fig, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _read_model_curve(model_name: str) -> tuple[list[float], list[float]]:
    """
    Read model density curve by computing averages from raw time series.

    Expects per-composition files:
      x_ethanol_XX/density_timeseries.csv
    with columns: step, rho_g_cm3
    """
    model_dir = CALC_PATH / model_name
    xs: list[float] = []
    rhos: list[float] = []

    for case_dir in sorted(model_dir.glob("x_ethanol_*")):
        x_ethanol = float(case_dir.name.replace("x_ethanol_", ""))

        ts_path = case_dir / "density_timeseries.csv"
        if not ts_path.exists():
            raise FileNotFoundError(f"Missing density time series: {ts_path}")

        rho_vals = []
        steps = []
        with ts_path.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                steps.append(int(row["step"]))
                rho_vals.append(float(row["rho_g_cm3"]))

        if not rho_vals:
            raise ValueError(f"No density samples found in {ts_path}")

        rho_mean = float(np.mean(rho_vals))
        xs.append(x_ethanol)
        rhos.append(rho_mean)

        if _debug_plot_enabled():
            fig, ax = plt.subplots()
            ax.plot(steps, rho_vals)
            ax.axhline(rho_mean, linestyle="--")
            ax.set_title(f"{model_name}  x={x_ethanol:.2f}  density timeseries")
            ax.set_xlabel("step")
            ax.set_ylabel("rho / g cm$^{-3}$")

            _savefig(
                fig,
                OUT_PATH / "debug" / model_name / f"x_{x_ethanol:.2f}_timeseries.svg",
            )

    return xs, rhos


def read_ref_curve() -> tuple[list[float], list[float]]:
    """
    Load densities given on a uniform weight-fraction grid.

    And convert to mole fraction.
    Densities from:
    M. Southard and D. Green, Perry’s Chemical Engineers’ Handbook,
    9th Edition. McGraw-Hill Education, 2018.

    Assumes:
      - 101 evenly spaced points
      - first line = 0 wt% ethanol
      - last line  = 100 wt%
    """
    ref_file = DATA_PATH / "densities_293.15.txt"
    rho_ref = np.loadtxt(ref_file)

    n = len(rho_ref)

    # weight fraction grid
    w = np.linspace(0.0, 1.0, n)

    # convert to mole fraction
    x = weight_to_mole_fraction(w)

    return list(x), list(rho_ref)
