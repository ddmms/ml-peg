# TODO: remove hardcoded things?
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytest

from ml_peg.analysis.liquids.ethanol_water_density.analysis import _rmse, _interp_1d, \
    _excess_curve, _peak_x_quadratic, x_to_phi_ethanol
from ml_peg.analysis.liquids.ethanol_water_density.io_tools import OUT_PATH, _debug_plot_enabled, _savefig, \
    _read_model_curve, read_ref_curve
from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models


MODELS = get_model_names(current_models)
MODEL_INDEX = {name: i for i, name in enumerate(MODELS)}  # duplicate in calc

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


OUT_PATH.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def ref_curve() -> tuple[np.ndarray, np.ndarray]:
    x_ref, rho_ref = read_ref_curve()
    x = np.asarray(x_ref, dtype=float)
    rho = np.asarray(rho_ref, dtype=float)

    # Ensure monotonic x for interpolation
    order = np.argsort(x)
    return x[order], rho[order]


@pytest.fixture
def model_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODELS:
        xs, rhos = _read_model_curve(model_name)
        x = np.asarray(xs, dtype=float)
        rho = np.asarray(rhos, dtype=float)

        order = np.argsort(x)
        curves[model_name] = (x[order], rho[order])
    return curves


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "density_parity.json",
    title="Ethanol–water density (293.15 K)",
    x_label="Reference density / g cm⁻³",
    y_label="Predicted density / g cm⁻³",
    #hoverdata={
    #    "x_ethanol": [],  # filled in fixture
    #},
)  # TODO: read docs!!! doesn't seem to work yet.
def densities_parity(ref_curve, model_curves) -> dict[str, list]:
    x_ref, rho_ref = ref_curve

    # Use the first model's x grid for hover labels (parity requires same-length lists)
    # We’ll choose the densest model grid if they differ.
    model_name_for_grid = max(model_curves, key=lambda m: len(model_curves[m][0]))
    x_grid = model_curves[model_name_for_grid][0]

    results: dict[str, list] = {"ref": []} | {m: [] for m in MODELS}

    rho_ref_on_grid = _interp_1d(x_ref, rho_ref, x_grid)
    results["ref"] = list(rho_ref_on_grid)

    for m in MODELS:
        x_m, rho_m = model_curves[m]
        # Interpolate model to x_grid if needed
        if len(x_m) != len(x_grid) or np.any(np.abs(x_m - x_grid) > 1e-12):
            # This assumes model spans the grid range; otherwise raise.
            rho_m_on_grid = _interp_1d(x_m, rho_m, x_grid)
        else:
            rho_m_on_grid = rho_m
        results[m] = list(rho_m_on_grid)

    ## Patch hoverdata list in-place (decorator reads the dict)
    ## NOTE: if your decorator captures hoverdata at decoration time,
    ## switch to hoverdata={"x_ethanol": x_labels()} fixture pattern like the docs.
    #densities_parity.__wrapped__.__dict__.setdefault("hoverdata", {})["x_ethanol"] = list(x_grid)

    return results

@pytest.fixture
def debug_curve_plots(ref_curve, model_curves) -> None:  # TODO should I remove or use a different format?
    if not _debug_plot_enabled():
        return
    print("plotting curves")

    x_ref, rho_ref = ref_curve

    for m, (x_m, rho_m) in model_curves.items():
        rho_ref_m = _interp_1d(x_ref, rho_ref, x_m)

        fig, ax = plt.subplots()
        ax.plot(x_ref, rho_ref, label="ref (dense)")
        ax.plot(x_m, rho_m, marker="o", label=f"{m} (model)")
        ax.plot(x_m, rho_ref_m, marker="x", label="ref on model grid")
        ax.set_title(f"Density curve: {m}")
        ax.set_xlabel("x_ethanol")
        ax.set_ylabel("rho / g cm$^{-3}$")
        ax.legend()

        print("saving a curve at:", OUT_PATH / "debug" / m / "density_curve.svg")

        # excess density
        _savefig(fig, OUT_PATH / "debug" / m / "density_curve.svg")
        rho_ref_m = _interp_1d(x_ref, rho_ref, x_m)

        fig, ax = plt.subplots()
        ax.plot(x_ref, _excess_curve(x_ref, rho_ref), label="ref (dense)")
        ax.plot(x_m, _excess_curve(x_m, rho_m), marker="o", label=f"{m} (model)")
        ax.plot(x_m, _excess_curve(x_m, rho_ref_m), marker="x", label="ref on model grid")
        ax.set_title(f"Density curve: {m}")
        ax.set_xlabel("x_ethanol")
        ax.set_ylabel("rho / g cm$^{-3}$")
        ax.legend()

        print("saving a curve at:", OUT_PATH / "debug" / m / "excess_density_curve.svg")
        _savefig(fig, OUT_PATH / "debug" / m / "excess_density_curve.svg")

        # volume fraction plot
        phi_ref = x_to_phi_ethanol(x_ref, rho_ref)
        phi_m   = x_to_phi_ethanol(x_m, rho_m)

        fig, ax = plt.subplots()
        ax.plot(phi_ref, rho_ref, label="ref (dense)")
        ax.plot(phi_m, rho_m, marker="o", label=f"{m} (model)")
        ax.plot(phi_m, rho_ref_m, marker="x", label="ref on model grid")

        ax.set_title(f"Density curve (volume fraction): {m}")
        ax.set_xlabel(r"$\phi_\mathrm{ethanol}$")
        ax.set_ylabel("rho / g cm$^{-3}$")
        ax.legend()

        out_phi = OUT_PATH / "debug" / m / "density_curve_phi.svg"
        print("saving a curve at:", out_phi)
        _savefig(fig, out_phi)


@pytest.fixture
def rmse_density(ref_curve, model_curves) -> dict[str, float]:
    x_ref, rho_ref = ref_curve
    out: dict[str, float] = {}
    for m, (x_m, rho_m) in model_curves.items():
        rho_ref_m = _interp_1d(x_ref, rho_ref, x_m)
        out[m] = _rmse(rho_m, rho_ref_m)
    return out


@pytest.fixture
def rmse_excess_density(ref_curve, model_curves) -> dict[str, float]:
    """
    RMSE of excess density (detrended by each dataset's own pure endpoints).
    """
    x_ref, rho_ref = ref_curve
    out: dict[str, float] = {}

    for m, (x_m, rho_m) in model_curves.items():
        rho_ref_m = _interp_1d(x_ref, rho_ref, x_m)

        ex_ref = _excess_curve(x_m, rho_ref_m)
        ex_m = _excess_curve(x_m, rho_m)

        out[m] = _rmse(ex_m, ex_ref)

    return out


@pytest.fixture
def peak_x_error(ref_curve, model_curves) -> dict[str, float]:
    """
    Absolute error in the x-position of the maximum excess density.

    Ref peak is computed on the dense reference curve.
    Model peak is computed on its (coarse) grid with a local quadratic refinement.
    """
    x_ref, rho_ref = ref_curve
    ex_ref_dense = _excess_curve(x_ref, rho_ref)
    x_peak_ref = _peak_x_quadratic(x_ref, ex_ref_dense)
    print("ref peak at:", x_peak_ref)

    out: dict[str, float] = {}
    for m, (x_m, rho_m) in model_curves.items():
        ex_m = _excess_curve(x_m, rho_m)
        x_peak_m = _peak_x_quadratic(x_m, ex_m)
        out[m] = float(abs(x_peak_m - x_peak_ref))

    return out

# -----------------------------------------------------------------------------
# Table
# -----------------------------------------------------------------------------

@pytest.fixture
@build_table(
    thresholds=DEFAULT_THRESHOLDS,
    filename=OUT_PATH / "density_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "RMSE density": "RMSE between model and reference density at model compositions (g cm⁻³).",
        "RMSE excess density": (
            "RMSE after subtracting each curve’s linear baseline between pure endpoints (g cm⁻³)."
        ),
        "Peak x error": (
            "Absolute difference in mole-fraction location of maximum excess density."
        ),
    },
)
def metrics(
    rmse_density: dict[str, float],
    rmse_excess_density: dict[str, float],
    peak_x_error: dict[str, float],
) -> dict[str, dict]:
    return {
        "RMSE density": rmse_density,
        "RMSE excess density": rmse_excess_density,
        "Peak x error": peak_x_error,
    }


def test_ethanol_water_density(metrics: dict[str, dict], densities_parity: dict[str, list], debug_curve_plots) -> None:
    """
    Launch analysis (decorators handle writing JSON artifacts for the app).
    """
    print(MODEL_INDEX)  # TODO: these print statements may be useful for debugging, but should I remove?
    print({key0:{MODEL_INDEX[name]: value for name, value in value0.items()} for key0, value0 in metrics.items()})
    return
