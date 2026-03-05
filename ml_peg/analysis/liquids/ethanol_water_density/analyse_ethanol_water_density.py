"""Analyse ethanol-water density curves."""

# TODO: remove hardcoded things?
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_peg.analysis.liquids.ethanol_water_density._analysis import (
    _excess_volume,
    _interp_1d,
    _peak_x_quadratic,
    _rmse,
)
from ml_peg.analysis.liquids.ethanol_water_density.io_tools import (
    OUT_PATH,
    _read_model_curve,
    read_ref_curve,
)
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
    """
    Return the reference density curve on a sorted mole-fraction grid.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Sorted mole fractions and reference densities.
    """
    x_ref, rho_ref = read_ref_curve()
    x = np.asarray(x_ref, dtype=float)
    rho = np.asarray(rho_ref, dtype=float)

    # Ensure monotonic x for interpolation
    order = np.argsort(x)
    return x[order], rho[order]


@pytest.fixture
def model_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Return simulated model density curves on sorted composition grids.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Mapping from model name to x-grid and density values.
    """
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODELS:
        xs, rhos = _read_model_curve(model_name)
        x = np.asarray(xs, dtype=float)
        rho = np.asarray(rhos, dtype=float)

        order = np.argsort(x)
        curves[model_name] = (x[order], rho[order])
    return curves


def labels() -> list:
    """
    Get list of calculated concentrations.

    Returns
    -------
    list
        List of all calculated concentrations.
    """
    for model_name in MODELS:
        labels_list, _ = _read_model_curve(model_name)
        break
    return labels_list


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "density_parity.json",
    title="Ethanol–water density (293.15 K)",
    x_label="Reference density / g cm⁻³",
    y_label="Predicted density / g cm⁻³",
    hoverdata={
        "Labels": labels(),
    },
)
def densities_parity(ref_curve, model_curves) -> dict[str, list]:
    """
    Build parity-plot payload for model and reference densities.

    Parameters
    ----------
    ref_curve : tuple[numpy.ndarray, numpy.ndarray]
        Reference composition and density arrays.
    model_curves : dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Per-model composition and density arrays.

    Returns
    -------
    dict[str, list]
        Reference and model densities sampled on a common grid.
    """
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

    return results


@pytest.fixture
def rmse_density(ref_curve, model_curves) -> dict[str, float]:
    """
    Compute density RMSE versus interpolated reference values.

    Parameters
    ----------
    ref_curve : tuple[numpy.ndarray, numpy.ndarray]
        Reference composition and density arrays.
    model_curves : dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Per-model composition and density arrays.

    Returns
    -------
    dict[str, float]
        RMSE values in g/cm^3 keyed by model name.
    """
    x_ref, rho_ref = ref_curve
    out: dict[str, float] = {}
    for m, (x_m, rho_m) in model_curves.items():
        rho_ref_m = _interp_1d(x_ref, rho_ref, x_m)
        out[m] = _rmse(rho_m, rho_ref_m)
    return out


@pytest.fixture
def rmse_excess_volume(ref_curve, model_curves) -> dict[str, float]:
    """
    Compute RMSE of excess volume curves.

    Parameters
    ----------
    ref_curve : tuple[numpy.ndarray, numpy.ndarray]
        Reference composition and density arrays.
    model_curves : dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Per-model composition and density arrays.

    Returns
    -------
    dict[str, float]
        Excess-density RMSE values keyed by model name.
    """
    x_ref, rho_ref = ref_curve
    out: dict[str, float] = {}

    for m, (x_m, rho_m) in model_curves.items():
        rho_ref_m = _interp_1d(x_ref, rho_ref, x_m)

        ex_ref = _excess_volume(x_m, rho_ref_m)
        ex_m = _excess_volume(x_m, rho_m)

        out[m] = _rmse(ex_m, ex_ref)

    return out


@pytest.fixture
def peak_x_error(ref_curve, model_curves) -> dict[str, float]:
    """
    Compute absolute error in composition of minimum excess volume.

    Parameters
    ----------
    ref_curve : tuple[numpy.ndarray, numpy.ndarray]
        Reference composition and density arrays.
    model_curves : dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Per-model composition and density arrays.

    Returns
    -------
    dict[str, float]
        Absolute peak-position error keyed by model name.
    """
    x_ref, rho_ref = ref_curve
    ex_ref_dense = _excess_volume(x_ref, rho_ref)
    x_peak_ref = _peak_x_quadratic(x_ref, ex_ref_dense)
    print("ref peak at:", x_peak_ref)

    out: dict[str, float] = {}
    for m, (x_m, rho_m) in model_curves.items():
        ex_m = _excess_volume(x_m, rho_m)
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
        "RMSE density": "RMSE between model and reference density"
        "at model compositions (g cm⁻³).",
        "RMSE excess volume": (
            "RMSE of the excess volumebetween pure endpoints (cm³ mol^-1)."
        ),
        "Peak x error": (
            "Absolute difference in mole-fraction location of maximum excess density."
        ),
    },
)
def metrics(
    rmse_density: dict[str, float],
    rmse_excess_volume: dict[str, float],
    peak_x_error: dict[str, float],
) -> dict[str, dict]:
    """
    Combine individual metrics into the table payload.

    Parameters
    ----------
    rmse_density : dict[str, float]
        Density RMSE values.
    rmse_excess_volume : dict[str, float]
        Excess-volume RMSE values.
    peak_x_error : dict[str, float]
        Peak-position errors.

    Returns
    -------
    dict[str, dict]
        Metric-name to per-model mapping.
    """
    return {
        "RMSE density": rmse_density,
        "RMSE excess volume": rmse_excess_volume,
        "Peak x error": peak_x_error,
    }


def test_ethanol_water_density(
    metrics: dict[str, dict], densities_parity: dict[str, list]
) -> None:
    """
    Execute density analysis fixtures and emit debug output.

    Parameters
    ----------
    metrics : dict[str, dict]
        Metrics table payload.
    densities_parity : dict[str, list]
        Parity plot payload.

    Returns
    -------
    None
        The test validates fixture execution and writes artifacts.
    """
    print(
        MODEL_INDEX
    )  # TODO: these print statements may be useful for debugging, but should I remove?
    print(
        {
            key0: {MODEL_INDEX[name]: value for name, value in value0.items()}
            for key0, value0 in metrics.items()
        }
    )
    print(densities_parity)
    return
