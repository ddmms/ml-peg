"""Analyse ethanol-water density curves."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

CATEGORY = "molecular_dynamics"
BENCHMARK = "ethanol_water_density"
CALC_PATH = CALCS_ROOT / CATEGORY / BENCHMARK / "outputs"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK

MODELS = get_model_names(current_models)
MODEL_INDEX = {name: i for i, name in enumerate(MODELS)}  # duplicate in calc

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

M_WATER = 18.01528  # g/mol
M_ETOH = 46.06844  # g/mol
LOG_INTERVAL_PS = 0.1
EQUILIB_TIME_PS = 500

OUT_PATH.mkdir(parents=True, exist_ok=True)

# Save per-composition elemental info (and composition dir labels) for app filtering.
INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*/*.traj",
    index=0,
    write_info=True,
    write_structs=False,  # all files are "mock.traj", flat write would collide
    out_path=OUT_PATH,
    include_dirs=True,
)


def weight_to_mole_fraction(w):
    r"""
    Convert ethanol weight fraction to mole fraction.

    Parameters
    ----------
    w : array-like
        Ethanol weight fraction :math:`m_\mathrm{ethanol} / m_\mathrm{total}`.

    Returns
    -------
    numpy.ndarray
        Ethanol mole fraction.
    """
    n_e = w / M_ETOH
    n_w = (1 - w) / M_WATER
    return n_e / (n_e + n_w)


def _excess_volume(x: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    """
    Compute excess volume given molar fraction and density respectively.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid (mol fraction).
    rhos : numpy.ndarray
        Density.

    Returns
    -------
    numpy.ndarray
        Excess values ``y - y_linear``.
    """
    return (x * M_ETOH + (1 - x) * M_WATER) / rhos - (
        x * M_ETOH / rhos[-1] + (1 - x) * M_WATER / rhos[0]
    )


def _quadratic_min_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Fit a parabola around the minimum and locate its vertex.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid.
    y : numpy.ndarray
        Property values.

    Returns
    -------
    tuple[numpy.ndarray | None, numpy.ndarray | None, float]
        Parabola coefficients, x-bracket of the fit, and estimated composition
        of the minimum. Coefficients are None when no local fit is possible.
    """
    i = int(np.argmin(y))
    if len(x) < 3 or i == 0 or i == len(x) - 1:
        return None, None, float(x[i])

    # Fit a parabola to (i-1, i, i+1)
    xs = x[i - 1 : i + 2]
    ys = y[i - 1 : i + 2]

    # y = ax^2 + bx + c
    coeffs = np.polyfit(xs, ys, deg=2)
    a, b = coeffs[0], coeffs[1]
    if abs(a) < 1e-16:
        return None, xs, float(x[i])

    xv = -b / (2.0 * a)

    # Clamp to local bracket so we don't get silly extrapolation
    return coeffs, xs, float(np.clip(xv, xs.min(), xs.max()))


def _peak_x_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate x position of the minimum by local quadratic fitting.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid.
    y : numpy.ndarray
        Property values.

    Returns
    -------
    float
        Estimated composition of the minimum.
    """
    return _quadratic_min_fit(x, y)[2]


@pytest.fixture(scope="session")
def ref_curve() -> tuple[np.ndarray, np.ndarray]:
    """
    Return the reference density curve on a sorted mole-fraction grid.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Sorted mole fractions and reference densities.
    """
    data_dir = (
        download_s3_data(
            key=f"inputs/{CATEGORY}/ethanol_water_density/ethanol_water_density.zip",
            filename="ethanol_water_density.zip",
        )
        / "ethanol_water_density"
    )
    ref_file = data_dir / "densities_293.15.txt"
    rho_ref = np.loadtxt(ref_file)

    n = len(rho_ref)

    # weight fraction grid
    w = np.linspace(0.0, 1.0, n)

    # convert to mole fraction
    x = weight_to_mole_fraction(w)

    return x, rho_ref


def compute_density(fname, density_col=13):
    """
    Compute average density from NPT log file.

    Parameters
    ----------
    fname
        Path to the log file.
    density_col
        Which column the density numbers are in.

    Returns
    -------
    float
        Average density in g/cm3.
    """
    density_series = []
    try:
        with open(fname) as lines:
            for line in lines:
                items = line.strip().split()
                if len(items) != 15:
                    continue
                density_series.append(float(items[13]))
    except OSError:
        return np.nan
    skip_frames = int(EQUILIB_TIME_PS / LOG_INTERVAL_PS)
    equilibrated = density_series[skip_frames:]
    if len(equilibrated) == 0:
        return np.nan
    return np.mean(equilibrated)


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
        model_dir = CALC_PATH / model_name
        xs = []
        rhos = []
        if model_dir.is_dir():
            for case_dir in model_dir.iterdir():
                rhos.append(compute_density(case_dir / f"{model_name}.log"))
                xs.append(float(case_dir.name.split("_")[-1]))
        else:
            # A model in `current_models` may not have been calculated; leave its curve
            # empty so downstream metrics resolve to NaN rather than crashing.
            warn(
                f"No outputs for model {model_name}; metrics set to NaN.", stacklevel=2
            )
        x = np.asarray(xs, dtype=float)
        rho = np.asarray(rhos, dtype=float)

        order = np.argsort(x)
        curves[model_name] = (x[order], rho[order])
    return curves


def plot_model_curves(
    model_name: str,
    ref_curve: tuple[np.ndarray, np.ndarray],
    model_curve: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    Plot density, excess volume, and excess-volume minimum figures for one model.

    Parameters
    ----------
    model_name
        Name of MLIP.
    ref_curve
        Reference composition and density arrays.
    model_curve
        Model composition and density arrays.
    """
    x_ref, rho_ref = ref_curve
    x_m, rho_m = model_curve
    model_dir = OUT_PATH / model_name

    ex_ref = _excess_volume(x_ref, rho_ref)
    ex_m = _excess_volume(x_m, rho_m)

    @plot_scatter(
        filename=model_dir / "figure_density.json",
        title="Ethanol–water density (293.15 K)",
        x_label="Ethanol mole fraction",
        y_label="Density / g cm⁻³",
        show_line=True,
    )
    def plot_density() -> dict[str, list]:
        """
        Plot density against mole fraction.

        Returns
        -------
        dict[str, list]
            Mole fractions and densities for the reference and model.
        """
        return {
            "ref": [list(x_ref), list(rho_ref)],
            model_name: [list(x_m), list(rho_m)],
        }

    @plot_scatter(
        filename=model_dir / "figure_excess_volume.json",
        title="Excess molar volume (293.15 K)",
        x_label="Ethanol mole fraction",
        y_label="Excess molar volume / cm³ mol⁻¹",
        show_line=True,
    )
    def plot_excess_volume() -> dict[str, list]:
        """
        Plot excess molar volume against mole fraction.

        Returns
        -------
        dict[str, list]
            Mole fractions and excess volumes for the reference and model.
        """
        return {
            "ref": [list(x_ref), list(ex_ref)],
            model_name: [list(x_m), list(ex_m)],
        }

    @plot_scatter(
        filename=model_dir / "figure_excess_volume_minimum.json",
        title="Excess molar volume minimum (293.15 K)",
        x_label="Ethanol mole fraction",
        y_label="Excess molar volume / cm³ mol⁻¹",
        show_line=True,
    )
    def plot_excess_volume_minimum() -> dict[str, list]:
        """
        Plot excess volume curves with quadratic fits and their minima.

        Returns
        -------
        dict[str, list]
            Mole fractions and excess volumes, plus fit and minimum traces.
        """
        results = {
            "ref": [list(x_ref), list(ex_ref)],
            model_name: [list(x_m), list(ex_m)],
        }
        for name, x, ex in (("Reference", x_ref, ex_ref), (model_name, x_m, ex_m)):
            coeffs, x_bracket, x_min = _quadratic_min_fit(x, ex)
            if coeffs is not None:
                x_fit = np.linspace(x_bracket.min(), x_bracket.max(), 50)
                results[f"{name} fit"] = [
                    list(x_fit),
                    list(np.polyval(coeffs, x_fit)),
                ]
                y_min = float(np.polyval(coeffs, x_min))
            else:
                y_min = float(ex[int(np.argmin(ex))])
            results[f"{name} minimum"] = [[x_min], [y_min]]
        return results

    plot_density()
    plot_excess_volume()
    plot_excess_volume_minimum()


@pytest.fixture
def curve_plots(ref_curve, model_curves) -> None:
    """
    Write per-model density and excess-volume figures.

    Parameters
    ----------
    ref_curve : tuple[numpy.ndarray, numpy.ndarray]
        Reference composition and density arrays.
    model_curves : dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Per-model composition and density arrays.
    """
    for model_name, (x_m, rho_m) in model_curves.items():
        if rho_m.size == 0 or not np.all(np.isfinite(rho_m)):
            continue
        plot_model_curves(model_name, ref_curve, (x_m, rho_m))


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
        if rho_m.size == 0 or not np.all(np.isfinite(rho_m)):
            out[m] = np.nan
            continue
        rho_ref_m = np.interp(x_m, x_ref, rho_ref)
        out[m] = rmse(rho_m, rho_ref_m)
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
        if rho_m.size == 0 or not np.all(np.isfinite(rho_m)):
            out[m] = np.nan
            continue
        rho_ref_m = np.interp(x_m, x_ref, rho_ref)

        ex_ref = _excess_volume(x_m, rho_ref_m)
        ex_m = _excess_volume(x_m, rho_m)

        out[m] = rmse(ex_m, ex_ref)

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
        if rho_m.size == 0 or not np.all(np.isfinite(rho_m)):
            out[m] = np.nan
            continue
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
            "RMSE of the excess volume between pure endpoints (cm³ mol⁻¹)."
        ),
        "Peak x error": (
            "Absolute difference in mole-fraction location of minimum excess volume."
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


def test_ethanol_water_density(metrics: dict[str, dict], curve_plots: None) -> None:
    """
    Execute density analysis fixtures and emit debug output.

    Parameters
    ----------
    metrics : dict[str, dict]
        Metrics table payload.
    curve_plots : None
        Per-model density and excess-volume figures.

    Returns
    -------
    None
        The test validates fixture execution and writes artifacts.
    """
    return
