"""Analyse the metal self-diffusion benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

import numpy as np
from plotly.colors import qualitative
import pytest

from ml_peg.analysis.utils.decorators import (
    build_table,
    plot_scatter,
)
from ml_peg.analysis.utils.utils import (
    get_struct_info,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.molecular_dynamics.metal_self_diffusion.calc_metal_self_diffusion import (  # noqa: E501
    VACF_SAMPLES,
    VACF_STEPS,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "metal_self_diffusion" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "metal_self_diffusion"

VACF_TAPER_START = 500.0
VACF_TAPER_END = 1000.0

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    index=0,
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
    include_filenames=True,
    per_file_info={
        "temperatures": lambda atoms: np.asarray(
            atoms[0].info["temperatures"]
        ).tolist(),
        "diffusion_constants": lambda atoms: np.asarray(
            atoms[0].info["diffusion_constants"]
        ).tolist(),
    },
)


@pytest.fixture(scope="session")
def vacfs() -> dict:
    """
    Load and cosine-taper the VACFs for analysis and plotting.

    Returns
    -------
    dict
        Shared lag times and tapered VACFs by model, element, and temperature.
    """
    results = {"times": None} | {model_name: {} for model_name in MODELS}

    for model_name in MODELS:
        for label, temperatures in zip(
            INFO["filenames"], INFO["temperatures"], strict=True
        ):
            vacfs = np.full((len(temperatures), VACF_SAMPLES, VACF_STEPS + 1), np.nan)

            for t, temp in enumerate(temperatures):
                for s in range(VACF_SAMPLES):
                    file = (
                        CALC_PATH / model_name / label / f"{temp:g}K" / f"vacf_{s}.npy"
                    )
                    if not file.is_file():
                        continue
                    data = np.load(file)
                    if data.shape != (2, VACF_STEPS + 1):
                        warn(f"Skipping malformed VACF file: {file}", stacklevel=2)
                        continue
                    times, values = data
                    phase = np.clip(
                        (times - VACF_TAPER_START)
                        / (VACF_TAPER_END - VACF_TAPER_START),
                        0,
                        1,
                    )
                    window = (1 + np.cos(np.pi * phase)) / 2
                    vacfs[t, s] = values * window
                    if results["times"] is None:
                        results["times"] = times

            results[model_name][label] = {
                temp: vacfs[t]
                for t, temp in enumerate(temperatures)
                if np.isfinite(vacfs[t]).any()
            }

    for element in INFO["filenames"]:

        @plot_scatter(
            filename=OUT_PATH / f"{element}_vacfs.json",
            title=f"{element} VACFs",
            x_label="Time / ps",
            y_label=(
                r"$\text{Velocity autocorrelation} / "
                r"\mathrm{\mathring{A}}^{2}\,\mathrm{fs}^{-2}$"
            ),
            show_line=True,
            show_markers=False,
            hovertemplate=(
                "<b>Time: </b>%{x:.3f} ps<br>"
                "<b>Velocity autocorrelation: </b>%{y:.3g} Å²/fs²"
                "<extra>%{fullData.name}</extra>"
            ),
        )
        def curves(element=element):
            """Return mean curves for the selected element."""
            return {
                f"{model_name}, {temperature:g} K": (
                    (results["times"] / 1000).tolist(),
                    np.nanmean(values, axis=0).tolist(),
                )
                for model_name in MODELS
                for temperature, values in results[model_name].get(element, {}).items()
            }

        curves()

    return results


@pytest.fixture(scope="session")
def diffusion_constants(vacfs: dict) -> dict:
    """
    Integrate VACFs and plot all models against experiment per element.

    Parameters
    ----------
    vacfs
        Shared lag times and tapered VACFs by model, element, and temperature.

    Returns
    -------
    dict
        Diffusion coefficients in m²/s, uncertainties, and fits by model and element.
    """
    results = {model_name: {} for model_name in ("ref", *MODELS)}

    for label, temperatures, references in zip(
        INFO["filenames"],
        INFO["temperatures"],
        INFO["diffusion_constants"],
        strict=True,
    ):
        diffusion = np.asarray(references)
        fit = np.polyfit(temperatures, diffusion, 1)
        results["ref"][label] = {
            "diffusion": diffusion,
            "fit": fit,
            "slope": float(fit[0]),
        }

    for model_name in MODELS:
        for label, temperatures in zip(
            INFO["filenames"], INFO["temperatures"], strict=True
        ):
            if not vacfs[model_name].get(label):
                continue
            diffusion = np.full(len(temperatures), np.nan)
            diffusion_error = np.full(len(temperatures), np.nan)
            for t, temperature in enumerate(temperatures):
                values = vacfs[model_name][label].get(temperature)
                if values is None or not np.isfinite(values).all():
                    continue
                samples = (
                    np.trapezoid(values, vacfs["times"], axis=-1)
                    / 3
                    * 1e-5  # Conversion: 1 A^2/fs = 1e-5 m^2/s
                )
                diffusion[t] = samples.mean()
                diffusion_error[t] = 2 * samples.std(ddof=1) / np.sqrt(len(samples))
            valid = np.isfinite(diffusion)
            valid_temperatures = np.asarray(temperatures)[valid]
            fit = (
                np.polyfit(valid_temperatures, diffusion[valid], 1)
                if len(np.unique(valid_temperatures)) >= 2
                else None
            )
            results[model_name][label] = {
                "diffusion": diffusion,
                "diffusion_error": diffusion_error,
                "fit": fit,
                "slope": float(fit[0]) if fit is not None and valid.all() else None,
            }

    for element, temperatures in zip(
        INFO["filenames"], INFO["temperatures"], strict=True
    ):

        @plot_scatter(
            filename=OUT_PATH / f"{element}_diffusion.json",
            title=f"{element} self-diffusion",
            x_label="Temperature / K",
            y_label=(
                r"$\text{Diffusion constant} / "
                r"10^{-9}\,\mathrm{m}^{2}\,\mathrm{s}^{-1}$"
            ),
            show_line=False,
            hovertemplate=(
                "<b>Temperature: </b>%{x:.0f} K<br>"
                "<b>Diffusion constant: </b>%{y:.3f} × 10⁻⁹ m²/s"
                "<extra>%{fullData.name}</extra>"
            ),
        )
        def curves(element=element, temperatures=temperatures):
            """Return the reference and model diffusion curves for this element."""
            return {
                model_name: (
                    list(temperatures),
                    (values[element]["diffusion"] * 1e9).tolist(),
                )
                for model_name, values in results.items()
                if element in values
            }

        curves()
        filename = OUT_PATH / f"{element}_diffusion.json"
        figure = json.loads(filename.read_text())
        points = [trace for trace in figure["data"] if trace["mode"] == "markers"]
        colors = {}
        for i, trace in enumerate(points):
            color = qualitative.Plotly[i % len(qualitative.Plotly)]
            colors[trace["name"]] = color
            trace.setdefault("marker", {})["color"] = color
            trace["legendgroup"] = trace["name"]
        for trace in figure["data"]:
            if trace["mode"] == "lines" and trace["name"] in colors:
                trace["line"]["color"] = colors[trace["name"]]
        for model_name, values in results.items():
            if element not in values or values[element]["fit"] is None:
                continue
            name = "Reference" if model_name == "ref" else model_name
            valid = np.isfinite(values[element]["diffusion"])
            valid_temperatures = np.asarray(temperatures)[valid]
            endpoints = np.array([valid_temperatures.min(), valid_temperatures.max()])
            figure["data"].append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": endpoints.tolist(),
                    "y": (np.polyval(values[element]["fit"], endpoints) * 1e9).tolist(),
                    "line": {"color": colors[name], "dash": "dash"},
                    "name": name,
                    "legendgroup": name,
                    "showlegend": False,
                    "hovertemplate": next(
                        trace["hovertemplate"]
                        for trace in points
                        if trace["name"] == name
                    ),
                }
            )
        filename.write_text(json.dumps(figure))
        errors = {}
        for model_name in MODELS:
            if element not in results[model_name]:
                continue
            errors[model_name] = [
                float(error) if np.isfinite(error) else None
                for error in results[model_name][element]["diffusion_error"]
            ]
        filename = OUT_PATH / f"{element}_diffusion_errors.json"
        filename.write_text(json.dumps(errors, allow_nan=False))
    return results


@pytest.fixture(scope="session")
def get_amre(diffusion_constants: dict) -> dict[str, float | None]:
    """
    Average absolute mean relative errors across elements.

    Parameters
    ----------
    diffusion_constants
        Reference and predicted diffusion coefficients by model and element.

    Returns
    -------
    dict
        Percentage errors by model, or None for incomplete results.
    """
    results = {}
    for model_name in MODELS:
        if any(
            label not in diffusion_constants[model_name]
            or not np.isfinite(
                diffusion_constants[model_name][label]["diffusion"]
            ).all()
            for label in INFO["filenames"]
        ):
            results[model_name] = None
            continue
        element_errors = []
        for label in diffusion_constants[model_name]:
            reference = diffusion_constants["ref"][label]["diffusion"]
            predicted = diffusion_constants[model_name][label]["diffusion"]
            element_errors.append(
                100 * abs(np.mean((predicted - reference) / reference))
            )
        results[model_name] = float(np.mean(element_errors)) if element_errors else None
    return results


@pytest.fixture(scope="session")
def get_slope_are(diffusion_constants: dict) -> dict[str, float | None]:
    """
    Average absolute relative linear slope errors equally over elements.

    Parameters
    ----------
    diffusion_constants
        Reference and predicted diffusion slopes by model and element.

    Returns
    -------
    dict
        Percentage slope errors by model, or None for incomplete results.
    """
    results = {}
    for model_name in MODELS:
        if any(
            label not in diffusion_constants[model_name]
            or diffusion_constants[model_name][label]["slope"] is None
            for label in INFO["filenames"]
        ):
            results[model_name] = None
            continue
        element_errors = []
        for label in diffusion_constants[model_name]:
            predicted_slope = diffusion_constants[model_name][label]["slope"]
            reference_slope = diffusion_constants["ref"][label]["slope"]
            element_errors.append(
                100 * abs(predicted_slope - reference_slope) / abs(reference_slope)
            )
        results[model_name] = float(np.mean(element_errors)) if element_errors else None
    return results


@pytest.fixture(scope="session")
@build_table(
    filename=OUT_PATH / "metal_self_diffusion_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    get_amre: dict[str, float | None], get_slope_are: dict[str, float | None]
) -> dict:
    """
    Build the absolute mean relative error and relative slope error table.

    Parameters
    ----------
    get_amre
        Absolute mean relative errors by model.
    get_slope_are
        Absolute relative slope errors by model.

    Returns
    -------
    dict
        Metric values by metric name and model.
    """
    return {
        "Diffusion coefficient error": get_amre,
        "Temperature slope error": get_slope_are,
    }


def test_metal_self_diffusion(metrics: dict) -> None:
    """
    Run the VACF, diffusion, error, and metric-table fixture chain.

    Parameters
    ----------
    metrics
        Metrics produced by the complete analysis fixture chain.
    """
