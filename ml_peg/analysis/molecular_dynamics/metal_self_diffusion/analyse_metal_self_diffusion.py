"""Analyse the metal self-diffusion benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    """Load all temperatures per model/element once for analysis and plotting."""
    results = {"times": None} | {model_name: {} for model_name in MODELS}

    for model_name in MODELS:
        for label, temperatures in zip(
            INFO["filenames"], INFO["temperatures"], strict=True
        ):
            vacfs = np.full((len(temperatures), VACF_SAMPLES, VACF_STEPS + 1), np.nan)

            for t, temp in enumerate(temperatures):
                for s in range(VACF_SAMPLES):
                    file = (
                        CALC_PATH / model_name / label / f"{temp:g}K"
                        / f"vacf_{s}.npy"
                    )
                    if not file.is_file():
                        continue
                    times, vacfs[t, s] = np.load(file)
                    if results["times"] is None:
                        results["times"] = times

            results[model_name][label] = {
                temp: vacfs[t] for t, temp in enumerate(temperatures)
                if np.isfinite(vacfs[t]).any()
            }

    for element in INFO["filenames"]:

        @plot_scatter(
            filename=OUT_PATH / f"{element}_vacfs.json",
            title=f"{element} VACFs",
            x_label="Time / ps",
            y_label=r"Velocity autocorrelation / $\text{Å}^{2}/\text{fs}^{2}$",
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
    """Integrate VACFs and plot all models against experiment per element.

    Returned diffusion coefficients are in m²/s.
    """
    results = {model_name: {} for model_name in ("ref", *MODELS)}

    for label, temperatures, references in zip(
        INFO["filenames"],
        INFO["temperatures"],
        INFO["diffusion_constants"],
        strict=True,
    ):
        diffusion = np.asarray(references)
        results["ref"][label] = {
            "diffusion": diffusion,
            "slope": float(np.polyfit(temperatures, diffusion, 1)[0]),
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
            results[model_name][label] = {
                "diffusion": diffusion,
                "diffusion_error": diffusion_error,
                "slope": (
                    float(np.polyfit(temperatures, diffusion, 1)[0])
                    if np.isfinite(diffusion).all() else None
                ),
            }

    for element, temperatures in zip(
        INFO["filenames"], INFO["temperatures"], strict=True
    ):

        @plot_scatter(
            filename=OUT_PATH / f"{element}_diffusion.json",
            title=f"{element} self-diffusion",
            x_label="Temperature / K",
            y_label=r"Diffusion constant / $10^{-9}\,\text{m}^{2}/\text{s}$",
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
    """Average absolute mean relative errors across elements."""
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
    """Average absolute relative linear slope errors equally over elements."""
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
    """Build the absolute mean relative error and relative slope error table."""
    return {"AMRE": get_amre, "Slope ARE": get_slope_are}


def test_metal_self_diffusion(metrics: dict) -> None:
    """Run the VACF, diffusion, error, and metric-table fixture chain."""
