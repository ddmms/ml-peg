"""
Analyse the MOF inelastic neutron scattering benchmark.

Reads the shared phonon outputs written by
``ml_peg/calcs/porous_materials/mof_phonons/calc_mof_phonons.py`` and scores
the simulated INS spectra against digitised experimental spectra using the
Wasserstein-1 distance, along with the fraction of imaginary modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_violin
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
    OUTPUT_PATH,
    load_ins_reference,
    load_reference_structures,
    mean_imaginary_deviation,
    wasserstein_spectrum_distance,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = OUTPUT_PATH
OUT_PATH = APP_ROOT / "data" / "porous_materials" / "mof_ins"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_CONFIG_PATH)

INS_REFERENCE = load_ins_reference()
# Score against measured spectra. The bundled data also carries published DFT
# spectra for a subset of frameworks; switching this to "DFT" scores against
# those instead, and requires the matching level_of_theory in metrics.yml.
REFERENCE_TYPE = "experiment"

FRAMEWORKS = sorted(
    name
    for name, entry in INS_REFERENCE.items()
    if REFERENCE_TYPE in entry and name in set(load_reference_structures())
)


def load_spectrum(model_name: str, mof: str) -> dict[str, Any] | None:
    """
    Load one simulated INS spectrum.

    Parameters
    ----------
    model_name
        Name of the model whose output should be read.
    mof
        Framework name.

    Returns
    -------
    dict[str, Any] | None
        Spectrum record, or ``None`` when it is missing or unreadable.
    """
    path = CALC_PATH / model_name / f"{mof}_ins.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to load {path}: {exc}")
        return None


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_ins_wasserstein.json",
    title="Per-framework INS spectral distance",
    y_label="Wasserstein-1 distance / meV",
    hoverdata={"Framework": FRAMEWORKS},
)
def wasserstein_distances() -> dict[str, list]:
    """
    Get the per-framework Wasserstein distance for each model.

    Returns
    -------
    dict[str, list]
        Mapping of model name to its per-framework distances in meV, ordered
        to match ``FRAMEWORKS``.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {}
    for model_name in MODELS:
        distances = []
        for mof in FRAMEWORKS:
            reference = INS_REFERENCE[mof][REFERENCE_TYPE]
            spectrum = load_spectrum(model_name, mof)
            if spectrum is None:
                distances.append(float("nan"))
                continue
            # A reference may declare the sub-range over which it is
            # meaningful; otherwise the overlap of the two ranges is used.
            declared = reference.get("window_meV")
            distances.append(
                wasserstein_spectrum_distance(
                    reference["energy_meV"],
                    reference["intensity"],
                    spectrum["energy_meV"],
                    spectrum["intensity"],
                    window=tuple(declared) if declared else None,
                )
            )
        results[model_name] = distances
    return results


@pytest.fixture
def mean_wasserstein(
    wasserstein_distances: dict[str, list],
) -> dict[str, float | None]:
    """
    Average each model's Wasserstein distances over the frameworks.

    Parameters
    ----------
    wasserstein_distances
        Per-framework distances for each model.

    Returns
    -------
    dict[str, float | None]
        Mean Wasserstein-1 distance in meV, or ``None`` when a model produced
        no comparable spectrum.
    """
    results: dict[str, float | None] = {}
    for model_name, distances in wasserstein_distances.items():
        finite = [value for value in distances if np.isfinite(value)]
        results[model_name] = float(np.mean(finite)) if finite else None
    return results


@pytest.fixture
def imaginary_mode_errors() -> dict[str, float | None]:
    """
    Get the mean deviation of the imaginary-mode percentage from zero.

    Shares its definition with the heat capacity benchmark, since both are
    scored from the same phonon calculation.

    Returns
    -------
    dict[str, float | None]
        Mean absolute percentage of imaginary modes for each model.
    """
    return {model_name: mean_imaginary_deviation(model_name) for model_name in MODELS}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "mof_ins_metrics_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics(
    mean_wasserstein: dict[str, float | None],
    imaginary_mode_errors: dict[str, float | None],
) -> dict[str, dict]:
    """
    Assemble the INS metrics table.

    Parameters
    ----------
    mean_wasserstein
        Mean Wasserstein-1 distance for each model.
    imaginary_mode_errors
        Imaginary-mode percentage MAE for each model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Wasserstein distance": mean_wasserstein,
        "Imaginary modes MAE": imaginary_mode_errors,
    }


def test_mof_ins(
    metrics: dict[str, dict], wasserstein_distances: dict[str, list]
) -> None:
    """
    Run the MOF INS analysis.

    Parameters
    ----------
    metrics
        All INS metrics.
    wasserstein_distances
        Per-framework distances for each model.
    """
    assert FRAMEWORKS, "No frameworks with a reference INS spectrum were found"
    for distances in wasserstein_distances.values():
        assert len(distances) == len(FRAMEWORKS)
    assert set(metrics) == {"Wasserstein distance", "Imaginary modes MAE"}
    assert (OUT_PATH / "mof_ins_metrics_table.json").exists()
