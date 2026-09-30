"""
Analyse the MOF bulk modulus benchmark.

Fits a Birch-Murnaghan equation of state to the energy-volume curves written
by ``calc_mof_bulk_modulus.py`` and scores the resulting bulk modulus against
experimental values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase.eos import EquationOfState
from ase.units import kJ
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
    load_bulk_modulus_reference,
    unconverged_eos_points,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "porous_materials" / "mof_bulk_modulus" / "outputs"
OUT_PATH = APP_ROOT / "data" / "porous_materials" / "mof_bulk_modulus"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_CONFIG_PATH)

BULK_MODULUS_REFERENCE = load_bulk_modulus_reference()

# A fit to a flat or non-convex energy-volume curve returns a bulk modulus at
# or near zero rather than raising. No real framework is that compliant - the
# softest experimental value in this set is 0.35 GPa - so anything below this
# floor is treated as a failed fit rather than a prediction.
MIN_BULK_MODULUS_GPA = 1.0e-3

# Frameworks whose experimental value can be compared against a fit about a
# single relaxed structure. Records marked "excluded" carry a reason and are
# deliberately left out of every statistic rather than silently dropped.
FRAMEWORKS = sorted(
    name for name, entry in BULK_MODULUS_REFERENCE.items() if not entry.get("excluded")
)
EXCLUDED = sorted(
    name for name, entry in BULK_MODULUS_REFERENCE.items() if entry.get("excluded")
)


def fit_bulk_modulus(volumes: list[float], energies: list[float]) -> float:
    """
    Fit a Birch-Murnaghan equation of state and return the bulk modulus.

    Parameters
    ----------
    volumes
        Cell volumes in cubic Angstrom.
    energies
        Total energies in eV, one per volume.

    Returns
    -------
    float
        Bulk modulus in GPa, or NaN if the fit fails or is degenerate.
    """
    try:
        _, _, bulk_modulus = EquationOfState(
            list(volumes), list(energies), eos="birchmurnaghan"
        ).fit()
    except Exception as exc:
        print(f"Equation-of-state fit failed: {exc}")
        return float("nan")

    # ASE returns the bulk modulus in eV/Angstrom^3.
    bulk_gpa = float(bulk_modulus / kJ * 1.0e24)
    if not np.isfinite(bulk_gpa) or bulk_gpa < MIN_BULK_MODULUS_GPA:
        print(f"Degenerate equation-of-state fit: B = {bulk_gpa:.3g} GPa")
        return float("nan")
    return bulk_gpa


def load_eos(model_name: str, mof: str) -> dict[str, Any] | None:
    """
    Load one framework's energy-volume curve.

    Parameters
    ----------
    model_name
        Name of the model whose output should be read.
    mof
        Framework name.

    Returns
    -------
    dict[str, Any] | None
        Curve record, or ``None`` when missing or unreadable.
    """
    path = CALC_PATH / model_name / f"{mof}_eos.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to load {path}: {exc}")
        return None


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_bulk_modulus.json",
    title="MOF bulk modulus",
    x_label="Predicted bulk modulus / GPa",
    y_label="Experimental bulk modulus / GPa",
    hoverdata={"Framework": FRAMEWORKS},
)
def bulk_moduli() -> dict[str, list]:
    """
    Collect reference and predicted bulk moduli.

    Returns
    -------
    dict[str, list]
        Reference and per-model bulk moduli in GPa, one entry per scored
        framework.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {
        "ref": [BULK_MODULUS_REFERENCE[name]["value"] for name in FRAMEWORKS]
    }
    for model_name in MODELS:
        predicted = []
        for mof in FRAMEWORKS:
            record = load_eos(model_name, mof)
            if record is None:
                predicted.append(float("nan"))
                continue
            unconverged = unconverged_eos_points(record)
            if unconverged:
                scales = ", ".join(f"{s:.2f}" for s in unconverged)
                print(
                    f"{model_name}/{mof}: fixed-cell relaxation hit the step "
                    f"limit at volume scales {scales}; its bulk modulus is "
                    "fitted from partially converged energies."
                )
            predicted.append(
                fit_bulk_modulus(record["volumes_ang3"], record["energies_eV"])
            )
        results[model_name] = predicted
    return results


@pytest.fixture
def bulk_modulus_errors(bulk_moduli: dict[str, list]) -> dict[str, float | None]:
    """
    Get the bulk modulus MAE for each model.

    Frameworks whose scan did not complete are dropped rather than
    propagating NaN through the whole metric.

    Parameters
    ----------
    bulk_moduli
        Reference and predicted bulk moduli.

    Returns
    -------
    dict[str, float | None]
        Mean absolute error in GPa, or ``None`` when a model produced no
        usable fit.
    """
    results: dict[str, float | None] = {}
    reference = bulk_moduli["ref"]
    for model_name in MODELS:
        pairs = [
            (ref, pred)
            for ref, pred in zip(reference, bulk_moduli[model_name], strict=True)
            if np.isfinite(pred)
        ]
        results[model_name] = (
            mae([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "mof_bulk_modulus_metrics_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics(bulk_modulus_errors: dict[str, float | None]) -> dict[str, dict]:
    """
    Assemble the bulk modulus metrics table.

    Parameters
    ----------
    bulk_modulus_errors
        Bulk modulus MAE for each model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {"Bulk modulus MAE": bulk_modulus_errors}


def test_mof_bulk_modulus(
    metrics: dict[str, dict], bulk_moduli: dict[str, list]
) -> None:
    """
    Run the MOF bulk modulus analysis.

    Parameters
    ----------
    metrics
        All bulk modulus metrics.
    bulk_moduli
        Reference and predicted bulk moduli.
    """
    assert FRAMEWORKS, "No frameworks with a usable bulk-modulus reference were found"
    assert len(bulk_moduli["ref"]) == len(FRAMEWORKS)
    # Excluded records must never reach a scored quantity.
    assert not set(FRAMEWORKS) & set(EXCLUDED)
    assert (OUT_PATH / "mof_bulk_modulus_metrics_table.json").exists()
