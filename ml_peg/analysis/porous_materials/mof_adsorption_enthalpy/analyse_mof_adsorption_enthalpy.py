"""Analyse MOF adsorption enthalpy benchmark."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from ase import units
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    get_struct_info,
    load_metrics_config,
    mae,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.porous_materials.mof_adsorption_enthalpy.widom_utils import (
    load_reference,
    scored_frameworks,
    widom_estimates,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "porous_materials" / "mof_adsorption_enthalpy" / "outputs"
OUT_PATH = APP_ROOT / "data" / "porous_materials" / "mof_adsorption_enthalpy"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_CONFIG_PATH)

REFERENCE = load_reference()
TEMPERATURE = REFERENCE["temperature"]
FRAMEWORKS = scored_frameworks(REFERENCE)


def get_estimates(model_name: str, name: str, kind: str) -> dict[str, float] | None:
    """
    Get Widom estimates for one framework.

    Parameters
    ----------
    model_name
        Name of model.
    name
        Name of framework.
    kind
        Either "rigid" or "npt".

    Returns
    -------
    dict[str, float] | None
        Widom estimates, or None if outputs are missing.
    """
    stem = REFERENCE["frameworks"][name]["structure"].removesuffix(".cif")
    out_dir = CALC_PATH / model_name / stem
    energies_file = out_dir / f"widom_{kind}.npy"
    if not (out_dir / "info.json").exists() or not energies_file.exists():
        return None
    with open(out_dir / "info.json", encoding="utf8") as file:
        info = json.load(file)
    volume = info["volume"] if kind == "rigid" else np.mean(info["npt_volumes"])
    density = info["mass"] / units.kg / (volume * 1e-30)
    return widom_estimates(np.load(energies_file), TEMPERATURE, density)


def get_enthalpies(kind: str) -> dict[str, list]:
    """
    Get reference and predicted enthalpies for scored frameworks.

    Parameters
    ----------
    kind
        Either "rigid" or "npt".

    Returns
    -------
    dict[str, list]
        Reference and predicted enthalpies in kJ/mol.
    """
    results = {
        "ref": [REFERENCE["frameworks"][name]["enthalpy"] for name in FRAMEWORKS]
    }
    for model_name in MODELS:
        results[model_name] = []
        for name in FRAMEWORKS:
            estimates = get_estimates(model_name, name, kind)
            results[model_name].append(
                np.nan if estimates is None else estimates["enthalpy"]
            )
    return results


def get_errors(error_func: Callable, enthalpies: dict[str, list]) -> dict:
    """
    Get error of enthalpies for each model.

    Parameters
    ----------
    error_func
        Error function, such as ``mae`` or ``rmse``.
    enthalpies
        Reference and predicted enthalpies.

    Returns
    -------
    dict
        Error for each model, or None if any scored framework is missing.
    """
    errors = {}
    for model_name in MODELS:
        error = error_func(enthalpies["ref"], enthalpies[model_name])
        errors[model_name] = None if np.isnan(error) else error
    return errors


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_enthalpy_rigid.json",
    title="Enthalpy of adsorption (rigid framework)",
    x_label="Predicted enthalpy / kJ/mol",
    y_label="Experimental enthalpy / kJ/mol",
    hoverdata={"Framework": FRAMEWORKS},
)
def enthalpies_rigid() -> dict[str, list]:
    """
    Get enthalpies of adsorption for the relaxed framework.

    Returns
    -------
    dict[str, list]
        Reference and predicted enthalpies in kJ/mol.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    return get_enthalpies("rigid")


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_enthalpy_npt.json",
    title="Enthalpy of adsorption (NPT snapshots)",
    x_label="Predicted enthalpy / kJ/mol",
    y_label="Experimental enthalpy / kJ/mol",
    hoverdata={"Framework": FRAMEWORKS},
)
def enthalpies_npt() -> dict[str, list]:
    """
    Get enthalpies of adsorption averaged over NPT snapshots.

    Returns
    -------
    dict[str, list]
        Reference and predicted enthalpies in kJ/mol.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    return get_enthalpies("npt")


@pytest.fixture
def unphysical_insertions() -> dict[str, float | None]:
    """
    Get percentage of evaluated insertions below the energy floor.

    All frameworks are included, whether or not they have a reference value.

    Returns
    -------
    dict[str, float | None]
        Mean percentage of unphysical insertions for each model.
    """
    results = {}
    for model_name in MODELS:
        fractions = [
            estimates["unphysical_fraction"]
            for name in REFERENCE["frameworks"]
            for kind in ("rigid", "npt")
            if (estimates := get_estimates(model_name, name, kind)) is not None
        ]
        results[model_name] = 100 * float(np.mean(fractions)) if fractions else None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "mof_adsorption_enthalpy_metrics_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics(
    enthalpies_rigid: dict[str, list],
    enthalpies_npt: dict[str, list],
    unphysical_insertions: dict[str, float | None],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    enthalpies_rigid
        Enthalpies for the relaxed framework.
    enthalpies_npt
        Enthalpies averaged over NPT snapshots.
    unphysical_insertions
        Percentage of unphysical insertions.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Rigid MAE": get_errors(mae, enthalpies_rigid),
        "Rigid RMSE": get_errors(rmse, enthalpies_rigid),
        "NPT MAE": get_errors(mae, enthalpies_npt),
        "NPT RMSE": get_errors(rmse, enthalpies_npt),
        "Unphysical insertions": unphysical_insertions,
    }


def test_mof_adsorption_enthalpy(metrics: dict[str, dict]) -> None:
    """
    Run MOF adsorption enthalpy analysis.

    Parameters
    ----------
    metrics
        All metrics.
    """
    get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*/relaxed.extxyz",
        include_dirs=True,
        write_structs=False,
        out_path=OUT_PATH,
    )
