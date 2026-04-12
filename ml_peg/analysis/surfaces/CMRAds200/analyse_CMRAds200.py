"""Analyse CMRAds200 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "surfaces" / "CMRAds200" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "CMRAds200"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)


def labels() -> list:
    """
    Get list of labels.

    Returns
    -------
    list
        List of all energy labels.
    """
    structs = read(CALC_PATH / "mol_surface_structs.extxyz", index=":")
    return [struct.info["sys_formula"] for struct in structs]


def system_names() -> list:
    """
    Get list of system names.

    Returns
    -------
    list
        List of all system names.
    """
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            structs = read(model_dir / "mol_surface_structs.extxyz", index=":")
            system_names = [struct.info["sys_formula"] for struct in structs]
            break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_adsorption_energies.json",
    title="Adsorption energies",
    x_label="Predicted adsorption energy / eV",
    y_label="Reference adsorption energy / eV",
    hoverdata={
        "System": system_names(),
    },
)
def adsorption_energies() -> dict[str, list]:
    """
    Get adsorption energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted adsorption energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            results[model_name] = []
            continue
        mol_surface_list = read(model_dir / "mol_surface_structs.extxyz", index=":")
        for mol_surface_idx, mol_surface in enumerate(mol_surface_list):

            # Get pre-calculated adsorption energies
            pred_ads_energy = mol_surface.info["pred_adsorption_energy"]
            results[model_name].append(pred_ads_energy)

            if not ref_stored:
                ref_ads_energy = mol_surface.info["PBE_adsorption_energy"]
                results["ref"].append(ref_ads_energy)

            # Write molecule-surface structure to app data
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system_name}.xyz", mol_surface)

        ref_stored = True
    return results


@pytest.fixture
def cmrads_mae(adsorption_energies) -> dict[str, float]:
    """
    Get mean absolute error for adsorption energies.

    Parameters
    ----------
    adsorption_energies
        Dictionary of reference and predicted adsorption energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted adsorption energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            adsorption_energies["ref"], adsorption_energies[model_name]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "cmrads_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(cmrads_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all CMRAds200 metrics.

    Parameters
    ----------
    cmrads_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": cmrads_mae,
    }


def test_cmrads200(metrics: dict[str, dict]) -> None:
    """
    Run CMRAds200 test.

    Parameters
    ----------
    metrics
        All CMRAds200 metrics.
    """
    return
