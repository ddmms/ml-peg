"""Analyse scaling_pol benchmark."""
from __future__ import annotations

from pathlib import Path

from ase import units
import numpy as np
from ase.io import read, write
import pytest


from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import mae, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models


import os

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "electric_field" / "energy_response" / "outputs"
OUT_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def labels() -> dict[str, list]:
    """
    Axis, legend, and hover label names.

    Returns
    -------
    dict
        Mapping from dataframe column names to
        human-readable labels used in plots.
    """
    datasets = [f.name for f in CALC_PATH.glob("*.xyz")]

    structs = []
    for dataset in datasets:
        structs = read(CALC_PATH / dataset, index=":")
        if all('external_field' in struct.info for struct in structs):
            structs.append(structs)
    
    return {
        'substance':[str(struct.get_chemical_formula()) for struct in structs],
        'energy':[struct.info["energy"] for struct in structs],
        'external_field':[struct.info["external_field"] for struct in structs],
    }


def energy_response() -> dict[str, dict]:
    """
    Get energy_responses for all structures.


    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energy responses.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False
    
    for model_name in MODELS:
        dset_results = {}
        datasets = [f.name for f in (CALC_PATH/model_name).glob("*.xyz")]
    
        for dataset in datasets:
            structs = read(CALC_PATH / model_name / dataset, index=":")
        
            # Model precitions
            no_field = [
                struct.get_potential_energy() 
                for struct in structs if not any(struct.info['external_field'])
            ]
            field = [
                struct.get_potential_energy() 
                for struct in structs if any(struct.info['external_field'])
            ]
            dset_results[dataset] = np.abs(
                np.array(field)-np.array(no_field)
            )

            # Reference values from ORCA
            if not ref_stored:
                field = [
                    struct.info["REF_energy"] 
                    for struct in structs if any(struct.info['external_field'])
                ]
                no_field = [
                    struct.info["REF_energy"] 
                    for struct in structs if not any(struct.info['external_field'])
                ]
                dset_results[dataset] = np.abs(
                    np.array(field)-np.array(no_field)
                )
                
                # Write structures for app
                #structs_dir = OUT_PATH / model_name
                #structs_dir.mkdir(parents=True, exist_ok=True)
                #write(structs_dir / dataset, structs)

        results[model_name] = dset_results
        results["ref"] = dset_results
        ref_stored = True    
    return results



@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_energy_responses.json",
    title="Relative energy_responses",
    x_label="Predicted energy response / meV",
    y_label="Reference energy response / meV",
    hoverdata={
        "Labels": labels(),
    },
)
def energy_responses() -> dict[str, list]:
    """
    Get energy responses for all datasets.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted energy responses.
    """
    results = energy_response()
    flattened_results = {}
    for key, res in results.items():
        flattened_results[key] = np.concatenate([v for k,v in res.items()])
    return flattened_results


@pytest.fixture
def total_mae(energy_responses: dict[str, list]) -> dict[str, float]:
    """
    Get total MAE of all energy responses.

    Parameters
    ----------
    energy_responses
        Reference and predicted energy responses for all structures.

    Returns
    -------
    dict[str, float]
        Dictionary of total MAE values for each model.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(energy_responses["ref"], energy_responses[model_name])
    return results


@pytest.fixture
def alkane_mae() -> dict[str, float]:
    """
    Get MAE of alkane energy responses.

    Parameters
    ----------
    energy_responses
        Reference and predicted energy responses for all alkanes.

    Returns
    -------
    dict[str, float]
        Dictionary of alkane MAE values for each model.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            energy_response()["ref"]["ALKANES.xyz"], energy_response()[model_name]["ALKANES.xyz"]
        )
    return results


@pytest.fixture
def cumulene_mae() -> dict[str, float]:
    """
    Get MAE of cumulene energy responses.

    Parameters
    ----------
    energy_responses
        Reference and predicted energy responses for all cumulenes.

    Returns
    -------
    dict[str, float]
        Dictionary of cumulene MAE values for each model.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            energy_response()["ref"]["CUMULENES.xyz"], energy_response()[model_name]["CUMULENES.xyz"]
        )
    return results



@pytest.fixture
@build_table(
    filename=OUT_PATH / "energy_response_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    weights=DEFAULT_WEIGHTS,
    thresholds=DEFAULT_THRESHOLDS,

)
def metrics(
    total_mae: dict[str, float],
    alkane_mae: dict[str, float],
    cumulene_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all energy_response metrics.

    Parameters
    ----------
    tota_mae
        Total MAE value for all models.
    alkane_mae
        Alkane MAE value for all models.
    cumulene_mae
        Cumulene MAE value for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Total MAE": total_mae,
        "Alkanes MAE": alkane_mae,
        "Cumulenes MAE": cumulene_mae,
    }


def test_energy_response(metrics: dict[str, dict]) -> None:
    """
    Run new benchmark analysis.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return


