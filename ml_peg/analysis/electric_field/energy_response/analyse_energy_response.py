"""Analyse scaling_pol benchmark."""
from __future__ import annotations

"""
from __future__ import annotations

from pathlib import Path

from ase import units
import numpy as np
from ase.io import read, write
import pytest

from ml_peg.analysis.external_field.energy_response.energy_response import get_energy_response
#from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    #build_dispersion_name_map,
    load_metrics_config,
    #mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
#DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "external_field" / "energy_response" / "outputs"
OUT_PATH = APP_ROOT / "data" / "external_field" / "energy_response"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


# Linear organic molecules data sets for electric field response.
DATASETS = [
    "ALKANES",
    "CUMULENES",
]
"""


"""
The goal is to build a table, using the decorator @build_table and to make a 
plot using @plot_parity.

@build_table requires an output from its hos function of the form:

{
    "metric_1": {"model_1": value_1, "model_2": value_2, ...},
    "metric_2": {"model_1": value_3, "model_2": value_4, ...},
    ...
}

where metric_1, metric_2, ... are columns in the table. The plot, on the other 
hand, requires an output of the form:

{
    "ref": ref_values_list,
    "model_1": model_1_values_list,
    "model_2": model_2_values_list,
    ...
}

Essentially, we need to generate the plot data first, since the metric is only
evaluated over this data.

The plot will also need hover_data which takes an output from a 'label' function
of the form:

{
    "label_1": label_1_list,
    "label_2": label_2_list,
    ...
}

Here label_1 is a category and the list will be over all datapoints. Thus, when
hovering over a data point there should be visible all label_k of tha point.


"""


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

"""
MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / [category] / [benchmark_name] / "outputs"
OUT_PATH = APP_ROOT / "data" / [category] / [benchmark_name]
"""

import os

MODELS = get_model_names(current_models)
#DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "electric_field" / "energy_response" / "outputs"
OUT_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


# Linear organic molecules data sets for electric field response.
DATASETS = [
    "ALKANES",
    "CUMULENES",
]



#REF_VALUES = {"path_b": 0.27, "path_c": 2.5}



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




def energy_response(dataset: str) -> dict[str, list]:
    """
    Get energy_responses for all structures.


    RETURN SHOULD LOOK LIKE THIS
    {
        "ref": ref_values_list,
        "model_1": model_1_values_list,
        "model_2": model_2_values_list,
        ...
    }


    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energy_responses.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False
    
    #datasets = [f.name for f in CALC_PATH.glob("*.xyz")]
    
    for model_name in MODELS:
        structs = read(CALC_PATH / model_name / dataset, index=":")

        #results[model_name] = [struct.get_potential_energy() for struct in structs if any(struct.info['external_field'])]
        no_field = np.array([struct.get_potential_energy() for struct in structs if not any(struct.info['external_field'])])
        field = np.arra([struct.get_potential_energy() for struct in structs if any(struct.info['external_field'])])
        results[model_name] = np.abs(field-no_field)

        if not ref_stored:
            #results["ref"] = [struct.info["REF_energy"] for struct in structs if any(struct.info['external_field'])]
            field = np.array([struct.info["REF_energy"] for struct in structs if any(struct.info['external_field'])])
            no_field = np.array([struct.info["REF_energy"] for struct in structs if not any(struct.info['external_field'])])
            results["ref"] = np.abs(field-no_field)

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / dataset, structs)
            ref_stored = True
    
    return results



@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_energy_responses.json",
    title="Relative energy_responses",
    x_label="Predicted energy response / eV",
    y_label="Reference energy response / eV",
    hoverdata={
        "Labels": labels(),
    },
)
def energy_responses() -> dict[str, list]:
    """
    Get energy_responses for all structures.

    merges all the response output of all datasets


    RETURN SHOULD LOOK LIKE THIS
    {
        "ref": ref_values_list,
        "model_1": model_1_values_list,
        "model_2": model_2_values_list,
        ...
    }


    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energy_responses.
    """
    #results = {"ref": []} | {mlip: [] for mlip in MODELS}
    results = {}
    #ref_stored = False
    
    datasets = [f.name for f in CALC_PATH.glob("*.xyz")]
    for dataset in datasets:
        for d in energy_response(dataset):
            for key, value in d.items():
                results.setdefault(key, []).append(value)

    return results



@pytest.fixture
def metric_1(energy_responses: dict[str, list]) -> dict[str, float]:
    """
    Get metric 1.

    Parameters
    ----------
    energy_responses
        Reference and predicted energy_responses for all structures.

    Returns
    -------
    dict[str, float]
        Dictionary of metric 1 values for each model.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(energy_responses["ref"], energy_responses[model_name])

    return results



def metric_2X(
    energy_response: Callable[[str], dict[str, list]],
) -> dict[str, float]:
    """
    Get metric 2.

    Parameters
    ----------
    energy_responses
        Reference and predicted energy_responses for all structures.

    Returns
    -------
    dict[str, float]
        Dictionary of metric 1 values for each model.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(energy_response["ref"], energy_response[model_name])

    return results


@pytest.fixture
def metric_2() -> dict[str, float]:
    """
    Get metric 2.

    Parameters
    ----------
    energy_responses
        Reference and predicted energy_responses for all structures.

    Returns
    -------
    dict[str, float]
        Dictionary of metric 1 values for each model.
    """
    results = {}
    for model_name in MODELS:

        results[model_name] = mae(energy_responses["ref"], energy_responses[model_name])

    return metric_2X(
        energy_response('ALKANES.xyz')
    )


#@pytest.fixture
def metric_2xx(energy_responses: dict[str, list]) -> dict[str, float]:
    """
    Get metric 2.

    Returns
    -------
    dict[str, float]
        Dictionary of metric 2 values for each model.
    """
    results = {}
    for model_name in MODELS:
        structs = read(CALC_PATH / model_name / "ALKANES.xyz", index=":")
        results[model_name] = mae(
            pred_properties, [struct.info["property"] for struct in structs]
        )

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "energy_respone_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    #metric_tooltips={
    #    "Model": "Name of the model",
    #    "Metric 1": "Description for metric 1 (units)",
    #    "Metric 2": "Description for metric 2 (units)",
    #},
    thresholds=DEFAULT_THRESHOLDS,

)
def metrics(
    metric_1: dict[str, float], metric_2: dict[str, float]
) -> dict[str, dict]:
    """
    Get all energy_response metrics.

    Parameters
    ----------
    metric_1
        Metric 1 value for all models.
    metric_2
        Metric 2 value for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Metric 1": metric_1,
        "Metric 2": metric_2,
    }


def test_energy_respone(metrics: dict[str, dict]) -> None:
    """
    Run new benchmark analysis.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return


