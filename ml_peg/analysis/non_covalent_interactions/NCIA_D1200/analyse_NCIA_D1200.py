"""Analyse NCIA D1200 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import (
    build_table,
    plot_density_scatter,
)
from ml_peg.analysis.utils.utils import (
    build_d3_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

EV_TO_KCAL = units.mol / units.kcal
CALC_PATH = CALCS_ROOT / "non_covalent_interactions" / "NCIA_D1200" / "outputs"
OUT_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "NCIA_D1200"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def labels() -> list:
    """
    Get list of system names.

    Returns
    -------
    list
        List of all system names.
    """
    for model in MODELS:
        labels_list = sorted([path.stem for path in (CALC_PATH / model).glob("*.xyz")])
        break
    return labels_list


@pytest.fixture
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted interaction energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    ref_stored = False

    for model_name in MODELS:
        for label in labels():
            atoms = read(CALC_PATH / model_name / f"{label}.xyz")
            if not ref_stored:
                results["ref"].append(atoms.info["ref_int_energy"] * EV_TO_KCAL)

            results[model_name].append(atoms.info["model_int_energy"] * EV_TO_KCAL)

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)

        ref_stored = True
    return results


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_ncia_d1200_density.json",
    title="Interaction energy density plot",
    x_label="Reference energy / kcal/mol",
    y_label="Predicted energy / kcal/mol",
    annotation_metadata={"system_count": "Systems"},
)
def interaction_density(interaction_energies: dict[str, list]) -> dict[str, dict]:
    """
    Build density scatter inputs for interaction energies.

    Parameters
    ----------
    interaction_energies
        Reference and predicted interaction energies per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model names to density-plot payloads.
    """
    ref_vals = interaction_energies["ref"]
    density_inputs: dict[str, dict] = {}
    for model_name in MODELS:
        preds = interaction_energies.get(model_name, [])
        density_inputs[model_name] = {
            "ref": ref_vals,
            "pred": preds,
            "meta": {"system_count": len([val for val in preds if val is not None])},
        }
    return density_inputs


@pytest.fixture
def get_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for energies.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            interaction_energies["ref"], interaction_energies[model_name]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ncia_d1200_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
    }


def test_ncia_d1200(
    metrics: dict[str, dict],
    interaction_density: dict[str, dict],
) -> None:
    """
    Run NCIA D1200 test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    interaction_density
        Density-scatter inputs for all models (drives saved plots).
    """
    return
