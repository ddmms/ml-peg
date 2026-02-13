"""Analyse PLF547 benchmark. 10.1021/acs.jcim.9b01171."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "supramolecular" / "PLF547" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "PLF547"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


def get_info() -> dict[str, list[int]]:
    """
    Get dictionary of info for PLF547 structures.

    Returns
    -------
    dict[str, list[int]]
        Dictionary of system indices, complex atom counts, and complex charges.
    """
    info = {"labels": [], "charged": []}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))

            info["labels"] = [path.stem for path in xyz_files]

            for xyz_file in xyz_files:
                structs = read(xyz_file, index=":")
                info["charged"].append(
                    any(struct.info.get("charge") != 0 for struct in structs)
                )
            break

    return info


INFO = get_info()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_plf547.json",
    title="Interaction energies",
    x_label="Predicted energy / kcal/mol",
    y_label="Reference energy / kcal/mol",
    hoverdata={
        "Labels": INFO["labels"],
        "Charged": INFO["charged"],
    },
)
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
        for label in INFO["labels"]:
            atoms = read(CALC_PATH / model_name / f"{label}.xyz", index=0)

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
def total_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for energies for all systems.

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
def charged_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for charged systems only.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for charged systems.
    """
    # Get charges for filtering
    charged_indices = [i for i, charged in enumerate(INFO["charged"]) if charged]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name] and charged_indices:
            ref_charged = [interaction_energies["ref"][i] for i in charged_indices]
            pred_charged = [
                interaction_energies[model_name][i] for i in charged_indices
            ]
            results[model_name] = mae(ref_charged, pred_charged)
        else:
            results[model_name] = None
    return results


@pytest.fixture
def neutral_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for neutral systems only.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for neutral systems.
    """
    # Get charges for filtering
    charged_indices = [i for i, charged in enumerate(INFO["charged"]) if not charged]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name] and charged_indices:
            ref_neutral = [interaction_energies["ref"][i] for i in charged_indices]
            pred_neutral = [
                interaction_energies[model_name][i] for i in charged_indices
            ]
            results[model_name] = mae(ref_neutral, pred_neutral)
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "plf547_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    total_mae: dict[str, float],
    charged_mae: dict[str, float],
    neutral_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    total_mae
        Mean absolute errors for all models.
    charged_mae
        Mean absolute errors for all models for charged systems.
    neutral_mae
        Mean absolute errors for all models for neutral systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Neutral MAE": neutral_mae,
        "Charged MAE": charged_mae,
        "Overall MAE": total_mae,
    }


def test_plf547(metrics: dict[str, dict]) -> None:
    """
    Run PLF547 test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
