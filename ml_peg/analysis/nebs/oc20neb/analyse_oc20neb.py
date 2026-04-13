"""Analyse OC20NEB benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DATA_PATH = CALCS_ROOT / "nebs" / "oc20neb" / "data"
CALC_PATH = CALCS_ROOT / "nebs" / "oc20neb" / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / "oc20neb"
SCATTER_FILENAME = OUT_PATH / "oc20neb_interactive.json"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

REF_VALUES = {
    "desorption_ood_87_9841_0_111-1": 2.061,
    "dissociation_ood_268_6292_46_211-5": 1.505,
    "transfer_id_601_1482_1_211-5": 0.868,
}
REACTIONS = [
    "desorption_ood_87_9841_0_111-1",
    "dissociation_ood_268_6292_46_211-5",
    "transfer_id_601_1482_1_211-5",
]
METRICS = ["delta_E", "barrier", "fmax"]
METRIC_LABELS = {"delta_E": "delta_E", "barrier": "barrier", "fmax": "unconverged"}
FMAX = 0.05


def plot_nebs(model: str, reaction: str) -> None:
    """
    Plot NEB paths and save all structure files.

    Parameters
    ----------
    model
        Name of MLIP.
    reaction
        Reaction id for NEB.
    """

    @plot_scatter(
        filename=OUT_PATH / f"figure_{model}_neb_{reaction}.json",
        title=f"NEB path {reaction}",
        x_label="Image",
        y_label="Energy / eV",
        show_line=True,
    )
    def plot_neb() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot a NEB and save the structure file.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Dictionary of tuples of image/energy for each model.
        """
        results = {}
        structs = read(
            CALC_PATH / f"{reaction}_{model}.xyz",
            index=":",
        )
        results[model] = [
            list(range(len(structs))),
            [struct.info["mlip_energy"] for struct in structs],
        ]
        structs_dir = OUT_PATH / model
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / f"{model}-{reaction}.xyz", structs)

        return results

    plot_neb()


def _get_ref_data():
    """
    Get reference barrier and delta_E for all reactions.

    Returns
    -------
    dict[str, float]
        Dictionary of reference barrier and delta_E with DFT.
    """
    ref_data = {}
    for path in DATA_PATH.glob("*.xyz"):
        reaction = str(path).split("/")[-1].split(".")[0]
        traj = read(path, ":")
        energy = np.array([at.info["DFT_energy"] for at in traj])

        ref_data[reaction] = {
            "energy": energy,
            "barrier": energy.max() - energy[0],
            "delta_E": energy[-1] - energy[0],
        }
    return ref_data


@pytest.fixture
def oc20neb_stats() -> dict[str, dict[str, float]]:
    """
    Get error in energy barrier for all reactions.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier errors for all models.
    """
    ref_data = _get_ref_data()
    ref_cache: dict[str, dict[str, Any]] = {}

    for reaction in REACTIONS:
        ref_traj_path = DATA_PATH / f"{reaction}.xyz"
        ref_profile = ref_data[reaction]["energy"]

        ref_cache[reaction] = {"profile": ref_profile, "traj_path": ref_traj_path}

    stats: dict[str, dict[str, Any]] = {}

    for model_name in MODELS:
        metrics_data: dict[str, dict[str, Any]] = {
            key: {"points": [], "ref": [], "pred": [], "mae": None} for key in METRICS
        }
        for reaction in REACTIONS:
            with open(
                CALC_PATH / f"{reaction}-{model_name}-neb-results.dat", encoding="utf8"
            ) as f:
                data = f.readlines()
                pred_barrier, pred_delta_e, pred_fmax = tuple(
                    float(x) for x in data[1].split()
                )

            data_paths = {
                "ref_profile": str(ref_cache[reaction]["traj_path"]),
                "pred_profile": str(
                    CALC_PATH / f"{reaction}-{model_name}-neb-band.extxyz"
                ),
            }

            # Store metric points
            metric_values = {
                "delta_E": (ref_data[reaction]["delta_E"], pred_delta_e),
                "barrier": (ref_data[reaction]["barrier"], pred_barrier),
                "fmax": (None, pred_fmax),
            }

            for metric_key, (ref_val, pred_val) in metric_values.items():
                if metric_key != "fmax":
                    metrics_data[metric_key]["ref"].append(ref_val)
                    metrics_data[metric_key]["pred"].append(pred_val)
                    metrics_data[metric_key]["points"].append(
                        {
                            "id": reaction,
                            "reaction": reaction,
                            "ref": ref_val,
                            "pred": pred_val,
                            "data_paths": data_paths,
                        }
                    )
                else:
                    metrics_data[metric_key]["pred"].append(pred_val)
                    metrics_data[metric_key]["points"].append(
                        {
                            "id": reaction,
                            "reaction": reaction,
                            "pred": pred_val,
                            "data_paths": data_paths,
                        }
                    )

        # Calculate MAEs
        for metric_key in METRICS:
            if metric_key != "fmax":
                ref_vals = metrics_data[metric_key]["ref"]
                pred_vals = metrics_data[metric_key]["pred"]
                metrics_data[metric_key]["mae"] = mae(ref_vals, pred_vals)

        unconverged_percentage = (
            np.sum([fmax > FMAX for fmax in metrics_data["fmax"]["pred"]])
            / len(metrics_data["fmax"]["pred"])
            * 100
        )

        stats[model_name] = {
            "model": model_name,
            "metrics": metrics_data,
            "unconverged": unconverged_percentage,
        }

    return stats


@pytest.fixture
@build_table(
    filename=OUT_PATH / "oc20neb_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(oc20neb_stats: dict[str, dict[str, float]]) -> dict[str, dict]:
    """
    Get all oc20neb metrics.

    Parameters
    ----------
    oc20neb_stats
        Barriers, reaction energies, and convergence for all reactions and all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    table_data: dict[str, dict[str, float | None]] = {}

    # Get MAE for delta_E and barrier metrics
    for metric_key in METRICS:
        if metric_key != "fmax":
            table_data[metric_key] = {
                model: oc20neb_stats[model]["metrics"][metric_key]["mae"]
                for model in MODELS
            }

    # Get unconverged percentage
    table_data["fmax"] = {
        model: oc20neb_stats[model]["unconverged"] for model in MODELS
    }
    return table_data


@pytest.fixture
@cell_to_scatter(
    filename=SCATTER_FILENAME,
    x_label="Predicted",
    y_label="Reference",
)
def interactive_dataset(oc20neb_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Generate pre-made scatter figures for each model-metric pair for the Dash app.

    Parameters
    ----------
    oc20neb_stats
        Aggregated statistics per model from ``oc20neb_stats``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dataset containing pre-generated scatter plots, MAEs,
        and stability metadata.
    """
    dataset = {
        "metrics": {
            metric: label for metric, label in METRIC_LABELS.items() if metric != "fmax"
        },
        "models": {},
    }

    for model_name, model_data in oc20neb_stats.items():
        dataset["models"][model_name] = {"metrics": {}}
        for metric_key in METRICS:
            if metric_key != "fmax":
                dataset["models"][model_name]["metrics"][metric_key] = {
                    "points": model_data["metrics"][metric_key]["points"],
                    "mae": model_data["metrics"][metric_key]["mae"],
                }
    return dataset


def test_oc20neb(metrics, interactive_dataset) -> None:
    """
    Run oc20neb test.

    Parameters
    ----------
    metrics
        All oc20neb metrics.
    interactive_dataset
        Scatter metadata produced by the ``interactive_dataset`` fixture.
    """
    return
