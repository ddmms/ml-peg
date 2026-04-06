"""Analyse Li diffusion benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DATA_PATH = CALCS_ROOT / "nebs" / "surface_reaction" / "data"
CALC_PATH = CALCS_ROOT / "nebs" / "surface_reaction" / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / "surface_reaction"

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
            #            CALC_PATH / f"surface_reaction_{reaction}-{model}.xyz",
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
    pred_data: dict[str, dict[str, float | str]] = {}

    for model_name in MODELS:
        pred_data[model_name] = {}
        for reaction in REACTIONS:
            with open(
                CALC_PATH / f"{reaction}-{model_name}-neb-results.dat", encoding="utf8"
            ) as f:
                data = f.readlines()
                pred_barrier, pred_delta_e, pred_fmax = tuple(
                    float(x) for x in data[1].split()
                )

            pred_data[model_name][reaction] = {
                "delta_E": pred_delta_e,
                "barrier": pred_barrier,
                "fmax": pred_fmax,
            }

    stats = {}
    for model in MODELS:
        stats[model] = {}
        for metric_key in METRICS:
            if metric_key == "fmax":
                # if fmax, count how many cases are not converged.
                unconverged_percentage = (
                    np.sum(
                        [
                            values[metric_key] > 0.05
                            for reaction, values in pred_data[model].items()
                        ]
                    )
                    / len(pred_data[model])
                    * 100
                )

                stats[model][f"{metric_key}"] = unconverged_percentage
            else:
                ref_values = [
                    values[metric_key] for reaction, values in ref_data.items()
                ]
                pred_values = [
                    pred_data[model][reaction][metric_key]
                    for reaction, values in ref_data.items()
                ]
                stats[model][f"{metric_key}"] = mae(ref_values, pred_values)

    return stats


@pytest.fixture
@build_table(
    filename=OUT_PATH / "surface_reaction_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(oc20neb_stats: dict[str, dict[str, float]]) -> dict[str, dict]:
    """
    Get all surface reactions metrics.

    Parameters
    ----------
    oc20neb_stats
        Barriers, reaction energies, and convergence for all reactions and all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    table_data = {}
    for metric_key in METRICS:
        table_data[metric_key] = {
            model: oc20neb_stats[model][metric_key] for model in MODELS
        }
    return table_data


def test_surface_reaction(metrics: dict[str, dict]) -> None:
    """
    Run surface reaction test.

    Parameters
    ----------
    metrics
        All surface reaction metrics.
    """
    return
