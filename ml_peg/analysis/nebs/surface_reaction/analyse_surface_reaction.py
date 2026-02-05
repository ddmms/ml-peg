"""Analyse Li diffusion benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ase.io import read, write
import pytest
import numpy as np

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config
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
	"desorption_ood_87_9841_0_111-1" : 2.061,
	"dissociation_ood_268_6292_46_211-5" : 1.505, 
	"transfer_id_601_1482_1_211-5": 0.868,
}
REACTIONS = [
    "desorption_ood_87_9841_0_111-1",
	"dissociation_ood_268_6292_46_211-5",
	"transfer_id_601_1482_1_211-5"
]

def plot_nebs(model: str, reaction:str) -> None:
    """
    Plot NEB paths and save all structure files.

    Parameters
    ----------
    model
        Name of MLIP.
    path
        Path "b" or "c" for NEB.
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


@pytest.fixture
def energies() -> dict[str, list]:
    ref_traj = read(DATA_PATH / "transfer_id_601_1482_1_211-5.xyz", ":")
    results = {"ref": [at.info["DFT_energy"] for at in ref_traj]} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        structs = read(CALC_PATH / f"transfer_id_601_1482_1_211-5_{model_name}.xyz", ":")

        results[model_name] = [struct.info["mlip_energy"] for struct in structs] 

    return results


#@pytest.fixture
#def reaction_energy_error(energies: dict[str, list]) -> dict[str, float]:
#    """
#    Get error in path B energy barrier.
#
#    Returns
#    -------
#    dict[str, float]
#        Dictionary of predicted barrier errors for all models.
#    """
##    OUT_PATH.mkdir(parents=True, exist_ok=True)
#    results = {}
#    for model_name in MODELS:
##        plot_nebs(model_name, "transfer_id_601_1482_1_211-5")
#
#        pred_reaction_energy = energies[model_name][-1] - energies[model_name][0]
#        ref_reaction_energy = energies["ref"][-1] - energies["ref"][0]
##        pred_barrier = np.max(energy) - energy[0]
#        results[model_name] = np.abs(pred_reaction_energy - ref_reaction_energy)
#
#    return results

@pytest.fixture
def forward_barrier_error() -> dict[str, dict[str, float]]:
    """
    Get error in path B energy barrier.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier errors for all models.
    """
#    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        reaction_dict = {}
        for reaction in REACTIONS:
            plot_nebs(model_name, reaction)
            structs = read(CALC_PATH / f"{reaction}_{model_name}.xyz", ":")
            energies = [struct.info["mlip_energy"] for struct in structs]
            pred_forward_barrier = np.max(energies) - energies[0]
            reaction_dict[reaction] = np.abs(pred_forward_barrier - REF_VALUES[reaction])
        results[model_name] = reaction_dict

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "surface_reaction_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
	thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
#    reaction_energy_error: dict[str, float], forward_barrier_error: dict[str, float]
    forward_barrier_error: dict[str, dict[str, float]]
) -> dict[str, dict]:
    """
    Get all new benchmark metrics.

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
    metrics_dict = {}
    for reaction in REACTIONS:
        metrics_dict[f"{reaction} barrier error"] = {
            model : forward_barrier_error[model][reaction] for model in MODELS
		}
    return metrics_dict


def test_surface_reaction(metrics: dict[str, dict]) -> None:
    """
    Run OC157 test.

    Parameters
    ----------
    metrics
        All OC157 metrics.
    """
    return
