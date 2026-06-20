"""
Analyse graphene oxide benchmark (DFT/PBE).

The calc stores per-atom formation energies (isolated-atom subtracted). Here
those values are made relative to the first structure in the dataset, so both
DFT and MLIP share a common energy zero regardless of their absolute offsets.
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import plotly.graph_objects as go
import pytest
from tqdm import tqdm

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "carbon" / "graphene_oxide" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "graphene_oxide"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

ENERGY_METRIC_KEY = "Energy MAE"
FORCE_METRIC_KEY = "Force MAE"


def _decode_config(config_type: str) -> str:
    """
    Append decoded ratio labels to a config_type string.

    Parameters
    ----------
    config_type
        Raw config type string, e.g. '0.10-0.50' or '0.10-0.50-0.30'.

    Returns
    -------
    str
        Config type with ratio labels, e.g. '0.10-0.50 (O/C=0.10, OH/O=0.50)'.
    """
    parts = config_type.split("-")
    if len(parts) == 2:
        return f"{config_type} (O/C={parts[0]}, OH/O={parts[1]})"
    if len(parts) == 3:
        return f"{config_type} (O/C={parts[0]}, OH/O={parts[1]}, edge={parts[2]})"
    return config_type


@pytest.fixture
def all_results() -> dict[str, dict]:
    """
    Load calc results for all models.

    ``ref_energy_rel`` / ``pred_energy_rel`` are per-atom formation energies
    (isolated-atom subtracted) written by the calc. Subtracting the first
    structure's value here gives a common zero for both DFT and MLIP,
    cancelling any absolute energy offset between the two methods. The first
    structure is excluded from the output since its relative energy is zero.

    Returns
    -------
    dict[str, dict]
        Model -> energy, force, and metadata lists for plotting and metrics.
    """
    data: dict[str, dict] = {
        model: {
            "ref": [],
            "pred": [],
            "config_type": [],
            "force_mae": [],
            "force_abs_error_sum": 0.0,
            "force_component_count": 0,
            "structure_idx": [],
        }
        for model in MODELS
    }

    for model in MODELS:
        struct_dir = OUT_PATH / model
        struct_dir.mkdir(parents=True, exist_ok=True)
        atoms_list = read(CALC_PATH / model / "results.xyz", ":")
        atoms_0 = atoms_list[0]
        for idx, atoms in enumerate(tqdm(atoms_list, desc=model)):
            if idx == 0:
                continue
            data[model]["ref"].append(
                atoms.info["ref_energy_rel"] - atoms_0.info["ref_energy_rel"]
            )
            data[model]["pred"].append(
                atoms.info["pred_energy_rel"] - atoms_0.info["pred_energy_rel"]
            )
            data[model]["config_type"].append(atoms.info.get("config_type", "unknown"))
            data[model]["structure_idx"].append(idx)

            ref_forces = np.asarray(atoms.arrays["ref_forces"], dtype=float)
            pred_forces = np.asarray(atoms.arrays["pred_forces"], dtype=float)
            force_abs_error = np.abs(pred_forces - ref_forces)
            data[model]["force_mae"].append(float(np.mean(force_abs_error)))
            data[model]["force_abs_error_sum"] += float(np.sum(force_abs_error))
            data[model]["force_component_count"] += int(force_abs_error.size)

            write(struct_dir / f"{idx}.xyz", atoms)

    return data


@pytest.fixture
@cell_to_scatter(
    filename=OUT_PATH / "graphene_oxide_scatter.json",
    x_label="Predicted energy / eV atom⁻¹",
    y_label="Reference energy / eV atom⁻¹",
)
def interactive_dataset(all_results: dict) -> dict:
    """
    Build cell_to_scatter dataset with config_type as hover label.

    Parameters
    ----------
    all_results
        Model -> energy, force, and metadata lists from all_results fixture.

    Returns
    -------
    dict
        Data bundle consumed by cell_to_scatter decorator.
    """
    dataset: dict = {
        "metrics": {ENERGY_METRIC_KEY: ENERGY_METRIC_KEY},
        "models": {},
    }
    for model in MODELS:
        d = all_results[model]
        points = [
            {"id": _decode_config(ct), "ref": r, "pred": p}
            for ct, r, p in zip(d["config_type"], d["ref"], d["pred"], strict=True)
        ]
        dataset["models"][model] = {"metrics": {ENERGY_METRIC_KEY: {"points": points}}}
    return dataset


@pytest.fixture
def force_error_plots(all_results: dict) -> dict[str, Path]:
    """
    Write per-structure force error plots for each model.

    Parameters
    ----------
    all_results
        Model -> energy, force, and metadata lists from all_results fixture.

    Returns
    -------
    dict[str, Path]
        Model -> written Plotly JSON path.
    """
    paths: dict[str, Path] = {}
    for model in MODELS:
        d = all_results[model]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=d["structure_idx"],
                y=d["force_mae"],
                mode="markers",
                customdata=[[_decode_config(ct)] for ct in d["config_type"]],
                hovertemplate=(
                    "<b>Structure: </b>%{x}<br>"
                    "<b>Force MAE: </b>%{y:.4f} eV/Å<br>"
                    "<b>Config: </b>%{customdata[0]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.update_layout(
            title={"text": f"{model} - {FORCE_METRIC_KEY}"},
            xaxis={"title": {"text": "Structure index"}},
            yaxis={"title": {"text": "Mean absolute force error / eV/Å"}},
        )
        path = OUT_PATH / f"figure_{model}_force_mae.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_json(path)
        paths[model] = path
    return paths


@pytest.fixture
@build_table(
    filename=OUT_PATH / "graphene_oxide_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(all_results: dict) -> dict:
    """
    Compute overall energy and force MAE per model.

    Parameters
    ----------
    all_results
        Model -> energy, force, and metadata lists from all_results fixture.

    Returns
    -------
    dict
        Metric name -> model -> MAE value.
    """
    return {
        ENERGY_METRIC_KEY: {
            model: mae(all_results[model]["ref"], all_results[model]["pred"])
            for model in MODELS
        },
        FORCE_METRIC_KEY: {
            model: (
                all_results[model]["force_abs_error_sum"]
                / all_results[model]["force_component_count"]
            )
            for model in MODELS
        },
    }


def test_graphene_oxide(
    interactive_dataset: dict,
    force_error_plots: dict[str, Path],
    metrics: dict,
) -> None:
    """
    Run graphene oxide analysis (drives all fixtures).

    Parameters
    ----------
    interactive_dataset
        Pre-generated energy scatter figures from cell_to_scatter fixture.
    force_error_plots
        Pre-generated force error plot paths.
    metrics
        Energy and force MAE per model from metrics fixture.
    """
    return
