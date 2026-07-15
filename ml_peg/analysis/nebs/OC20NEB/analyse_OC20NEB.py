"""Analyse OC20NEB benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase.io import read, write
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "nebs" / "OC20NEB" / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / "OC20NEB"
SCATTER_FILENAME = OUT_PATH / "oc20neb_interactive.json"
REFERENCE_LABEL = "RPBE"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

METRIC_LABELS = {
    "delta_E": "Reaction Energy MAE",
    "barrier": "Barrier MAE",
    "fmax": "Convergence",
}
FMAX = 0.05


def _build_neb_profile_figure(
    *,
    ref_traj: list,
    pred_traj: list,
    model_label: str,
    system_label: str,
    reference_label: str = REFERENCE_LABEL,
) -> go.Figure:
    """
    Build an interactive Plotly figure showing DFT and MLIP NEB energy profiles.

    Each data point is individually clickable; ``pointNumber`` maps directly to
    the NEB image index in the trajectory files.

    Parameters
    ----------
    ref_traj
        DFT reference trajectory images.
    pred_traj
        Predicted (MLIP) trajectory images.
    model_label
        Display name for the predicted model trace.
    system_label
        Title displayed above the plot.
    reference_label
        Legend label for the reference (DFT) trace.

    Returns
    -------
    go.Figure
        Interactive Plotly figure showing both energy profiles.
    """
    ref_energies = np.array([at.info["DFT_energy"] for at in ref_traj])
    pred_energies = np.array([at.get_potential_energy() for at in pred_traj])

    ref_shifted = ref_energies - ref_energies[0]
    pred_shifted = pred_energies - pred_energies[0]
    image_indices = list(range(len(ref_shifted)))

    hover_ref = [
        f"<b>Image {i}</b><br>ΔE = {e:.3f} eV<br><i>Click to view geometry</i>"
        for i, e in enumerate(ref_shifted)
    ]
    hover_pred = [
        f"<b>Image {i}</b><br>ΔE = {e:.3f} eV<br><i>Click to view geometry</i>"
        for i, e in enumerate(pred_shifted)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=image_indices,
            y=ref_shifted.tolist(),
            mode="lines+markers",
            name=reference_label,
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 10, "symbol": "circle", "color": "#1f77b4"},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_ref,
        )
    )

    pred_color = "#d62728"
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pred_shifted))),
            y=pred_shifted.tolist(),
            mode="lines+markers",
            name=model_label or "MLIP",
            line={"color": pred_color, "width": 2, "dash": "dash"},
            marker={"size": 10, "symbol": "circle", "color": pred_color},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_pred,
        )
    )

    fig.update_layout(
        title={"text": system_label, "font": {"size": 15}},
        xaxis={
            "title": "NEB Image Index",
            "tickmode": "linear",
            "dtick": 1,
            "gridcolor": "#eee",
        },
        yaxis={"title": "ΔEnergy (eV)", "gridcolor": "#eee"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        clickmode="event",
    )

    return fig


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
            CALC_PATH / model / f"{reaction}-neb-band.extxyz",
            index=":",
        )
        results[model] = [
            list(range(len(structs))),
            [struct.info["mlip_energy"] for struct in structs],
        ]
        structs_dir = OUT_PATH / model
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / model / f"{reaction}-neb-band.extxyz", structs)

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
    for path in CALC_PATH.glob("*.xyz"):
        reaction = path.stem.split("-dft")[0]
        traj = read(path, ":")
        energy = np.array([at.info["DFT_energy"] for at in traj])

        ref_data[reaction] = {
            "traj": traj,
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

    dft_assets_dir = OUT_PATH / "DFT"
    dft_assets_dir.mkdir(parents=True, exist_ok=True)

    for reaction in ref_data.keys():
        ref_traj_path = CALC_PATH / f"{reaction}-dft.xyz"
        ref_profile = ref_data[reaction]["energy"]

        write(dft_assets_dir / ref_traj_path.name, ref_data[reaction]["traj"])
        ref_cache[reaction] = {
            "profile": ref_profile,
            "traj_path": ref_traj_path,
        }

    stats: dict[str, dict[str, Any]] = {}

    for model_name in MODELS:
        metrics_data: dict[str, dict[str, Any]] = {
            key: {"points": [], "ref": [], "pred": [], "mae": None}
            for key in METRIC_LABELS.keys()
        }
        model_assets_dir = OUT_PATH / model_name
        model_assets_dir.mkdir(parents=True, exist_ok=True)

        for reaction in ref_data.keys():
            with open(
                CALC_PATH / model_name / f"{reaction}-neb-results.dat", encoding="utf8"
            ) as f:
                data = f.readlines()
                pred_barrier, pred_delta_e, pred_fmax = tuple(
                    float(x) for x in data[1].split()
                )

            pred_traj_path = CALC_PATH / model_name / f"{reaction}-neb-band.extxyz"
            pred_traj = read(pred_traj_path, ":")
            write(model_assets_dir / pred_traj_path.name, pred_traj)

            data_paths = {
                "ref": f"/assets/nebs/OC20NEB/DFT/{reaction}-dft.xyz",
                "pred": f"/assets/nebs/OC20NEB/{model_name}/{pred_traj_path.name}",
            }

            profile_fig = _build_neb_profile_figure(
                ref_traj=ref_data[reaction]["traj"],
                pred_traj=pred_traj,
                model_label=model_name,
                system_label=reaction,
            )
            profile_figure = json.loads(profile_fig.to_json())

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
                            "profile_figure": profile_figure,
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
        for metric_key in METRIC_LABELS.keys():
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
    Get all OC20NEB metrics.

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
    for metric_key, label in METRIC_LABELS.items():
        if metric_key != "fmax":
            table_data[label] = {
                model: oc20neb_stats[model]["metrics"][metric_key]["mae"]
                for model in MODELS
            }

    # Get unconverged percentage
    table_data["Convergence"] = {
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
        for metric_key in METRIC_LABELS.keys():
            if metric_key != "fmax":
                dataset["models"][model_name]["metrics"][metric_key] = {
                    "points": model_data["metrics"][metric_key]["points"],
                    "mae": model_data["metrics"][metric_key]["mae"],
                }
    return dataset


def test_oc20neb(metrics, interactive_dataset) -> None:
    """
    Run OC20NEB test.

    Parameters
    ----------
    metrics
        All OC20NEB metrics.
    interactive_dataset
        Scatter metadata produced by the ``interactive_dataset`` fixture.
    """
    write_struct_info(
        data_path=list(CALC_PATH.glob("mock/*-neb-band.extxyz")),
        out_path=OUT_PATH,
        index=0,
    )
