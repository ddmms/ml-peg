"""Analyse O diffusion on 2D TMDs benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "nebs" / "O_diffusion_2D_TMDs" / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / "O_diffusion_2D_TMDs"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Reference DFT-PBE barriers from Liu et al. (eV)
REF_VALUES = {
    "MoS2": 2.53,
    "MoSe2": 1.55,
    "MoTe2": 0.90,
    "WS2": 2.68,
    "WSe2": 1.66,
    "WTe2": 1.01,
}

COMPOUNDS = ["MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2"]


def plot_nebs(model: str, compound: str) -> None:
    """
    Plot NEB paths and save all structure files.

    Parameters
    ----------
    model
        Name of MLIP.
    compound
        TMD compound name.
    """

    @plot_scatter(
        filename=OUT_PATH / f"figure_{model}_O_diffusion_{compound}.json",
        title=f"O diffusion on {compound}",
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
        results: dict[str, tuple[list[float], list[float]]] = {}
        structs = read(
            CALC_PATH / f"O_diffusion_{compound}-{model}-neb-band.extxyz",
            index=":",
        )
        energies = [struct.get_potential_energy() for struct in structs]
        results[model] = (
            list(range(len(structs))),
            energies,
        )
        structs_dir = OUT_PATH / model
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / f"{model}-{compound}-neb-band.extxyz", structs)

        return results

    plot_neb()

    # Add a horizontal reference line at initial energy + ref barrier
    fig_path = OUT_PATH / f"figure_{model}_O_diffusion_{compound}.json"
    if fig_path.exists():
        with open(fig_path, encoding="utf8") as f:
            fig = json.load(f)

        structs = read(
            CALC_PATH / f"O_diffusion_{compound}-{model}-neb-band.extxyz",
            index=":",
        )
        y0 = structs[0].get_potential_energy()
        y_ref = y0 + REF_VALUES[compound]

        layout = fig.setdefault("layout", {})
        shapes = layout.setdefault("shapes", [])
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "x1": 1,
                "y0": y_ref,
                "y1": y_ref,
                "line": {"color": "red", "width": 1, "dash": "dash"},
            }
        )

        # Add legend entry for the reference line
        data = fig.setdefault("data", [])
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Reference barrier",
                "x": [0, 1],
                "y": [y_ref, y_ref],
                "line": {"color": "red", "width": 1, "dash": "dash"},
                "hoverinfo": "skip",
                "showlegend": True,
                "xaxis": "x",
                "yaxis": "y",
            }
        )

        with open(fig_path, "w", encoding="utf8") as f:
            json.dump(fig, f)


@pytest.fixture
def barrier_errors() -> dict[str, dict[str, float]]:
    """
    Get error in diffusion barriers for all compounds.

    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary of predicted barrier errors for all models and compounds.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {model: {} for model in MODELS}

    for model_name in MODELS:
        for compound in COMPOUNDS:
            plot_nebs(model_name, compound)
            with open(
                CALC_PATH / f"O_diffusion_{compound}-{model_name}-neb-results.dat",
                encoding="utf8",
            ) as f:
                data = f.readlines()
                pred_barrier, _, _ = tuple(float(x) for x in data[1].split())
            results[model_name][compound] = abs(REF_VALUES[compound] - pred_barrier)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "O_diffusion_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(barrier_errors: dict[str, dict[str, float]]) -> dict[str, dict]:
    """
    Get all O diffusion metrics.

    Parameters
    ----------
    barrier_errors
        Diffusion barrier errors for all models and compounds.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    metrics_dict: dict[str, dict[str, float]] = {}
    for compound in COMPOUNDS:
        metrics_dict[f"{compound} barrier error"] = {
            model: barrier_errors[model][compound] for model in MODELS
        }
    return metrics_dict


def test_o_diffusion(metrics: dict[str, dict]) -> None:
    """
    Run O diffusion test.

    Parameters
    ----------
    metrics
        All O diffusion metrics.
    """
    return
