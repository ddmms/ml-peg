"""Analyse Li diffusion benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "nebs" / "li_diffusion" / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / "li_diffusion"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

REF_VALUES = {"path_b": 0.27, "path_c": 2.5}


def plot_nebs(model_name: str, path: Literal["b", "c"]) -> None:
    """
    Plot NEB paths and save all structure files.

    Parameters
    ----------
    model_name
        Name of MLIP.
    path
        Path "b" or "c" for NEB.
    """

    @plot_scatter(
        filename=OUT_PATH / model_name / f"figure_neb_{path.lower()}.json",
        title=f"NEB path {path.upper()}",
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
            CALC_PATH / model_name / f"li_diffusion_{path.lower()}-neb-band.extxyz",
            index=":",
        )
        results[model_name] = [
            list(range(len(structs))),
            [struct.get_potential_energy() for struct in structs],
        ]
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / f"{path.lower()}-neb-band.extxyz", structs)

        return results

    plot_neb()


@pytest.fixture
def path_b_error() -> dict[str, float]:
    """
    Get error in path B energy barrier.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier errors for all models.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        plot_nebs(model_name, "b")
        with open(
            CALC_PATH / model_name / "li_diffusion_b-neb-results.dat", encoding="utf8"
        ) as f:
            data = f.readlines()
            pred_barrier, _, _ = tuple(float(x) for x in data[1].split())
        results[model_name] = abs(REF_VALUES["path_b"] - pred_barrier)
    return results


@pytest.fixture
def path_c_error() -> dict[str, float]:
    """
    Get error in path C energy barrier.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier errors for all models.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        plot_nebs(model_name, "c")
        with open(
            CALC_PATH / model_name / "li_diffusion_c-neb-results.dat", encoding="utf8"
        ) as f:
            data = f.readlines()
            pred_barrier, _, _ = tuple(float(x) for x in data[1].split())
        results[model_name] = abs(REF_VALUES["path_c"] - pred_barrier)
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "li_diffusion_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    path_b_error: dict[str, float], path_c_error: dict[str, float]
) -> dict[str, dict]:
    """
    Get all Li diffusion metrics.

    Parameters
    ----------
    path_b_error
        Mean absolute errors for all models.
    path_c_error
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Path B error": path_b_error,
        "Path C error": path_c_error,
    }


def test_li_diffusion(metrics: dict[str, dict]) -> None:
    """
    Run Li diffusion test.

    Parameters
    ----------
    metrics
        All Li diffusion metrics.
    """
    write_struct_info(
        data_path=CALC_PATH / "mock" / "li_diffusion_b-neb-band.extxyz",
        out_path=OUT_PATH,
        index=0,
    )
