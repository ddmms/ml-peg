"""Analyse SBH17 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_dispersion_name_map()
CALC_PATH = CALCS_ROOT / "surfaces" / "SBH17" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "SBH17"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Extract system metadata from mock calculation
SYSTEM_INFO = get_struct_info(
    calc_path=CALC_PATH,
    include_filenames=True,
    per_file_info={
        "system": lambda structs: structs[0].info["system"],
        "ref": lambda structs: structs[0].info["ref"],
    },
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)

# Reference values are taken from the mock calculation, which is always complete
REF_BARRIERS = SYSTEM_INFO["ref"]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_surface_barriers.json",
    title="SBH17 dissociative chemisorption barriers",
    x_label="Predicted barrier / eV",
    y_label="Reference barrier / eV",
    hoverdata={
        "System": SYSTEM_INFO["system"],
    },
)
def surface_barriers() -> dict[str, list]:
    """
    Get barriers for all SBH17 systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted surface barriers.
    """
    results = {"ref": list(REF_BARRIERS)} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        for filename in SYSTEM_INFO["filenames"]:
            xyz_file = model_dir / f"{filename}.xyz"
            if not xyz_file.exists():
                results[model_name].append(float("nan"))
                continue

            structs = read(xyz_file, index=":")

            gp_energy = structs[0].get_potential_energy()
            ts_energy = structs[1].get_potential_energy()

            results[model_name].append(ts_energy - gp_energy)

    return results


@pytest.fixture
def sbh17_errors(surface_barriers) -> dict[str, float]:
    """
    Get mean absolute error for surface barriers.

    Parameters
    ----------
    surface_barriers
        Dictionary of reference and predicted surface barriers.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(surface_barriers["ref"], surface_barriers[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "SBH17_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    # mlip_name_map=D3_MODEL_NAMES,
)
def metrics(sbh17_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all SBH17 metrics.

    Parameters
    ----------
    sbh17_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": sbh17_errors,
    }


def test_sbh17(metrics: dict[str, dict]) -> None:
    """
    Run SBH17 test.

    Parameters
    ----------
    metrics
        All SBH17 metrics.
    """
    return
