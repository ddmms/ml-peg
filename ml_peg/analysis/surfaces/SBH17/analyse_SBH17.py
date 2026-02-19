"""Analyse SBH17 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "surfaces" / "SBH17" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "SBH17"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of SBH17 system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    system_names.append(atoms.info["system"])
                break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_surface_barriers.json",
    title="SBH17 dissociative chemisorption barriers",
    x_label="Predicted barrier / eV",
    y_label="Reference barrier / eV",
    hoverdata={
        "System": get_system_names(),
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
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))
        if not xyz_files:
            continue

        for xyz_file in xyz_files:
            structs = read(xyz_file, index=":")

            gp_energy = structs[0].get_potential_energy()
            system = structs[0].info["system"]
            ts_energy = structs[1].get_potential_energy()

            barrier = ts_energy - gp_energy
            results[model_name].append(barrier)

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", structs)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(structs[0].info["ref"])

        ref_stored = True

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
        if surface_barriers[model_name]:
            results[model_name] = mae(
                surface_barriers["ref"], surface_barriers[model_name]
            )
        else:
            results[model_name] = None
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
