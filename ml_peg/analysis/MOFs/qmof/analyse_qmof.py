"""Analyse qmof benchmark."""

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
CALC_PATH = CALCS_ROOT / "MOFs" / "qmof" / "outputs"
OUT_PATH = APP_ROOT / "data" / "MOFs" / "qmof"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of qmof system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_file = "qmof_valid_structures.xyz"
            mofs = read(model_dir / xyz_file, index=":")
            for mof in mofs:
                system_names.append(mof.info["qmof_id"])
            break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "qmof_energies.json",
    title="QMOF Energies",
    x_label="Predicted energy [eV]",
    y_label="Reference energy [eV]",
    hoverdata={
        "System": get_system_names(),
    },
)
def qmof_energies() -> dict[str, list]:
    """
    Get lattice energies for all qmof systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted lattice energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_file = "qmof_valid_structures.xyz"
        mofs = read(model_dir / xyz_file, index=":")
        for mof in mofs:
            mof_energy = mof.get_potential_energy()

            results[model_name].append(mof_energy)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(mof.info["dft_energy"])

        # Copy individual structure files to app data directory
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / xyz_file, mofs)

        ref_stored = True

    return results


@pytest.fixture
def qmof_errors(qmof_energies) -> dict[str, float]:
    """
    Get mean absolute error for lattice energies.

    Parameters
    ----------
    qmof_energies
        Dictionary of reference and predicted lattice energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted lattice energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if qmof_energies[model_name]:
            results[model_name] = mae(qmof_energies["ref"], qmof_energies[model_name])
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "qmof_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(qmof_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all qmof metrics.

    Parameters
    ----------
    qmof_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": qmof_errors,
    }


def test_qmof(metrics: dict[str, dict]) -> None:
    """
    Run qmof test.

    Parameters
    ----------
    metrics
        All qmof metrics.
    """
    return
