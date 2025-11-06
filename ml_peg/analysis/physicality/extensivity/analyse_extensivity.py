"""Analyse extensivity benchmark."""

from __future__ import annotations

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "extensivity" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "extensivity"

DEFAULT_THRESHOLDS = {
    "ΔE": (0, 10.0),
}


@pytest.fixture
def energy_difference() -> dict[str, float]:
    """
    Get difference in energies between combined and individual slabs.

    Returns
    -------
    dict[str, float]
        Dictionary of absolute energy differences for all models.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        slab_1, slab_2, combined = read(CALC_PATH / model_name / "slabs.xyz", index=":")

        energy_1 = slab_1.get_potential_energy()
        energy_2 = slab_2.get_potential_energy()
        energy_combined = combined.get_potential_energy()

        results[model_name] = 1000 * abs(energy_combined - (energy_1 + energy_2))

        # Write structures in order as glob is unsorted
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / "slabs.xyz", combined)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "extensivity_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "ΔE": "Absolute energy difference between isolated and combined slab systems "
        "(meV)",
    },
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(energy_difference: dict[str, float]) -> dict[str, dict]:
    """
    Get all extensivity metrics.

    Parameters
    ----------
    energy_difference
        Dictionary of absolute energy differences for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "ΔE": energy_difference,
    }


def test_extensivity(metrics: dict[str, dict]) -> None:
    """
    Run extensivity analysis.

    Parameters
    ----------
    metrics
        All extensivity atoms metrics.
    """
    return
