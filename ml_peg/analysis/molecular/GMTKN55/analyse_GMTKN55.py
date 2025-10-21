"""Analyse GMTKN55 benchmark."""

from __future__ import annotations

from ase.io import read
from ase.units import kcal, mol
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "molecular" / "GMTKN55" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "GMTKN55"


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_rel_energies.json",
    title="Relative energies",
    x_label="Predicted relative energy / kcal/mol",
    y_label="Reference relative energy / kcal/mol",
)
def rel_energies():
    """
    Calculate relative energies for all subsets, between each set of systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False
    for model_name in MODELS:
        for subset in [dir.name for dir in (CALC_PATH / model_name).glob("*")]:
            for system_path in (CALC_PATH / model_name / subset).glob("*.xyz"):
                structs = read(system_path, index=":")
                comp_value = 0
                for struct in structs:
                    comp_value += (
                        struct.get_potential_energy()
                        * struct.info["count"]
                        * mol
                        / kcal
                    )
            if not ref_stored:
                results["ref"].append(struct.info["ref_value"])

            results[model_name].append(comp_value)

        ref_stored = True
    return results


@pytest.fixture
def rel_energy_mae(rel_energies):
    """
    Calculate MAE for all models.

    Parameters
    ----------
    rel_energies
        Relative errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(rel_energies["ref"], rel_energies[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "gmtkn55_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "MAE": "Mean Absolute Error (kcal/mol)",
    },
)
def metrics(rel_energy_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all GMTKN55 metrics.

    Parameters
    ----------
    rel_energy_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Relative Energy Error": rel_energy_mae,
    }


def test_gmtkn55(metrics):
    """
    Run GMTKN55 test.

    Parameters
    ----------
    metrics
        All GMTKN55 metrics.
    """
    return
