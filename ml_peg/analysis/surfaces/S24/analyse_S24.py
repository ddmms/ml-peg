"""Analyse S24 benchmark."""

from __future__ import annotations

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "surfaces" / "S24" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "S24"

S24_THRESHOLDS = {"MAE": (0.05, 0.5)}


def compute_adsorption_energy(
    surface_e: float, mol_surf_e: float, molecule_e: float
) -> float:
    """
    Compute adsorption energy.

    Parameters
    ----------
    surface_e
        Energy of the clean surface.
    mol_surf_e
        Energy of the molecule+surface system.
    molecule_e
        Energy of the isolated molecule.

    Returns
    -------
    float
        Adsorption energy.
    """
    return mol_surf_e - (surface_e + molecule_e)


def system_names() -> list:
    """
    Get list of system names.

    Returns
    -------
    list
        List of all system names.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for system_path in sorted(model_dir.glob("*.xyz")):
                mol_surface = read(system_path)
                if "system_name" in mol_surface.info:
                    system_names.append(mol_surface.info["system_name"])
            break
    return system_names


def sys_ids() -> list:
    """
    Get list of system IDs.

    Returns
    -------
    list
        List of all system IDs.
    """
    sys_ids = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for system_path in sorted(model_dir.glob("*.xyz")):
                mol_surface = read(system_path)
                if "sys_id" in mol_surface.info:
                    sys_ids.append(mol_surface.info["sys_id"])
            break
    return sys_ids


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_adsorption_energies.json",
    title="Adsorption energies",
    x_label="Predicted adsorption energy / eV",
    y_label="Reference adsorption energy / eV",
    hoverdata={
        "System": system_names(),
        "Sys ID": sys_ids(),
    },
)
def adsorption_energies() -> dict[str, list]:
    """
    Get adsorption energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted adsorption energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            results[model_name] = []
            continue

        for system_path in sorted(model_dir.glob("*.xyz")):
            mol_surface = read(system_path)

            # Get pre-calculated adsorption energies
            pred_ads_energy = mol_surface.info["adsorption_energy"]
            results[model_name].append(pred_ads_energy)

            if not ref_stored:
                ref_ads_energy = mol_surface.info["ref_adsorption_energy"]
                results["ref"].append(ref_ads_energy)

            # Write molecule-surface structure to app data
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system_path.stem}.xyz", mol_surface)

        ref_stored = True
    return results


@pytest.fixture
def s24_mae(adsorption_energies) -> dict[str, float]:
    """
    Get mean absolute error for adsorption energies.

    Parameters
    ----------
    adsorption_energies
        Dictionary of reference and predicted adsorption energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted adsorption energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            adsorption_energies["ref"], adsorption_energies[model_name]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "s24_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "MAE": "Mean Absolute Error (eV)",
    },
    thresholds=S24_THRESHOLDS,
)
def metrics(s24_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all S24 metrics.

    Parameters
    ----------
    s24_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": s24_mae,
    }


def test_s24(metrics: dict[str, dict]) -> None:
    """
    Run S24 test.

    Parameters
    ----------
    metrics
        All S24 metrics.
    """
    return
