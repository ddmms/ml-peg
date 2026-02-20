"""Analyse FE1SIA benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
# D3_MODEL_NAMES = build_d3_name_map(MODELS)
D3_MODEL_NAMES = {m: m for m in MODELS}
CALC_PATH = CALCS_ROOT / "interstitial" / "FE1SIA" / "outputs"
OUT_PATH = APP_ROOT / "data" / "interstitial" / "FE1SIA"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of FE1SIA system names.

    Returns
    -------
    list[str]
        List of system names.
    """
    system_names = []
    # Try to find names from one of the models
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            for xyz in xyz_files:
                if xyz.stem != "ref":
                    system_names.append(xyz.stem)
            if system_names:
                break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_energy.json",
    title="FE1SIA Formation Energies",
    x_label="Predicted Formation Energy / eV",
    y_label="Reference Formation Energy / eV",
    hoverdata={
        "System": get_system_names(),
    },
)
def formation_energies() -> dict[str, list]:
    """
    Get formation energies for FE1SIA systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted formation energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        # Load bulk (ref)
        # Note: We rely on calc script having produced ref.xyz
        bulk_path = model_dir / "ref.xyz"
        if not bulk_path.exists():
            # If bulk is missing, we can't compute formation energy properly
            print(f"Warning: Bulk reference not found for {model_name}")
            continue

        bulk_atoms = read(bulk_path)
        e_bulk = bulk_atoms.get_potential_energy()
        n_bulk = len(bulk_atoms)

        xyz_files = sorted(model_dir.glob("*.xyz"))

        for xyz_file in xyz_files:
            if xyz_file.name == "ref.xyz":
                continue

            atoms = read(xyz_file)
            e_config = atoms.get_potential_energy()
            n_config = len(atoms)

            # Predicted formation energy
            # E_f = E_config - (N_config / N_bulk) * E_bulk
            pred_fe = e_config - (n_config / n_bulk) * e_bulk
            # print(model_name, pred_fe)

            results[model_name].append(pred_fe)

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / xyz_file.name, atoms)

            if not ref_stored:
                results["ref"].append(atoms.info["ref"])

        ref_stored = True

    return results


@pytest.fixture
def fe_errors(formation_energies) -> dict[str, float]:
    """
    Get RMSE for formation energies.

    Parameters
    ----------
    formation_energies
        Dictionary of reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of RMSEs for all models.
    """
    results = {}
    for model_name in MODELS:
        if formation_energies.get(model_name):
            results[model_name] = rmse(
                formation_energies["ref"], formation_energies[model_name]
            )
            # print(f"FE1SIA RMSD for {model_name}: {results[model_name]:.6f} eV")
            # print(formation_energies["ref"], formation_energies[model_name])
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "fe1sia_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(fe_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all FE1SIA metrics.

    Parameters
    ----------
    fe_errors
        RMSE errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "RMSD": fe_errors,
    }


def test_fe1sia_analysis(metrics: dict[str, dict]) -> None:
    """
    Run FE1SIA analysis test.

    Parameters
    ----------
    metrics
        All FE1SIA metrics.
    """
    return
