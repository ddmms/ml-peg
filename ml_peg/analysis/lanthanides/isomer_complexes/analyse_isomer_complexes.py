"""Analyse lanthanide isomer complex benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "lanthanides" / "isomer_complexes" / "outputs"
OUT_PATH = APP_ROOT / "data" / "lanthanides" / "isomer_complexes"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


def get_system_names() -> list[str]:
    """
    Get sorted list of system names.

    Returns
    -------
    list[str]
        Sorted list of system names.
    """
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            # Get unique labels without _iso1.xyz suffix
            return sorted(
                {path.stem.split("_iso")[0] for path in model_dir.glob("*.xyz")}
            )

    return []


def get_labels() -> list[tuple[str, str]]:
    """
    Get sorted list of (system, isomer) tuples for consistent ordering.

    Returns
    -------
    list[tuple[str, str]]
        List of (system, isomer) tuples sorted by system then isomer.
    """
    labels = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for system in get_system_names():
                labels[system] = sorted(
                    [
                        path.stem.split("_")[2]
                        for path in model_dir.glob(f"{system}_iso*.xyz")
                    ]
                )
    return labels


def build_hoverdata() -> dict[str, list[str]]:
    """
    Build hoverdata dictionary for parity plot.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with "System" and "Isomer" keys for hover information.
    """
    labels = get_labels()
    return {
        "System": [key for key, values in labels.items() for _ in values],
        "Isomer": [value for values in labels.values() for value in values],
    }


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_isomer_complexes.json",
    title="Lanthanide isomer relative energies",
    x_label="Model Delta E (kcal/mol)",
    y_label="r2SCAN-3c Delta E (kcal/mol)",
    hoverdata=build_hoverdata(),
)
def isomer_relative_energies() -> dict[str, list]:
    """
    Build parity data for lanthanide isomer complexes benchmark.

    Returns
    -------
    dict[str, list]
        Reference and per-model relative energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            # Model directory doesn't exist, fill with None
            results[model_name] = [None] * len(get_labels())
            continue

        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)

        labels = get_labels()
        for system in labels:
            pred_energies = []
            ref_energies = []
            for isomer in labels[system]:
                xyz_path = model_dir / f"{system}_{isomer}.xyz"
                atoms = read(xyz_path)

                pred_energies.append(atoms.info.get("model_energy") * EV_TO_KCAL)
                ref_energies.append(atoms.info.get("ref_energy") * EV_TO_KCAL)

                # Copy structure to app directory
                write(structs_dir / f"{system}_{isomer}.xyz", atoms)

            # Compute relative energies
            min_pred_energy = min(pred_energies)
            pred_energies = [energy - min_pred_energy for energy in pred_energies]
            results[model_name].extend(pred_energies)

            if not ref_stored:
                min_ref_energy = min(ref_energies)
                ref_energies = [energy - min_ref_energy for energy in ref_energies]
                results["ref"].extend(ref_energies)

        ref_stored = True

    return results


@pytest.fixture
def isomer_complex_errors(isomer_relative_energies) -> dict[str, float | None]:
    """
    Get mean absolute error for relative energies.

    Parameters
    ----------
    isomer_relative_energies
        Dictionary of reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted relative energy errors for all models.
    """
    results: dict[str, float | None] = {}
    for model_name in MODELS:
        preds = isomer_relative_energies.get(model_name, [])
        pairs = [
            (ref, pred)
            for ref, pred in zip(isomer_relative_energies["ref"], preds, strict=True)
            if pred is not None
        ]
        if not pairs:
            results[model_name] = None
            continue
        ref_vals, pred_vals = zip(*pairs, strict=True)
        results[model_name] = mae(list(ref_vals), list(pred_vals))
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "isomer_complexes_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(isomer_complex_errors: dict[str, float | None]) -> dict[str, dict]:
    """
    Collect metrics for lanthanide isomer complexes.

    Parameters
    ----------
    isomer_complex_errors
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metrics keyed by name for all models.
    """
    return {"MAE": isomer_complex_errors}


def test_isomer_complexes(metrics: dict[str, dict]) -> None:
    """
    Run lanthanide isomer complexes benchmark analysis.

    Parameters
    ----------
    metrics
        All lanthanide isomer complex metrics.
    """
    return
