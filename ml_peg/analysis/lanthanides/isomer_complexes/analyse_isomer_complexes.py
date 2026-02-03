"""Analyse lanthanide isomer complex benchmark."""

from __future__ import annotations

from pathlib import Path

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

# r2SCAN-3c references (kcal/mol) from Table S4 (lanthanides only)
# These are relative energies (relative to lowest energy isomer for each system)
R2SCAN_REF: dict[str, dict[str, float]] = {
    "Lu_ff6372": {"iso1": 2.15, "iso2": 12.96, "iso3": 0.00, "iso4": 2.08},
    "Ce_ff6372": {"iso1": 2.47, "iso2": 7.13, "iso3": 0.00, "iso4": 2.17},
    "Th_ff6372": {"iso1": 2.13, "iso2": 8.03, "iso3": 0.00, "iso4": 1.23},
    "Ce_1d271a": {"iso1": 0.00, "iso2": 2.20},
    "Sm_ed79e8": {"iso1": 2.99, "iso2": 0.00},
    "La_f1a50d": {"iso1": 0.00, "iso2": 3.11},
    "Ac_f1a50d": {"iso1": 0.00, "iso2": 3.52},
    "Eu_ff6372": {"iso1": 0.00, "iso2": 6.74},
    "Nd_c5f44a": {"iso1": 0.00, "iso2": 1.61},
}


def get_system_names() -> list[str]:
    """
    Get sorted list of system names.

    Returns
    -------
    list[str]
        Sorted list of system names from R2SCAN_REF.
    """
    return sorted(R2SCAN_REF.keys())


def get_reference_keys() -> list[tuple[str, str]]:
    """
    Get sorted list of (system, isomer) tuples for consistent ordering.

    Returns
    -------
    list[tuple[str, str]]
        List of (system, isomer) tuples sorted by system then isomer.
    """
    system_names = get_system_names()
    return [
        (system, isomer)
        for system in system_names
        for isomer in sorted(R2SCAN_REF[system].keys())
    ]


def get_reference_values() -> list[float]:
    """
    Get reference relative energies in sorted order.

    Returns
    -------
    list[float]
        Reference relative energies matching the order of get_reference_keys().
    """
    reference_keys = get_reference_keys()
    return [R2SCAN_REF[system][isomer] for system, isomer in reference_keys]


def build_hoverdata() -> dict[str, list[str]]:
    """
    Build hoverdata dictionary for parity plot.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with "System" and "Isomer" keys for hover information.
    """
    reference_keys = get_reference_keys()
    return {
        "System": [system for system, _ in reference_keys],
        "Isomer": [isomer for _, isomer in reference_keys],
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
    results = {"ref": get_reference_values()} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            # Model directory doesn't exist, fill with None
            results[model_name] = [None] * len(get_reference_keys())
            continue

        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)

        # Process each system separately to compute relative energies
        preds: list[float | None] = []
        for system_name in get_system_names():
            # Collect all isomers for this system
            isomer_data: dict[str, tuple[float, object]] = {}
            for isomer in sorted(R2SCAN_REF[system_name].keys()):
                xyz_path = model_dir / f"{system_name}_{isomer}.xyz"
                if xyz_path.exists():
                    atoms = read(xyz_path)
                    energy_kcal = atoms.info.get("energy_kcal")
                    if energy_kcal is not None:
                        isomer_data[isomer] = (energy_kcal, atoms)

            # Compute relative energies
            min_energy = min(energy for energy, _ in isomer_data.values())

            # Add predictions in sorted isomer order
            for isomer in sorted(R2SCAN_REF[system_name].keys()):
                if isomer in isomer_data:
                    energy_kcal, atoms = isomer_data[isomer]
                    rel_energy = energy_kcal - min_energy
                    preds.append(rel_energy)

                    # Copy structure to app directory
                    write(structs_dir / f"{system_name}_{isomer}.xyz", atoms)
                else:
                    preds.append(None)

        results[model_name] = preds

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
