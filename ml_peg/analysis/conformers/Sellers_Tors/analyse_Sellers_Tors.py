"""
Analyse the Sellers torsion scan benchmark for drug-like fragments.

A Comparison of Quantum and Molecular Mechanical Methods to Estimate Strain Energy in
Druglike Fragments
Benjamin D. Sellers, Natalie C. James, Alberto Gobbi
Journal of Chemical Information and Modeling 2017 57 (6), 1265-1275
DOI: 10.1021/acs.jcim.6b00614
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map()

CALC_PATH = CALCS_ROOT / "conformers" / "Sellers_Tors" / "outputs"
OUT_PATH = APP_ROOT / "data" / "conformers" / "Sellers_Tors"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


def sort_by_mol_id(path: Path) -> int:
    """
    Sort scan files numerically by fragment ID.

    Parameters
    ----------
    path
        Path to torsion scan file, named by fragment ID.

    Returns
    -------
    int
        Fragment ID.
    """
    return int(path.stem)


INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.extxyz",
    sort_key=sort_by_mol_id,
    info_keys=["mol_id", "smiles", "category", "dihedral"],
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
)

# Reference scans are model-independent, so load them once from the mock outputs
REF_SCANS = {
    path.stem: read(path, index=":")
    for path in sorted((CALC_PATH / "mock").glob("*.extxyz"), key=sort_by_mol_id)
}
SCAN_LABELS = list(REF_SCANS)
SCAN_SMILES = [scan[0].info["smiles"] for scan in REF_SCANS.values()]


def get_model_rel_energies(model_name: str) -> dict[str, np.ndarray]:
    """
    Get model relative energies for each scan, aligned to the reference minimum.

    Parameters
    ----------
    model_name
        Name of model to read outputs for.

    Returns
    -------
    dict[str, np.ndarray]
        Relative energies in kcal/mol for each scan. NaN if missing or failed.
    """
    rel_energies = {}
    for label, ref_scan in REF_SCANS.items():
        output_file = CALC_PATH / model_name / f"{label}.extxyz"
        if not output_file.exists():
            rel_energies[label] = np.full(len(ref_scan), np.nan)
            continue

        scan = read(output_file, index=":")
        energies = np.array(
            [atoms.info.get("model_energy", np.nan) for atoms in scan], dtype=float
        )
        ref_min_idx = int(np.argmin([atoms.info["ref_rel_energy"] for atoms in scan]))
        rel_energies[label] = (energies - energies[ref_min_idx]) * EV_TO_KCAL
    return rel_energies


def write_app_structs() -> None:
    """Write a single-frame structure file for each scan point for the app."""
    structs_dir = OUT_PATH / "structs"
    structs_dir.mkdir(parents=True, exist_ok=True)
    for label, scan in REF_SCANS.items():
        for i, atoms in enumerate(scan):
            write(structs_dir / f"{label}_{i:02d}.xyz", atoms)


@pytest.fixture
def model_rel_energies() -> dict[str, dict[str, np.ndarray]]:
    """
    Get model relative energies for all models and scans.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Relative energies in kcal/mol for each scan, for all models.
    """
    return {model_name: get_model_rel_energies(model_name) for model_name in MODELS}


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_sellers_tors.json",
    title="Torsion scan relative energies",
    x_label="Predicted relative energy / kcal/mol",
    y_label="Reference relative energy / kcal/mol",
    hoverdata={
        "Fragment": INFO["mol_id"],
        "SMILES": INFO["smiles"],
        "Dihedral / deg": INFO["dihedral"],
    },
    symbol_by=INFO["category"],
)
def torsion_energies(
    model_rel_energies: dict[str, dict[str, np.ndarray]],
) -> dict[str, list]:
    """
    Get relative energies for all torsion scan points.

    Parameters
    ----------
    model_rel_energies
        Relative energies for each scan, for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energies.
    """
    write_app_structs()

    results = {
        "ref": [
            atoms.info["ref_rel_energy"]
            for scan in REF_SCANS.values()
            for atoms in scan
        ]
    }
    for model_name in MODELS:
        results[model_name] = list(
            np.concatenate(
                [model_rel_energies[model_name][label] for label in SCAN_LABELS]
            )
        )
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_sellers_tors_barriers.json",
    title="Torsional barriers",
    x_label="Predicted barrier / kcal/mol",
    y_label="Reference barrier / kcal/mol",
    hoverdata={
        "Fragment": SCAN_LABELS,
        "SMILES": SCAN_SMILES,
    },
)
def torsion_barriers(
    model_rel_energies: dict[str, dict[str, np.ndarray]],
) -> dict[str, list]:
    """
    Get torsional barriers (maximum minus minimum energy) for each scan.

    Parameters
    ----------
    model_rel_energies
        Relative energies for each scan, for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted torsional barriers.
    """
    results = {
        "ref": [
            np.ptp([atoms.info["ref_rel_energy"] for atoms in scan])
            for scan in REF_SCANS.values()
        ]
    }
    for model_name in MODELS:
        # np.ptp propagates NaN from any failed scan point
        results[model_name] = [
            float(np.ptp(model_rel_energies[model_name][label]))
            for label in SCAN_LABELS
        ]
    return results


@pytest.fixture
def get_mae(torsion_energies: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for relative energies.

    Parameters
    ----------
    torsion_energies
        Dictionary of reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        Dictionary of relative energy MAEs for all models.
    """
    return {
        model_name: mae(torsion_energies["ref"], torsion_energies[model_name])
        for model_name in MODELS
    }


@pytest.fixture
def get_rmse(torsion_energies: dict[str, list]) -> dict[str, float]:
    """
    Get root mean squared error for relative energies.

    Parameters
    ----------
    torsion_energies
        Dictionary of reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        Dictionary of relative energy RMSEs for all models.
    """
    return {
        model_name: rmse(torsion_energies["ref"], torsion_energies[model_name])
        for model_name in MODELS
    }


@pytest.fixture
def get_barrier_mae(torsion_barriers: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for torsional barriers.

    Parameters
    ----------
    torsion_barriers
        Dictionary of reference and predicted torsional barriers.

    Returns
    -------
    dict[str, float]
        Dictionary of torsional barrier MAEs for all models.
    """
    return {
        model_name: mae(torsion_barriers["ref"], torsion_barriers[model_name])
        for model_name in MODELS
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "sellers_tors_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    get_mae: dict[str, float],
    get_rmse: dict[str, float],
    get_barrier_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.
    get_rmse
        Root mean squared errors for all models.
    get_barrier_mae
        Torsional barrier mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
        "RMSE": get_rmse,
        "Barrier MAE": get_barrier_mae,
    }


def test_sellers_tors(metrics: dict[str, dict]) -> None:
    """
    Run Sellers torsion scan test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
