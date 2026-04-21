"""Analyse Defectstab benchmark."""

from __future__ import annotations

import collections
from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "defect" / "Defectstab" / "outputs"
OUT_PATH = APP_ROOT / "data" / "defect" / "Defectstab"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of Defectstab system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for xyz_file in sorted(model_dir.glob("*.xyz")):
                atoms = read(xyz_file)
                if atoms.info.get("ref") is not None:
                    system_names.append(atoms.info.get("system", xyz_file.stem))
            if system_names:
                break
    return system_names


def get_subset_labels() -> list[str]:
    """
    Get subset label for each system, matching the order from get_system_names.

    Returns
    -------
    list[str]
        List of subset labels, one per system.
    """
    subset_labels = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            for xyz_file in sorted(model_dir.glob("*.xyz")):
                atoms = read(xyz_file)
                if atoms.info.get("ref") is not None:
                    subset_labels.append(atoms.info.get("subset", "unknown"))
            if subset_labels:
                break
    return subset_labels


def _compute_pred_fe(atoms) -> float | None:
    """
    Compute predicted formation energy from stored info.

    Parameters
    ----------
    atoms
        ASE Atoms object with info dict from calc output.

    Returns
    -------
    float or None
        Predicted formation energy, or None if missing data.
    """
    subset = atoms.info.get("subset", "")
    e_pred = atoms.info.get("energy_pred")

    if e_pred is None:
        return None

    if subset == "fe_sia":
        # E_f = E_config - (N_config / N_bulk) * E_bulk
        e_bulk_pred = atoms.info.get("e_bulk_pred")
        n_bulk = atoms.info.get("n_bulk")
        if e_bulk_pred is None or n_bulk is None:
            return None
        n_config = len(atoms)
        return e_pred - (n_config / n_bulk) * e_bulk_pred

    if subset == "boroncarbide_stoichiometry":
        # E_f = E - nB * E_B - nC * E_C
        eb_pred = atoms.info.get("E_B_pred")
        ec_pred = atoms.info.get("E_C_pred")
        if eb_pred is None or ec_pred is None:
            return None
        syms = atoms.get_chemical_symbols()
        n_b = syms.count("B")
        n_c = syms.count("C")
        return e_pred - n_b * eb_pred - n_c * ec_pred

    if subset == "boroncarbide_defects":
        e_nd_pred = atoms.info.get("e_nodefect_pred")
        if e_nd_pred is None:
            return None
        system = atoms.info.get("system", "")
        if "Bipolar" in system:
            # E_f = E - E_NoDefects
            return e_pred - e_nd_pred
        if "VB0" in system:
            # E_f = E - E_NoDefects + muB (B-rich)
            eb_pred = atoms.info.get("E_B_pred")
            if eb_pred is None:
                return None
            return e_pred - e_nd_pred + eb_pred
        return None

    if subset == "mapi_tetragonal":
        # Only VMAI has ref
        e_mai_pred = atoms.info.get("e_mai_pred")
        e_pris_pred = atoms.info.get("e_pris_pred")
        if e_mai_pred is None or e_pris_pred is None:
            return None
        # E_f = E(VMAI) + 0.5*E(MAI) - E(pristine)
        return e_pred + 0.5 * e_mai_pred - e_pris_pred

    return None


@pytest.fixture
def grouped_data() -> dict[str, dict[str, list[dict]]]:
    """
    Get data grouped by model and subset.

    Returns
    -------
    dict
        Data grouped by model and subset.
    """
    results = {mlip: {} for mlip in MODELS}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            ref_val = atoms.info.get("ref")

            # Skip files without a reference value
            if ref_val is None:
                continue

            subset_name = atoms.info.get("subset", "unknown")
            pred_fe = _compute_pred_fe(atoms)

            if subset_name not in results[model_name]:
                results[model_name][subset_name] = []

            results[model_name][subset_name].append(
                {
                    "name": xyz_file.stem,
                    "atoms": atoms,
                    "pred_fe": pred_fe,
                    "ref": ref_val,
                }
            )

            # Copy structure to output for App
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / xyz_file.name, atoms)

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_energy.json",
    title="Defectstab Formation Energies",
    x_label="Predicted Formation Energy / eV",
    y_label="Reference Formation Energy / eV",
    hoverdata={
        "System": get_system_names(),
        "Subset": get_subset_labels(),
    },
)
def formation_energies(grouped_data) -> dict[str, list]:
    """
    Get formation energies for Defectstab systems, flattened for parity plot.

    Parameters
    ----------
    grouped_data
        Dictionary of data grouped by model and subset.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted formation energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    # Gather all subset names across models
    all_subsets = set()
    for m in MODELS:
        all_subsets.update(grouped_data[m].keys())
    sorted_subsets = sorted(all_subsets)

    ref_populated = False

    for model_name in MODELS:
        model_preds = []
        model_refs = []

        for subset in sorted_subsets:
            entries = grouped_data[model_name].get(subset, [])
            for entry in entries:
                model_preds.append(entry["pred_fe"])
                if not ref_populated:
                    model_refs.append(entry["ref"])

        if not model_preds:
            continue

        results[model_name] = model_preds

        if not ref_populated and model_refs:
            results["ref"] = model_refs
            ref_populated = True

    return results


@pytest.fixture
def fe_errors(
    grouped_data,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Compute RMSD per subset and averaged over all subsets.

    Parameters
    ----------
    grouped_data
        Dictionary of data grouped by model and subset.

    Returns
    -------
    tuple
        (total_results, subset_results) where total_results is the
        average RMSD across subsets and subset_results is the per-subset
        RMSD for each model.
    """
    total_results = {}
    subset_results = collections.defaultdict(dict)

    for model_name in MODELS:
        subsets = grouped_data[model_name]
        if not subsets:
            total_results[model_name] = None
            continue

        subset_rmsds = []

        for subset_name, entries in subsets.items():
            if not entries:
                continue

            ref = np.array([d["ref"] for d in entries], dtype=float)
            pred = np.array([d["pred_fe"] for d in entries], dtype=float)

            mask = np.isfinite(ref) & np.isfinite(pred)
            if np.any(mask):
                subset_rmsd = rmse(ref[mask], pred[mask])
            else:
                subset_rmsd = None

            subset_results[subset_name][model_name] = subset_rmsd

            if subset_rmsd is not None:
                subset_rmsds.append(subset_rmsd)

        if subset_rmsds:
            total_results[model_name] = np.mean(subset_rmsds)
        else:
            total_results[model_name] = None

    return total_results, dict(subset_results)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "defectstab_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    fe_errors: tuple[dict[str, float], dict[str, dict[str, float]]],
) -> dict[str, dict]:
    """
    Get all Defectstab metrics.

    Returns total RMSD and per-subset RMSD as separate columns.

    Parameters
    ----------
    fe_errors
        Tuple of (total_results, subset_results).

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    total_results, subset_results = fe_errors
    return {"RMSD": total_results} | subset_results


def test_defectstab_analysis(
    metrics: dict[str, dict],
    formation_energies: dict[str, list],
) -> None:
    """
    Run Defectstab analysis test.

    Parameters
    ----------
    metrics
        All Defectstab metrics.
    formation_energies
        Parity plot data for formation energies.
    """
    return
