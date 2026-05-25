"""Analyse the Al-Cu-Mg-Zn metallurgy regression benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "alloy_metallurgy" / "alzncumg_regression" / "outputs"
DATA_PATH = CALCS_ROOT / "alloy_metallurgy" / "alzncumg_regression" / "data"
REFERENCE_PATH = DATA_PATH / "references" / "DFT.json"
OUT_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "alzncumg_regression"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def load_references() -> dict[str, Any]:
    """
    Load evalpot-format DFT references.

    Returns
    -------
    dict[str, Any]
        Reference data keyed by evalpot material parameter name.
    """
    with open(REFERENCE_PATH) as file:
        return json.load(file)


def reference_value(
    references: dict[str, Any], oqmd_id: str, property_name: str
) -> float | None:
    """
    Extract one scalar reference value.

    Parameters
    ----------
    references
        Evalpot reference dictionary.
    oqmd_id
        OQMD identifier without the ``OQMD_`` prefix.
    property_name
        Evalpot property suffix.

    Returns
    -------
    float | None
        Scalar value if present and numeric.
    """
    entry = references.get(f"{oqmd_id}-{property_name}")
    if not entry:
        return None
    try:
        return float(entry[0])
    except (TypeError, ValueError):
        return None


def load_model_records() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load calculated scalar records.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Records keyed by model name and OQMD ID.
    """
    records_by_model = {}
    for model_name in MODELS:
        record_path = CALC_PATH / model_name / "bulk_properties.json"
        if not record_path.exists():
            continue
        with open(record_path) as file:
            data = json.load(file)
        records_by_model[model_name] = {
            record["oqmd_id"]: record for record in data["structures"]
        }
    return records_by_model


def common_structure_ids(
    records_by_model: dict[str, dict[str, dict[str, Any]]],
    property_name: str,
) -> list[str]:
    """
    Get structure IDs present in all model outputs and reference data.

    Parameters
    ----------
    records_by_model
        Calculated records keyed by model.
    property_name
        Evalpot reference property suffix.

    Returns
    -------
    list[str]
        Common OQMD IDs.
    """
    references = load_references()
    if not records_by_model:
        return []

    common_ids = set.intersection(
        *(set(records) for records in records_by_model.values())
    )
    return sorted(
        oqmd_id
        for oqmd_id in common_ids
        if reference_value(references, oqmd_id, property_name) is not None
    )


def get_structure_ids() -> list[str]:
    """
    Get structure IDs used by the formation-energy parity plot.

    Returns
    -------
    list[str]
        OQMD IDs available across the current outputs.
    """
    return common_structure_ids(load_model_records(), "formation_energy")


STRUCTURE_IDS = get_structure_ids()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_formation_energy.json",
    title="Formation energies",
    x_label="Predicted formation energy / eV atom^-1",
    y_label="DFT formation energy / eV atom^-1",
    hoverdata={"OQMD ID": STRUCTURE_IDS},
)
def formation_energies() -> dict[str, list[float]]:
    """
    Get reference and predicted formation energies.

    Returns
    -------
    dict[str, list[float]]
        Reference and model formation energies in eV/atom.
    """
    references = load_references()
    records_by_model = load_model_records()
    structure_ids = common_structure_ids(records_by_model, "formation_energy")

    results = {"ref": []} | {model_name: [] for model_name in records_by_model}
    for oqmd_id in structure_ids:
        ref_value = reference_value(references, oqmd_id, "formation_energy")
        if ref_value is None:
            continue
        results["ref"].append(ref_value)
        for model_name, model_records in records_by_model.items():
            results[model_name].append(model_records[oqmd_id]["formation_energy"])

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_volume_peratom.json",
    title="Volumes per atom",
    x_label="Predicted volume / Angstrom^3 atom^-1",
    y_label="DFT volume / Angstrom^3 atom^-1",
    hoverdata={"OQMD ID": STRUCTURE_IDS},
)
def volumes_per_atom() -> dict[str, list[float]]:
    """
    Get reference and predicted volumes per atom.

    Returns
    -------
    dict[str, list[float]]
        Reference and model volumes in Angstrom^3/atom.
    """
    references = load_references()
    records_by_model = load_model_records()
    structure_ids = common_structure_ids(records_by_model, "volume_peratom")

    results = {"ref": []} | {model_name: [] for model_name in records_by_model}
    for oqmd_id in structure_ids:
        ref_value = reference_value(references, oqmd_id, "volume_peratom")
        if ref_value is None:
            continue
        results["ref"].append(ref_value)
        for model_name, model_records in records_by_model.items():
            results[model_name].append(model_records[oqmd_id]["volume_peratom"])

    return results


@pytest.fixture
def formation_energy_errors(
    formation_energies: dict[str, list[float]],
) -> dict[str, float]:
    """
    Get formation-energy mean absolute errors.

    Parameters
    ----------
    formation_energies
        Reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        MAE by model.
    """
    return {
        model_name: mae(formation_energies["ref"], values)
        for model_name, values in formation_energies.items()
        if model_name != "ref"
    }


@pytest.fixture
def volume_errors(volumes_per_atom: dict[str, list[float]]) -> dict[str, float]:
    """
    Get volume-per-atom mean absolute errors.

    Parameters
    ----------
    volumes_per_atom
        Reference and predicted volumes.

    Returns
    -------
    dict[str, float]
        MAE by model.
    """
    return {
        model_name: mae(volumes_per_atom["ref"], values)
        for model_name, values in volumes_per_atom.items()
        if model_name != "ref"
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "alzncumg_regression_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    formation_energy_errors: dict[str, float],
    volume_errors: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Get all first-slice benchmark metrics.

    Parameters
    ----------
    formation_energy_errors
        Formation-energy MAEs.
    volume_errors
        Volume-per-atom MAEs.

    Returns
    -------
    dict[str, dict[str, float]]
        Metrics by model.
    """
    copy_structures_to_app_data()
    return {
        "Formation Energy MAE": formation_energy_errors,
        "Volume MAE": volume_errors,
    }


def copy_structures_to_app_data() -> None:
    """Copy calculated structures into the app data directory."""
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue
        output_dir = OUT_PATH / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for structure_path in sorted(model_dir.glob("OQMD_*.xyz")):
            atoms = read(structure_path)
            write(output_dir / structure_path.name, atoms)


def test_alzncumg_regression(metrics: dict[str, dict[str, float]]) -> None:
    """
    Run Al-Cu-Mg-Zn metallurgy regression analysis.

    Parameters
    ----------
    metrics
        First-slice analysis metrics.
    """
    return
