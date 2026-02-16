"""Analyse LNCI16 benchmark."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from ase.io import read
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.element_filters import (
    build_element_set_masks,
    filter_hoverdata_dict,
    filter_results_dict,
    load_element_sets,
    write_element_sets_summary_file,
)
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "supramolecular" / "LNCI16" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "LNCI16"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)
ELEMENT_SETS_CONFIG_PATH = Path(__file__).parents[2] / "element_sets.yml"
ELEMENT_SETS = load_element_sets(ELEMENT_SETS_CONFIG_PATH)


def get_structure_info() -> list[dict[str, Any]]:
    """
    Get structure information for LNCI16 systems.

    This reads structures from the first model folder that contains data.
    The returned list is used as the single ordering for hover labels,
    filtering, and saved structure-index mapping.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per structure with keys:
        ``system``, ``atom_count``, ``charge``, ``is_charged``,
        ``elements``, and ``index``.
    """
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                structure_metadata: list[dict[str, Any]] = []
                for index, xyz_file in enumerate(xyz_files):
                    atoms = read(xyz_file)
                    charge = int(atoms.info.get("complex_charge", 0))
                    structure_metadata.append(
                        {
                            "system": atoms.info.get(
                                "system", f"system_{xyz_file.stem}"
                            ),
                            "atom_count": len(atoms),
                            "charge": charge,
                            "is_charged": charge != 0,
                            "elements": sorted(set(atoms.get_chemical_symbols())),
                            "index": index,
                        }
                    )
                return structure_metadata
    return []


def build_hoverdata_from_structure_info(
    structure_info: list[dict[str, Any]],
) -> dict[str, list]:
    """
    Build hover labels for the predicted-vs-reference scatter plot.

    Parameters
    ----------
    structure_info
        Structure information returned by :func:`get_structure_info`.

    Returns
    -------
    dict[str, list]
        Hover label columns mapped to values in structure order.
    """
    return {
        "System": [entry["system"] for entry in structure_info],
        "Elements": ["".join(entry["elements"]) for entry in structure_info],
        "Complex Atoms": [entry["atom_count"] for entry in structure_info],
        "Charge": [entry["charge"] for entry in structure_info],
        "Charged": [entry["is_charged"] for entry in structure_info],
    }


STRUCTURE_INFO = get_structure_info()
HOVERDATA = build_hoverdata_from_structure_info(STRUCTURE_INFO)


def compute_lnci16_mae(energies: dict[str, list]) -> dict[str, float | None]:
    """
    Compute mean absolute error (MAE) for each model.

    Parameters
    ----------
    energies
        Interaction energies with keys ``ref`` and one key per model.

    Returns
    -------
    dict[str, float | None]
        MAE value for each model. Returns ``None`` for models with no data.
    """
    results: dict[str, float | None] = {}
    ref_values = energies["ref"]
    for model_name in MODELS:
        model_values = energies[model_name]
        if ref_values and model_values:
            results[model_name] = mae(ref_values, model_values)
        else:
            results[model_name] = None
    return results


def write_lnci16_element_set_outputs(interaction_energies: dict[str, list]) -> None:
    """
    Write filtered LNCI16 outputs for every configured element set.

    For each element set (for example ``all`` and ``hcno``), this writes:
    1. A filtered predicted-vs-reference scatter plot JSON.
    2. A filtered metrics table JSON.
    3. A  ``element_sets.json`` file with counts and original
       structure positions.

    Parameters
    ----------
    interaction_energies
        Full LNCI16 interaction energies before filtering.
    """
    structure_elements = [set(entry["elements"]) for entry in STRUCTURE_INFO]
    element_set_masks = build_element_set_masks(structure_elements, ELEMENT_SETS)

    for element_set_key, element_set_mask in element_set_masks.items():
        filtered_results = filter_results_dict(interaction_energies, element_set_mask)
        filtered_hoverdata = filter_hoverdata_dict(HOVERDATA, element_set_mask)
        filtered_mae = compute_lnci16_mae(filtered_results)
        element_set_out_path = OUT_PATH / "element_sets" / element_set_key

        @plot_parity(
            filename=element_set_out_path / "figure_interaction_energies.json",
            title="LNCI16 Interaction Energies",
            x_label="Predicted interaction energy / kcal/mol",
            y_label="Reference interaction energy / kcal/mol",
            hoverdata=filtered_hoverdata,
        )
        def _filtered_interaction_energies(
            results: dict[str, list] = filtered_results,
        ) -> dict[str, list]:
            return results

        _filtered_interaction_energies()

        @build_table(
            filename=element_set_out_path / "lnci16_metrics_table.json",
            metric_tooltips=DEFAULT_TOOLTIPS,
            thresholds=DEFAULT_THRESHOLDS,
            weights=DEFAULT_WEIGHTS,
        )
        def _filtered_metrics(
            mae_by_model: dict[str, float | None] = filtered_mae,
        ) -> dict[str, dict]:
            return {"MAE": mae_by_model}

        _filtered_metrics()

    write_element_sets_summary_file(OUT_PATH, ELEMENT_SETS, element_set_masks)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="LNCI16 Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata=HOVERDATA,
)
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all LNCI16 systems.

    This fixture also copies structure files for the app and writes
    element-set specific outputs used by the element-set selector.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted interaction energies.
    """
    interaction_energy_results = {"ref": []} | {mlip: [] for mlip in MODELS}
    reference_is_stored = False

    for model_name in MODELS:
        model_output_dir = CALC_PATH / model_name

        if not model_output_dir.exists():
            interaction_energy_results[model_name] = []
            continue

        xyz_files = sorted(model_output_dir.glob("*.xyz"))
        if not xyz_files:
            interaction_energy_results[model_name] = []
            continue

        model_energies = []
        ref_energies = []

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            model_energies.append(atoms.info["E_int_model_kcal"])
            if not reference_is_stored:
                ref_energies.append(atoms.info["E_int_ref_kcal"])

        interaction_energy_results[model_name] = model_energies

        # Store reference energies (only once)
        if not reference_is_stored:
            interaction_energy_results["ref"] = ref_energies
            reference_is_stored = True

        # Copy individual structure files to app data directory
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)

        for i, xyz_file in enumerate(xyz_files):
            shutil.copy(xyz_file, structs_dir / f"{i}.xyz")

    write_lnci16_element_set_outputs(interaction_energy_results)
    return interaction_energy_results


@pytest.fixture
def lnci16_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for interaction energies.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for all models.
    """
    return compute_lnci16_mae(interaction_energies)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "lnci16_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(lnci16_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all LNCI16 metrics.

    Parameters
    ----------
    lnci16_mae
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": lnci16_mae,
    }


def test_lnci16(metrics: dict[str, dict]) -> None:
    """
    Run LNCI16 test.

    Parameters
    ----------
    metrics
        All LNCI16 metrics.
    """
    return
