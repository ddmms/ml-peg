"""Validate local AM26 outputs and produce metrics, diagnostics, and element info."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
import numpy as np
import orjson

from ml_peg.analysis.amorphous_materials.AM26.heatmaps import write_heatmaps
from ml_peg.analysis.amorphous_materials.AM26.metrics import (
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    calculate_am26_metrics,
    reference_normalisers,
)
from ml_peg.analysis.utils.decorators import build_table, merge_saved_models
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.amorphous_materials.AM26.calc_AM26 import load_reference_structures
from ml_peg.calcs.amorphous_materials.AM26.data import (
    REFERENCE_SHA256,
    get_reference_path,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "amorphous_materials" / "AM26" / "outputs"
OUT_PATH = APP_ROOT / "data" / "amorphous_materials" / "AM26"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)
METRIC_NAMES = tuple(DEFAULT_THRESHOLDS)


def reference_summary(references: Sequence[Atoms]) -> dict[str, dict[str, Any]]:
    """
    Build canonical system counts and reference normalisers using Phase 1.

    Parameters
    ----------
    references
        Canonical DFT frames, loaded independently of model availability.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-system expected structure count and native-unit reference deviations.
    """
    summary = {}
    for system in SYSTEM_ORDER:
        frames = [a for a in references if a.info["system"] == system]
        normalisers = reference_normalisers(
            [a.get_potential_energy() for a in frames],
            [len(a) for a in frames],
            [a.get_forces() for a in frames],
        )
        summary[system] = {
            "expected_structure_count": len(frames),
            "energy_std": normalisers["energy"],
            "force_std": normalisers["force"],
        }
    return summary


def validate_alignment(references: Sequence[Atoms], outputs: Sequence[Atoms]) -> None:
    """
    Require a complete, unchanged, ordered match to the canonical reference.

    Parameters
    ----------
    references
        Canonical reference frames in input order.
    outputs
        Phase 2 output frames, which are never reordered or reduced here.

    Raises
    ------
    ValueError
        If any identity, geometry, preserved reference, or schema check fails.
    """
    if len(outputs) != len(references):
        raise ValueError(f"Expected {len(references)} frames, found {len(outputs)}.")
    labels = [a.info["label"] for a in outputs]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate output labels.")
    for index, (reference, output) in enumerate(zip(references, outputs, strict=True)):
        input_index = output.info["input_index"]
        if (
            isinstance(input_index, (bool, np.bool_))
            or not isinstance(input_index, (int, np.integer))
            or input_index != index
        ):
            raise ValueError(f"Frame {index}: input_index is missing or out of order.")
        for key in ("label", "system", "energy_per_at", "virial", "cell_origin"):
            if not np.array_equal(output.info[key], reference.info[key]):
                raise ValueError(f"Frame {index}: {key} differs from the reference.")
        if len(output) != len(reference):
            raise ValueError(f"Frame {index}: atom count differs from the reference.")
        for key in ("numbers", "positions"):
            if not np.array_equal(output.arrays[key], reference.arrays[key]):
                raise ValueError(f"Frame {index}: {key} differs from the reference.")
        for key in ("cell", "pbc"):
            if not np.array_equal(getattr(output, key), getattr(reference, key)):
                raise ValueError(f"Frame {index}: {key} differs from the reference.")
        if output.info["ref_energy"] != reference.get_potential_energy():
            raise ValueError(f"Frame {index}: preserved DFT total energy differs.")
        if output.info["ref_energy_per_atom"] != reference.info["energy_per_at"]:
            raise ValueError(f"Frame {index}: preserved DFT energy per atom differs.")
        if not np.array_equal(output.arrays["ref_forces"], reference.get_forces()):
            raise ValueError(f"Frame {index}: preserved DFT force values/shape differ.")
        for key in ("energy_failed", "forces_failed"):
            if not isinstance(output.info[key], (bool, np.bool_)):
                raise ValueError(f"Frame {index}: {key} must be a boolean.")
        if output.calc is not None and not isinstance(
            output.calc, SinglePointCalculator
        ):
            raise ValueError("Analysis accepts stored single-point results only.")


def analyse_output(
    references: Sequence[Atoms], outputs: Sequence[Atoms]
) -> dict[str, Any]:
    """
    Validate one model's frames and calculate approved metrics without inference.

    Parameters
    ----------
    references
        Independent canonical DFT frames.
    outputs
        Phase 2 frames with reference fields and stored model predictions.

    Returns
    -------
    dict[str, Any]
        Aligned per-system diagnostics, independent property counts, and means.
    """
    validate_alignment(references, outputs)
    energies, forces = [], []
    energy_valid, force_valid = [], []
    for reference, output in zip(references, outputs, strict=True):
        results = output.calc.results if output.calc is not None else {}
        try:
            energy = float(results.get("energy", np.nan))
        except (TypeError, ValueError):
            energy = np.nan
        valid_energy = not output.info["energy_failed"] and np.isfinite(energy)
        try:
            force = np.asarray(results.get("forces", []), dtype=float)
        except (TypeError, ValueError):
            force = np.empty((0, 3))
        valid_force = (
            not output.info["forces_failed"]
            and force.shape == (len(reference), 3)
            and np.isfinite(force).all()
        )
        energies.append(energy if valid_energy else np.nan)
        forces.append(force if valid_force else np.full((len(reference), 3), np.nan))
        energy_valid.append(bool(valid_energy))
        force_valid.append(bool(valid_force))

    systems = [a.info["system"] for a in references]
    system_metrics, overall = calculate_am26_metrics(
        systems,
        [a.get_potential_energy() for a in references],
        energies,
        [len(a) for a in references],
        [a.get_forces() for a in references],
        forces,
    )
    for system in SYSTEM_ORDER:
        indices = [i for i, name in enumerate(systems) if name == system]
        valid_energy = sum(energy_valid[i] for i in indices)
        valid_force = sum(force_valid[i] for i in indices)
        system_metrics[system].update(
            expected_structure_count=len(indices),
            valid_energy_count=valid_energy,
            failed_energy_count=len(indices) - valid_energy,
            valid_force_count=valid_force,
            failed_force_count=len(indices) - valid_force,
        )
    return {
        "status": "aligned",
        "error": None,
        "systems": system_metrics,
        "overall": overall,
    }


def unavailable_result(
    summary: dict[str, dict[str, Any]], status: str, error: str | None = None
) -> dict[str, Any]:
    """
    Represent unavailable or invalid output without claiming evaluated counts.

    Parameters
    ----------
    summary
        Counts and normalisers calculated from the canonical reference.
    status
        ``missing_output`` for absent files or ``invalid_output`` for bad alignment.
    error
        Explanation of invalid output, if present.

    Returns
    -------
    dict[str, Any]
        Missing metrics are None; invalid metrics are NaN. Counts are unknown.
    """
    value = np.nan if status == "invalid_output" else None
    systems = {}
    for system in SYSTEM_ORDER:
        systems[system] = summary[system] | {
            "energy_rmse": value,
            "force_rmse": value,
            "Energy RelRMSE": value,
            "Force RelRMSE": value,
            "valid_energy_count": None,
            "failed_energy_count": None,
            "valid_force_count": None,
            "failed_force_count": None,
        }
    return {
        "status": status,
        "error": error,
        "systems": systems,
        "overall": dict.fromkeys(METRIC_NAMES, value),
    }


def collect_model_results(references: Sequence[Atoms]) -> dict[str, dict[str, Any]]:
    """
    Read selected models, distinguishing missing files from invalid results.

    Parameters
    ----------
    references
        Canonical reference loaded even when no model has output.

    Returns
    -------
    dict[str, dict[str, Any]]
        Model-keyed diagnostic records for this analysis selection.
    """
    summary = reference_summary(references)
    results = {}
    for model in MODELS:
        filename = CALC_PATH / model / "am26_results.extxyz"
        if not filename.exists():
            results[model] = unavailable_result(summary, "missing_output")
            continue
        try:
            outputs = read(filename, index=":", format="extxyz")
            results[model] = analyse_output(references, outputs)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            warn(f"AM26 {model}: invalid output: {message}", stacklevel=2)
            results[model] = unavailable_result(summary, "invalid_output", message)
    return results


def write_reference_info(data_path: str | Path | None = None) -> dict[str, Any]:
    """
    Save per-frame elements and labels from every canonical reference frame.

    Parameters
    ----------
    data_path
        Optional local file; the same verified resolver is used as for calculation.

    Returns
    -------
    dict[str, Any]
        Standard ML-PEG elements, labels, systems, and original input indices.
    """
    reference_path = get_reference_path(data_path)
    info = get_struct_info(
        calc_path=reference_path.parent.parent,
        model_name=reference_path.parent.name,
        glob_pattern=reference_path.name,
        index=":",
        info_keys=["label", "system"],
        write_info=False,
        write_structs=False,
    )
    info["input_index"] = list(range(len(info["elements"])))
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_bytes(orjson.dumps(info, option=orjson.OPT_INDENT_2))
    return info


def write_diagnostics(
    results: dict[str, dict[str, Any]], references: Sequence[Atoms]
) -> dict[str, Any]:
    """
    Save model-keyed diagnostics with standard incremental-model preservation.

    Parameters
    ----------
    results
        Diagnostics for the models selected in this run.
    references
        Canonical reference frames, used for counts and unavailable-model entries.

    Returns
    -------
    dict[str, Any]
        Versioned auxiliary bundle. Non-finite numbers serialize as JSON null;
        status and counts distinguish unavailable results from required failures.
    """
    filename = OUT_PATH / "system_metrics.json"
    merged = merge_saved_models(results, filename, key="models")
    summary = reference_summary(references)
    model_data = {
        model: merged.get(model, unavailable_result(summary, "missing_output"))
        for model in get_model_names()
    }
    bundle = {
        "schema_version": 1,
        "reference": {"sha256": REFERENCE_SHA256, "structure_count": len(references)},
        "system_order": SYSTEM_ORDER,
        "system_labels": SYSTEM_LABELS,
        "units": {
            "energy_rmse": "eV/atom",
            "energy_std": "eV/atom",
            "force_rmse": "eV/Å",
            "force_std": "eV/Å",
            "RelRMSE": "%",
        },
        "models": model_data,
    }
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    filename.write_bytes(orjson.dumps(bundle, option=orjson.OPT_INDENT_2))
    return bundle


def write_metrics_table(results: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Build the two-metric table using ML-PEG scoring and update behaviour.

    Parameters
    ----------
    results
        Current model diagnostics, with the approved overall metrics.

    Returns
    -------
    dict[str, dict]
        Exactly two metric-to-model mappings passed to the standard table builder.
    """

    @build_table(
        filename=OUT_PATH / "am26_metrics_table.json",
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS.copy(),
    )
    def metrics() -> dict[str, dict]:
        """Return the two overall metrics, preserving standard NaN/None semantics."""
        return {
            metric: {model: data["overall"][metric] for model, data in results.items()}
            for metric in METRIC_NAMES
        }

    return metrics()


def run_analysis() -> dict[str, dict[str, Any]]:
    """
    Run local analysis without inference or using model outputs as reference truth.

    Returns
    -------
    dict[str, dict[str, Any]]
        Diagnostics for this run's selected models.
    """
    reference_path = get_reference_path()
    references = load_reference_structures(reference_path)
    results = collect_model_results(references)
    write_reference_info(reference_path)
    bundle = write_diagnostics(results, references)
    write_heatmaps(bundle, OUT_PATH)
    write_metrics_table(results)
    return results


def test_am26_analysis() -> None:
    """Launch AM26 analysis through ML-PEG's standard pytest entry point."""
    run_analysis()
