"""Analyze geometry-optimization results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import platform
from typing import Any, cast

import pandas as pd

from ml_peg.analysis.bulk_crystal.geo_opt.io import (
    PathLike,
    read_geo_opt_jsonl,
    read_reference_jsonl,
)
from ml_peg.analysis.bulk_crystal.geo_opt.metrics import calc_geo_opt_metrics
from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    MATERIAL_ID,
    STRUCTURE,
    validate_geo_opt_dataframe,
    validate_reference_dataframe,
)
from ml_peg.analysis.bulk_crystal.geo_opt.symmetry import (
    ProgressConfig,
    get_sym_info_from_structs,
    pred_vs_ref_struct_symmetry,
)
from ml_peg.data.artifacts import (
    MATBENCH_DISCOVERY_ID,
    MATBENCH_DISCOVERY_VERSION,
    canonical_scientific_notation,
)

CANONICAL_SYMPRECS = (1e-2, 1e-5)
RESULT_SCHEMA_VERSION = 1


def _package_version(package_name: str) -> str | None:
    """
    Return an installed package version, or ``None`` when unavailable.

    Parameters
    ----------
    package_name
        Installed distribution name.

    Returns
    -------
    str | None
        Installed version, if available.
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def get_version_metadata() -> dict[str, str | None]:
    """
    Return runtime and dependency versions affecting geo-opt analysis.

    Returns
    -------
    dict[str, str | None]
        Versions keyed by runtime or dependency name.
    """
    return {
        "python": platform.python_version(),
        "ml_peg": _package_version("ml-peg"),
        "pandas": _package_version("pandas"),
        "ase": _package_version("ase"),
        "pymatgen": _package_version("pymatgen"),
        "moyopy": _package_version("moyopy"),
    }


def _validate_symprecs(symprecs: Sequence[float]) -> tuple[float, ...]:
    """
    Validate that symmetry tolerances are nonempty, positive, finite, and unique.

    Parameters
    ----------
    symprecs
        Symmetry tolerances to validate.

    Returns
    -------
    tuple[float, ...]
        Normalized symmetry tolerances.
    """
    normalized = tuple(float(symprec) for symprec in symprecs)
    if not normalized:
        raise ValueError("At least one symprec value is required")
    invalid_values = [
        symprec for symprec in normalized if not math.isfinite(symprec) or symprec <= 0
    ]
    if invalid_values:
        raise ValueError(
            f"symprec values must be positive and finite: {invalid_values!r}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"symprec values must be unique: {normalized!r}")
    return normalized


def _structures_from_dataframe(
    dataframe: pd.DataFrame, *, source_name: str
) -> dict[str, object]:
    """
    Build pymatgen structures from serialized dictionaries.

    Parameters
    ----------
    dataframe
        Table containing serialized structures.
    source_name
        Source label used in error messages.

    Returns
    -------
    dict[str, object]
        Pymatgen structures keyed by material ID.
    """
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError("Geo-opt analysis requires pymatgen") from exc

    structures: dict[str, object] = {}
    for material_id, structure_dict in dataframe[STRUCTURE].items():
        try:
            structures[str(material_id)] = Structure.from_dict(structure_dict)
        except Exception as exc:
            raise ValueError(
                f"Invalid structure dictionary for {source_name} "
                f"material_id={material_id!r}"
            ) from exc
    return structures


def _json_safe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to JSON-compatible records.

    Parameters
    ----------
    dataframe
        Table to serialize.

    Returns
    -------
    list[dict[str, Any]]
        JSON-compatible records.
    """
    serialized = dataframe.reset_index().to_json(orient="records", double_precision=15)
    return cast(list[dict[str, Any]], json.loads(serialized))


def _json_safe_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    """
    Convert NumPy scalars and non-finite floats to JSON-compatible values.

    Parameters
    ----------
    value
        Mapping to normalize.

    Returns
    -------
    dict[str, Any]
        JSON-compatible mapping.
    """
    json_safe: dict[str, Any] = {}
    for key, nested_value in value.items():
        item_method = getattr(nested_value, "item", None)
        scalar_value = item_method() if callable(item_method) else nested_value
        if isinstance(scalar_value, float) and not math.isfinite(scalar_value):
            json_safe[key] = None
        else:
            json_safe[key] = scalar_value
    return json_safe


def analyze_geo_opt_dataframes(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    *,
    symprecs: Sequence[float] = CANONICAL_SYMPRECS,
    angle_tolerance: float | None = None,
    include_analysis: bool = False,
    pbar: ProgressConfig = False,
) -> dict[str, Any]:
    """
    Analyze predicted structures against references.

    ``angle_tolerance`` is in radians. Per-structure records are omitted unless
    requested to avoid duplicating WBM-scale analysis tables in memory.

    Parameters
    ----------
    predictions
        Predicted geometry-optimization structures.
    references
        Reference structures.
    symprecs
        Symmetry tolerances to analyze.
    angle_tolerance
        Optional angular tolerance in radians.
    include_analysis
        Whether to include per-structure analysis records.
    pbar
        Whether and how to display progress.

    Returns
    -------
    dict[str, Any]
        Versioned metrics and optional per-structure analysis.
    """
    normalized_symprecs = _validate_symprecs(symprecs)
    normalized_angle_tolerance = (
        float(angle_tolerance) if angle_tolerance is not None else None
    )
    validated_predictions = validate_geo_opt_dataframe(predictions).set_index(
        MATERIAL_ID
    )
    validated_references = validate_reference_dataframe(references)
    if validated_predictions.empty:
        raise ValueError("At least one prediction record is required")

    missing_reference_ids = sorted(
        set(validated_predictions.index) - set(validated_references.index)
    )
    if missing_reference_ids:
        raise ValueError(
            "Reference structures are missing predicted material IDs: "
            f"{missing_reference_ids!r}"
        )

    aligned_references = validated_references.loc[validated_predictions.index]
    predicted_structures = _structures_from_dataframe(
        validated_predictions, source_name="predictions"
    )
    reference_structures = _structures_from_dataframe(
        aligned_references, source_name="references"
    )

    analyses: dict[str, dict[str, Any]] = {}
    for symprec in normalized_symprecs:
        predicted_symmetry = get_sym_info_from_structs(
            predicted_structures,
            pbar=pbar,
            symprec=symprec,
            angle_tolerance=normalized_angle_tolerance,
        )
        reference_symmetry = get_sym_info_from_structs(
            reference_structures,
            pbar=pbar,
            symprec=symprec,
            angle_tolerance=normalized_angle_tolerance,
        )
        comparison = pred_vs_ref_struct_symmetry(
            predicted_symmetry,
            reference_symmetry,
            predicted_structures,
            reference_structures,
            pbar=pbar,
        )
        symprec_key = f"symprec={canonical_scientific_notation(symprec)}"
        symprec_result: dict[str, Any] = {
            "symprec": symprec,
            "angle_tolerance": normalized_angle_tolerance,
            "metrics": _json_safe_mapping(calc_geo_opt_metrics(comparison)),
        }
        if include_analysis:
            symprec_result["analysis"] = _json_safe_records(comparison)
        analyses[symprec_key] = symprec_result

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "source": {
            "framework": MATBENCH_DISCOVERY_ID,
            "version": MATBENCH_DISCOVERY_VERSION,
        },
        "versions": get_version_metadata(),
        "n_predictions": len(validated_predictions),
        "n_references": len(aligned_references),
        "symprecs": analyses,
    }


def analyze_geo_opt_paths(
    predictions_path: PathLike,
    references_path: PathLike,
    *,
    symprecs: Sequence[float] = CANONICAL_SYMPRECS,
    angle_tolerance: float | None = None,
    include_analysis: bool = False,
    pbar: ProgressConfig = False,
) -> dict[str, Any]:
    """
    Analyze prediction and reference JSONL artifacts from local paths.

    Parameters
    ----------
    predictions_path
        Path to predicted geometry-optimization structures.
    references_path
        Path to reference structures.
    symprecs
        Symmetry tolerances to analyze.
    angle_tolerance
        Optional angular tolerance in radians.
    include_analysis
        Whether to include per-structure analysis records.
    pbar
        Whether and how to display progress.

    Returns
    -------
    dict[str, Any]
        Versioned metrics and optional per-structure analysis.
    """
    predictions = read_geo_opt_jsonl(predictions_path)
    references = read_reference_jsonl(references_path)
    return analyze_geo_opt_dataframes(
        predictions,
        references,
        symprecs=symprecs,
        angle_tolerance=angle_tolerance,
        include_analysis=include_analysis,
        pbar=pbar,
    )


def analyze_geo_opt(
    predictions: pd.DataFrame | PathLike,
    references: pd.DataFrame | PathLike,
    *,
    symprecs: Sequence[float] = CANONICAL_SYMPRECS,
    angle_tolerance: float | None = None,
    include_analysis: bool = False,
    pbar: ProgressConfig = False,
) -> dict[str, Any]:
    """
    Analyze geo-opt structures supplied as two DataFrames or two local paths.

    Parameters
    ----------
    predictions
        Prediction table or local JSONL path.
    references
        Reference table or local JSONL path.
    symprecs
        Symmetry tolerances to analyze.
    angle_tolerance
        Optional angular tolerance in radians.
    include_analysis
        Whether to include per-structure analysis records.
    pbar
        Whether and how to display progress.

    Returns
    -------
    dict[str, Any]
        Versioned metrics and optional per-structure analysis.
    """
    if isinstance(predictions, pd.DataFrame) and isinstance(references, pd.DataFrame):
        return analyze_geo_opt_dataframes(
            predictions,
            references,
            symprecs=symprecs,
            angle_tolerance=angle_tolerance,
            include_analysis=include_analysis,
            pbar=pbar,
        )
    if not isinstance(predictions, pd.DataFrame) and not isinstance(
        references, pd.DataFrame
    ):
        return analyze_geo_opt_paths(
            os.fspath(predictions),
            os.fspath(references),
            symprecs=symprecs,
            angle_tolerance=angle_tolerance,
            include_analysis=include_analysis,
            pbar=pbar,
        )
    raise TypeError(
        "predictions and references must both be DataFrames or both be paths"
    )
