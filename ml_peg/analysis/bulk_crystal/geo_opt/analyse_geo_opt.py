"""Analyze geometry-optimization results."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
from typing import TYPE_CHECKING, Any

import pandas as pd

from ml_peg.analysis.bulk_crystal.geo_opt.metrics import calc_geo_opt_metrics
from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    MATERIAL_ID,
    STRUCTURE,
    validate_geo_opt_dataframe,
    validate_reference_dataframe,
)
from ml_peg.analysis.bulk_crystal.geo_opt.symmetry import (
    ProgressConfig,
    _structure_distances,
    _symmetry_comparison,
    _validate_symmetry_parameters,
    get_sym_info_from_structs,
)
from ml_peg.data.artifacts import (
    MATBENCH_DISCOVERY_ID,
    MATBENCH_DISCOVERY_VERSION,
    PathLike,
    read_jsonl_artifact,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure

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
) -> dict[str, Structure]:
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
    dict[str, Structure]
        Pymatgen structures keyed by material ID.
    """
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError("Geo-opt analysis requires pymatgen") from exc

    structures: dict[str, Structure] = {}
    for material_id, structure_dict in dataframe[STRUCTURE].items():
        try:
            structures[str(material_id)] = Structure.from_dict(structure_dict)
        except Exception as exc:
            raise ValueError(
                f"Invalid structure dictionary for {source_name} "
                f"material_id={material_id!r}"
            ) from exc
    return structures


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
    Analyze predicted structures against references from tables or local JSONL.

    ``angle_tolerance`` is in radians. Per-structure records are omitted unless
    requested to avoid duplicating WBM-scale analysis tables in memory.

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
    if isinstance(predictions, pd.DataFrame) != isinstance(references, pd.DataFrame):
        raise TypeError(
            "predictions and references must both be DataFrames or both be paths"
        )
    if not isinstance(predictions, pd.DataFrame):
        predictions = read_jsonl_artifact(
            predictions, dtype={MATERIAL_ID: "string"}, precise_float=True
        )
    if not isinstance(references, pd.DataFrame):
        references = read_jsonl_artifact(
            references, dtype={MATERIAL_ID: "string"}, precise_float=True
        )
    normalized_symprecs = _validate_symprecs(symprecs)
    normalized_angle_tolerance = (
        float(angle_tolerance) if angle_tolerance is not None else None
    )
    _validate_symmetry_parameters(normalized_symprecs[0], normalized_angle_tolerance)
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

    distances = _structure_distances(
        predicted_structures,
        reference_structures,
        validated_predictions.index,
        pbar=pbar,
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
        comparison = _symmetry_comparison(
            predicted_symmetry, reference_symmetry, distances
        )
        mantissa, exponent = f"{Decimal(str(symprec)).normalize():e}".split("e")
        symprec_key = f"symprec={mantissa}e{int(exponent)}"
        symprec_result: dict[str, Any] = {
            "symprec": symprec,
            "angle_tolerance": normalized_angle_tolerance,
            "metrics": {
                name: value if math.isfinite(value) else None
                for name, value in calc_geo_opt_metrics(comparison).items()
            },
        }
        if include_analysis:
            symprec_result["analysis"] = json.loads(
                comparison.reset_index().to_json(orient="records", double_precision=15)
            )
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
