"""Schemas and validation for geometry-optimization artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any, TypedDict

import numpy as np
import pandas as pd

MATERIAL_ID = "material_id"
STRUCTURE = "structure"
ENERGY = "energy"
CONVERGED = "converged"
N_STEPS = "n_steps"

SPG_NUM = "spg_num"
HALL_NUM = "hall_num"
# International Hermann-Mauguin space-group name; site symmetry symbols are per-site.
INTERNATIONAL_SPG_NAME = "international_spg_name"
SITE_SYMMETRY_SYMBOLS = "site_symmetry_symbols"
WYCKOFF_SYMBOLS = "wyckoff_symbols"
N_SYM_OPS = "n_sym_ops"
N_ROT_SYMS = "n_rot_syms"
N_TRANS_SYMS = "n_trans_syms"
HALL_SYMBOL = "hall_symbol"
SYMPREC = "symprec"
ANGLE_TOLERANCE = "angle_tolerance"
SPG_NUM_DIFF = "spg_num_diff"
N_SYM_OPS_DIFF = "n_sym_ops_diff"
STRUCTURE_RMSD_VS_DFT = "structure_rmsd_vs_dft"
MAX_PAIR_DIST = "max_pair_dist"

GEO_OPT_FIELDS = (MATERIAL_ID, STRUCTURE, ENERGY, CONVERGED, N_STEPS)
SYMMETRY_FIELDS = (
    SPG_NUM,
    HALL_NUM,
    INTERNATIONAL_SPG_NAME,
    SITE_SYMMETRY_SYMBOLS,
    WYCKOFF_SYMBOLS,
    N_SYM_OPS,
    N_ROT_SYMS,
    N_TRANS_SYMS,
    HALL_SYMBOL,
    SYMPREC,
    ANGLE_TOLERANCE,
)
COMPARISON_FIELDS = (
    SPG_NUM_DIFF,
    N_SYM_OPS_DIFF,
    STRUCTURE_RMSD_VS_DFT,
    MAX_PAIR_DIST,
)
ANALYSIS_FIELDS = (MATERIAL_ID, *SYMMETRY_FIELDS, *COMPARISON_FIELDS)

_NUMERIC_ANALYSIS_FIELDS = (
    SPG_NUM,
    HALL_NUM,
    N_SYM_OPS,
    N_ROT_SYMS,
    N_TRANS_SYMS,
    SYMPREC,
    ANGLE_TOLERANCE,
    SPG_NUM_DIFF,
    N_SYM_OPS_DIFF,
    STRUCTURE_RMSD_VS_DFT,
    MAX_PAIR_DIST,
)
_INTEGER_ANALYSIS_FIELDS = (
    SPG_NUM,
    HALL_NUM,
    N_SYM_OPS,
    N_ROT_SYMS,
    N_TRANS_SYMS,
    SPG_NUM_DIFF,
    N_SYM_OPS_DIFF,
)
_NONNEGATIVE_ANALYSIS_FIELDS = (
    N_SYM_OPS,
    N_ROT_SYMS,
    N_TRANS_SYMS,
    STRUCTURE_RMSD_VS_DFT,
    MAX_PAIR_DIST,
)


class GeoOptRecord(TypedDict):
    """Final-structure record emitted by a geometry optimization."""

    material_id: str
    structure: dict[str, Any]
    energy: float
    converged: bool
    n_steps: int


def _missing_columns(
    dataframe: pd.DataFrame, required_columns: Sequence[str]
) -> list[str]:
    """
    Return absent required columns in their requested order.

    Parameters
    ----------
    dataframe
        Table whose columns are inspected.
    required_columns
        Column names that must be present.

    Returns
    -------
    list[str]
        Missing column names in requested order.
    """
    return [column for column in required_columns if column not in dataframe.columns]


def _validate_material_id_values(material_ids: pd.Series) -> None:
    """
    Require non-null, nonempty, unique string material identifiers.

    Parameters
    ----------
    material_ids
        Material identifiers to validate.
    """
    if material_ids.isna().any():
        raise ValueError("material_id values must not be null")

    invalid_mask = material_ids.map(
        lambda material_id: not isinstance(material_id, str) or not material_id.strip()
    )
    if invalid_mask.any():
        invalid_values = material_ids[invalid_mask].tolist()
        raise ValueError(
            f"material_id values must be non-empty strings; got {invalid_values!r}"
        )

    duplicate_ids = material_ids[material_ids.duplicated()].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate material_id values: {duplicate_ids!r}")


def _with_material_id_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Return a validated copy indexed by material ID.

    Parameters
    ----------
    dataframe
        Table containing material identifiers.

    Returns
    -------
    pandas.DataFrame
        Validated copy indexed by material ID.
    """
    normalized = dataframe.copy()
    if MATERIAL_ID in normalized.columns:
        _validate_material_id_values(normalized[MATERIAL_ID])
        normalized = normalized.set_index(MATERIAL_ID)
    elif normalized.index.name == MATERIAL_ID:
        material_ids = pd.Series(normalized.index, dtype=object)
        _validate_material_id_values(material_ids)
    else:
        raise ValueError(
            "Expected a 'material_id' column or an index named 'material_id'"
        )
    normalized.index.name = MATERIAL_ID
    return normalized


def validate_geo_opt_record(
    record: Mapping[str, object], *, record_number: int | None = None
) -> GeoOptRecord:
    """
    Validate one geo-opt record and normalize its scalar values.

    Parameters
    ----------
    record
        Geometry-optimization record to validate.
    record_number
        Optional record position used in error messages.

    Returns
    -------
    GeoOptRecord
        Validated record with normalized scalar values.
    """
    location = "" if record_number is None else f" at record {record_number}"
    missing_fields = [field for field in GEO_OPT_FIELDS if field not in record]
    if missing_fields:
        raise ValueError(f"Missing geo-opt fields{location}: {missing_fields!r}")

    material_id = record[MATERIAL_ID]
    if not isinstance(material_id, str) or not material_id.strip():
        raise ValueError(
            f"material_id must be a non-empty string{location}, got {material_id!r}"
        )

    structure = record[STRUCTURE]
    if not isinstance(structure, dict):
        raise ValueError(
            f"structure must be a dictionary{location}, got {type(structure).__name__}"
        )

    energy = record[ENERGY]
    if isinstance(energy, (bool, np.bool_)) or not isinstance(energy, Real):
        raise ValueError(f"energy must be numeric{location}, got {energy!r}")
    normalized_energy = float(energy)
    if not math.isfinite(normalized_energy):
        raise ValueError(f"energy must be finite{location}, got {energy!r}")

    converged = record[CONVERGED]
    if not isinstance(converged, (bool, np.bool_)):
        raise ValueError(f"converged must be boolean{location}, got {converged!r}")

    n_steps = record[N_STEPS]
    if (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, Integral)
        or n_steps < 0
    ):
        raise ValueError(
            f"n_steps must be a non-negative integer{location}, got {n_steps!r}"
        )

    return {
        MATERIAL_ID: material_id,
        STRUCTURE: structure,
        ENERGY: normalized_energy,
        CONVERGED: bool(converged),
        N_STEPS: int(n_steps),
    }


def validate_geo_opt_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize geo-opt records while retaining extra columns.

    Parameters
    ----------
    dataframe
        Geometry-optimization records to validate.

    Returns
    -------
    pandas.DataFrame
        Validated records with normalized values.
    """
    if missing_columns := _missing_columns(dataframe, GEO_OPT_FIELDS):
        raise ValueError(f"Missing geo-opt columns: {missing_columns!r}")

    normalized_records = [
        validate_geo_opt_record(record, record_number=record_number)
        for record_number, record in enumerate(dataframe.to_dict(orient="records"))
    ]
    normalized = dataframe.copy()
    for field in GEO_OPT_FIELDS:
        normalized[field] = [record[field] for record in normalized_records]
    duplicate_ids = normalized.loc[
        normalized[MATERIAL_ID].duplicated(), MATERIAL_ID
    ].unique()
    if duplicate_ids.size:
        raise ValueError(f"Duplicate material_id values: {duplicate_ids.tolist()!r}")
    return normalized


def validate_reference_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate references and return a material-ID-indexed copy.

    Parameters
    ----------
    dataframe
        Reference structures to validate.

    Returns
    -------
    pandas.DataFrame
        Validated references indexed by material ID.
    """
    normalized = _with_material_id_index(dataframe)
    if STRUCTURE not in normalized.columns:
        raise ValueError("Missing reference structure column: 'structure'")

    invalid_structures = [
        material_id
        for material_id, structure in normalized[STRUCTURE].items()
        if not isinstance(structure, dict)
    ]
    if invalid_structures:
        raise ValueError(
            "Reference structures must be dictionaries; invalid material IDs: "
            f"{invalid_structures!r}"
        )
    return normalized


def validate_analysis_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a complete symmetry-comparison analysis table.

    Parameters
    ----------
    dataframe
        Symmetry-comparison analysis to validate.

    Returns
    -------
    pandas.DataFrame
        Validated analysis indexed by material ID.
    """
    normalized = _with_material_id_index(dataframe)
    required_fields = (*SYMMETRY_FIELDS, *COMPARISON_FIELDS)
    if missing_columns := _missing_columns(normalized, required_fields):
        raise ValueError(f"Missing geo-opt analysis columns: {missing_columns!r}")

    for field in _NUMERIC_ANALYSIS_FIELDS:
        if field not in normalized:
            continue
        original = normalized[field]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_mask = original.notna() & ~np.isfinite(numeric)
        if invalid_mask.any():
            invalid_values = original[invalid_mask].tolist()
            raise ValueError(
                f"Analysis column {field!r} must be numeric and finite; "
                f"got {invalid_values!r}"
            )
        normalized[field] = numeric

    for field in _INTEGER_ANALYSIS_FIELDS:
        invalid_mask = normalized[field].notna() & (normalized[field] % 1 != 0)
        if invalid_mask.any():
            raise ValueError(f"Analysis column {field!r} must contain integers")

    for field in _NONNEGATIVE_ANALYSIS_FIELDS:
        if (normalized[field].dropna() < 0).any():
            raise ValueError(f"Analysis column {field!r} must be non-negative")

    for field, lower_bound, upper_bound in (
        (SPG_NUM, 1, 230),
        (HALL_NUM, 1, 530),
    ):
        if not normalized[field].dropna().between(lower_bound, upper_bound).all():
            raise ValueError(
                f"Analysis column {field!r} must be between "
                f"{lower_bound} and {upper_bound}"
            )

    for field in (INTERNATIONAL_SPG_NAME, HALL_SYMBOL):
        invalid_mask = ~normalized[field].map(lambda value: isinstance(value, str))
        if invalid_mask.any():
            invalid_values = normalized.loc[invalid_mask, field].tolist()
            raise ValueError(
                f"Analysis column {field!r} must contain strings; "
                f"got {invalid_values!r}"
            )

    for field in (SITE_SYMMETRY_SYMBOLS, WYCKOFF_SYMBOLS):
        invalid_mask = ~normalized[field].map(
            lambda value: isinstance(value, (list, tuple))
        )
        if invalid_mask.any():
            invalid_values = normalized.loc[invalid_mask, field].tolist()
            raise ValueError(
                f"Analysis column {field!r} must contain lists; got {invalid_values!r}"
            )
        normalized[field] = normalized[field].map(list)

    if normalized[SYMPREC].isna().any() or (normalized[SYMPREC] <= 0).any():
        raise ValueError("symprec values must be positive and non-null")
    normalized[ANGLE_TOLERANCE] = (
        normalized[ANGLE_TOLERANCE]
        .astype(object)
        .where(normalized[ANGLE_TOLERANCE].notna(), None)
    )
    return normalized
