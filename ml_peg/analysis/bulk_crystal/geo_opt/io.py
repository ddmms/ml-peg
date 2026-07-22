"""I/O for compressed and uncompressed geometry-optimization artifacts."""

from __future__ import annotations

import ast
from collections.abc import Sequence
import json
import os

import pandas as pd

from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    ANALYSIS_FIELDS,
    GEO_OPT_FIELDS,
    HALL_SYMBOL,
    INTERNATIONAL_SPG_NAME,
    MATERIAL_ID,
    SITE_SYMMETRY_SYMBOLS,
    WYCKOFF_SYMBOLS,
    GeoOptRecord,
    validate_analysis_dataframe,
    validate_geo_opt_dataframe,
    validate_reference_dataframe,
)
from ml_peg.data.artifacts import PathLike, read_csv_artifact, read_jsonl_artifact

_LIST_ANALYSIS_FIELDS = (SITE_SYMMETRY_SYMBOLS, WYCKOFF_SYMBOLS)


def _decode_list_value(value: object) -> object:
    """Decode JSON or pandas-repr list values from analysis CSV files."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid list encoding: {value!r}") from exc


def _normalize_mbd_analysis_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Map the original Matbench symmetry column to the corrected schema."""
    if SITE_SYMMETRY_SYMBOLS in dataframe or INTERNATIONAL_SPG_NAME not in dataframe:
        return dataframe
    normalized = dataframe.copy()
    normalized[SITE_SYMMETRY_SYMBOLS] = normalized[INTERNATIONAL_SPG_NAME].map(
        _decode_list_value
    )
    if HALL_SYMBOL in normalized:
        normalized[INTERNATIONAL_SPG_NAME] = normalized[HALL_SYMBOL]
    return normalized


def _decode_list_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with serialized CSV list columns decoded."""
    decoded = dataframe.copy()
    for field in _LIST_ANALYSIS_FIELDS:
        if field not in decoded:
            continue
        decoded[field] = decoded[field].map(_decode_list_value)
    return decoded


def _encode_list_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with list columns encoded as JSON strings."""
    encoded = dataframe.copy()
    for field in _LIST_ANALYSIS_FIELDS:
        encoded[field] = encoded[field].map(json.dumps)
    return encoded


def read_geo_opt_jsonl(file_path: PathLike) -> pd.DataFrame:
    """Read and validate a geo-opt JSONL file."""
    dataframe = read_jsonl_artifact(
        file_path, dtype={MATERIAL_ID: "string"}, precise_float=True
    )
    return validate_geo_opt_dataframe(dataframe)


def read_geo_opt_records(file_path: PathLike) -> list[GeoOptRecord]:
    """Read validated geo-opt JSONL records as dictionaries in file order."""
    dataframe = read_geo_opt_jsonl(file_path)
    return dataframe.loc[:, GEO_OPT_FIELDS].to_dict(orient="records")


def write_geo_opt_jsonl(
    records: pd.DataFrame | Sequence[GeoOptRecord], file_path: PathLike
) -> None:
    """Validate and write geo-opt records as JSONL."""
    dataframe = (
        records.copy()
        if isinstance(records, pd.DataFrame)
        else pd.DataFrame.from_records(records, columns=GEO_OPT_FIELDS)
    )
    normalized = validate_geo_opt_dataframe(dataframe)
    normalized.loc[:, GEO_OPT_FIELDS].to_json(
        os.fspath(file_path),
        orient="records",
        lines=True,
        compression="infer",
        double_precision=15,
    )


def read_reference_jsonl(file_path: PathLike) -> pd.DataFrame:
    """Read and validate reference structures from JSONL."""
    dataframe = read_jsonl_artifact(
        file_path, dtype={MATERIAL_ID: "string"}, precise_float=True
    )
    return validate_reference_dataframe(dataframe)


def read_analysis_csv(file_path: PathLike) -> pd.DataFrame:
    """Read validated per-structure analysis from plain or compressed CSV."""
    dataframe = _decode_list_columns(
        _normalize_mbd_analysis_columns(
            read_csv_artifact(file_path, dtype={MATERIAL_ID: str})
        )
    )
    return validate_analysis_dataframe(dataframe)


def write_analysis_csv(dataframe: pd.DataFrame, file_path: PathLike) -> None:
    """Validate and write an analysis CSV."""
    serialized = _encode_list_columns(validate_analysis_dataframe(dataframe))
    serialized.loc[:, ANALYSIS_FIELDS[1:]].to_csv(
        os.fspath(file_path),
        index=True,
        index_label=MATERIAL_ID,
        compression="infer",
    )
