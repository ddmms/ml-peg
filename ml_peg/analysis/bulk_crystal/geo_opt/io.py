"""I/O for compressed and uncompressed geometry-optimization artifacts."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os

import pandas as pd

from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    ANALYSIS_FIELDS,
    GEO_OPT_FIELDS,
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
    """
    Decode a JSON list from an analysis CSV cell.

    Parameters
    ----------
    value
        CSV cell value to decode.

    Returns
    -------
    object
        Decoded JSON value, or the unchanged non-string value.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid list encoding: {value!r}") from exc


def read_geo_opt_jsonl(file_path: PathLike) -> pd.DataFrame:
    """
    Read and validate a geo-opt JSONL file.

    Parameters
    ----------
    file_path
        Path to the JSONL artifact.

    Returns
    -------
    pandas.DataFrame
        Validated geometry-optimization records.
    """
    dataframe = read_jsonl_artifact(
        file_path, dtype={MATERIAL_ID: "string"}, precise_float=True
    )
    return validate_geo_opt_dataframe(dataframe)


def read_geo_opt_records(file_path: PathLike) -> list[GeoOptRecord]:
    """
    Read validated geo-opt JSONL records as dictionaries in file order.

    Parameters
    ----------
    file_path
        Path to the JSONL artifact.

    Returns
    -------
    list[GeoOptRecord]
        Validated records in file order.
    """
    dataframe = read_geo_opt_jsonl(file_path)
    return dataframe.loc[:, GEO_OPT_FIELDS].to_dict(orient="records")


def write_geo_opt_jsonl(
    records: pd.DataFrame | Sequence[GeoOptRecord], file_path: PathLike
) -> None:
    """
    Validate and write geo-opt records as JSONL.

    Parameters
    ----------
    records
        Geometry-optimization records to write.
    file_path
        Destination JSONL path.
    """
    dataframe = (
        records
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
    """
    Read and validate reference structures from JSONL.

    Parameters
    ----------
    file_path
        Path to the reference JSONL artifact.

    Returns
    -------
    pandas.DataFrame
        Validated reference structures indexed by material ID.
    """
    dataframe = read_jsonl_artifact(
        file_path, dtype={MATERIAL_ID: "string"}, precise_float=True
    )
    return validate_reference_dataframe(dataframe)


def read_analysis_csv(file_path: PathLike) -> pd.DataFrame:
    """
    Read validated per-structure analysis from plain or compressed CSV.

    Parameters
    ----------
    file_path
        Path to the analysis CSV artifact.

    Returns
    -------
    pandas.DataFrame
        Validated per-structure analysis.
    """
    dataframe = read_csv_artifact(file_path, dtype={MATERIAL_ID: str})
    for field in _LIST_ANALYSIS_FIELDS:
        if field in dataframe:
            dataframe[field] = dataframe[field].map(_decode_list_value)
    return validate_analysis_dataframe(dataframe)


def write_analysis_csv(dataframe: pd.DataFrame, file_path: PathLike) -> None:
    """
    Validate and write an analysis CSV.

    Parameters
    ----------
    dataframe
        Per-structure analysis to write.
    file_path
        Destination CSV path.
    """
    serialized = validate_analysis_dataframe(dataframe)
    for field in _LIST_ANALYSIS_FIELDS:
        serialized[field] = serialized[field].map(json.dumps)
    serialized.loc[:, ANALYSIS_FIELDS[1:]].to_csv(
        os.fspath(file_path),
        index=True,
        index_label=MATERIAL_ID,
        compression="infer",
    )
