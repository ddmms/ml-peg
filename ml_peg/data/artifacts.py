"""Read local benchmark artifacts and validate their material identifiers."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Any, Final

import pandas as pd

PathLike = str | os.PathLike[str]

MATBENCH_DISCOVERY_ID: Final = "matbench-discovery"
MATBENCH_DISCOVERY_VERSION: Final = "1.3.1"


def _checked_file_path(file_path: PathLike) -> str:
    """
    Return a local artifact path after confirming it names a file.

    Parameters
    ----------
    file_path
        Local artifact path.

    Returns
    -------
    str
        Validated filesystem path.
    """
    normalized_path = os.fspath(file_path)
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"Artifact file not found: {normalized_path!r}")
    return normalized_path


def read_csv_artifact(file_path: PathLike, **read_options: Any) -> pd.DataFrame:
    """
    Read a CSV artifact with compression inferred from its filename.

    Parameters
    ----------
    file_path
        Local CSV artifact path.
    **read_options
        Additional options passed to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Loaded artifact data.
    """
    return pd.read_csv(
        _checked_file_path(file_path), compression="infer", **read_options
    )


def read_jsonl_artifact(file_path: PathLike, **read_options: Any) -> pd.DataFrame:
    """
    Read a line-delimited JSON artifact with transparent compression.

    Parameters
    ----------
    file_path
        Local JSON Lines artifact path.
    **read_options
        Additional options passed to :func:`pandas.read_json`.

    Returns
    -------
    pandas.DataFrame
        Loaded artifact data.
    """
    return pd.read_json(
        _checked_file_path(file_path),
        lines=True,
        compression="infer",
        **read_options,
    )


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    artifact_name: str = "dataframe",
) -> None:
    """
    Raise if a dataframe lacks any required columns.

    Parameters
    ----------
    dataframe
        Dataframe to validate.
    required_columns
        Column names that must be present.
    artifact_name
        Artifact label used in error messages.
    """
    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{artifact_name} missing required columns: {sorted(missing_columns)}"
        )


def material_id_index(
    dataframe: pd.DataFrame,
    *,
    id_column: str = "material_id",
    artifact_name: str = "dataframe",
) -> pd.Index:
    """
    Return IDs from a column or named index after validating uniqueness.

    Parameters
    ----------
    dataframe
        Dataframe containing material identifiers.
    id_column
        Material identifier column or index name.
    artifact_name
        Artifact label used in error messages.

    Returns
    -------
    pandas.Index
        Validated material identifiers.
    """
    has_id_column = id_column in dataframe.columns
    has_id_index = dataframe.index.name == id_column
    if not has_id_column and not has_id_index:
        raise ValueError(
            f"{artifact_name} must contain {id_column!r} as a column or index"
        )

    if has_id_column:
        identifiers = pd.Index(dataframe[id_column], name=id_column)
        if has_id_index and not identifiers.equals(dataframe.index):
            raise ValueError(
                f"{artifact_name} has inconsistent {id_column!r} column and index"
            )
    else:
        identifiers = dataframe.index.copy()

    if identifiers.hasnans:
        raise ValueError(f"{artifact_name} contains missing {id_column!r} values")
    if identifiers.has_duplicates:
        duplicate_ids = identifiers[identifiers.duplicated()].unique().tolist()
        raise ValueError(
            f"{artifact_name} contains duplicate {id_column!r} values: "
            f"{duplicate_ids!r}"
        )
    return identifiers
