"""Typed helpers for reading and naming local benchmark artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
import os
import re
from typing import Any, Final

import pandas as pd

PathLike = str | os.PathLike[str]

MATBENCH_DISCOVERY_ID: Final = "matbench-discovery"
MATBENCH_DISCOVERY_VERSION: Final = "1.3.1"

ISO_DATE_PATTERN: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MOYO_VERSION_PATTERN: Final = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:[-+][0-9A-Za-z.-]+)?$"
)
_GEO_OPT_ANALYSIS_SUFFIX: Final = re.compile(
    r"^geo-opt-symprec=([^=]+)-moyo=([^=]+)\.csv\.gz$"
)


class ArtifactRole(str, Enum):
    """Canonical roles used in dated model artifact filenames."""

    discovery = "discovery"
    geo_opt = "geo_opt"
    geo_opt_analysis = "geo_opt_analysis"

    def __str__(self) -> str:
        """Return the role value."""
        return self.value


ARTIFACT_SUFFIXES: Final[dict[str, str]] = {
    str(ArtifactRole.discovery): "discovery.csv.gz",
    str(ArtifactRole.geo_opt): "geo-opt.jsonl.gz",
}


def _checked_file_path(file_path: PathLike) -> str:
    """Return a local artifact path after confirming it names a file."""
    normalized_path = os.fspath(file_path)
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"Artifact file not found: {normalized_path!r}")
    return normalized_path


def read_csv_artifact(file_path: PathLike, **read_options: Any) -> pd.DataFrame:
    """Read a CSV artifact with compression inferred from its filename."""
    return pd.read_csv(
        _checked_file_path(file_path), compression="infer", **read_options
    )


def read_jsonl_artifact(file_path: PathLike, **read_options: Any) -> pd.DataFrame:
    """Read a line-delimited JSON artifact with transparent compression."""
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
    """Raise if a dataframe lacks any required columns."""
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
    """Return IDs from a column or named index after validating uniqueness."""
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


def canonical_scientific_notation(value: float | str | Decimal) -> str:
    """Format a positive finite number as canonical notation like ``1e-5``."""
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value {value!r}") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise ValueError(f"Expected a positive finite number, got {value!r}")

    mantissa, _, exponent = f"{decimal_value.normalize():e}".partition("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def _iso_date(value: date | str) -> str:
    """Return a validated ``YYYY-MM-DD`` calendar date."""
    iso_date = value.isoformat() if isinstance(value, date) else value
    if not ISO_DATE_PATTERN.fullmatch(iso_date):
        raise ValueError(f"Expected an ISO date, got {value!r}")
    try:
        date.fromisoformat(iso_date)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date {value!r}") from exc
    return iso_date


def artifact_filename(
    artifact_date: date | str,
    role: str | ArtifactRole,
    *,
    symprec: float | str | Decimal | None = None,
    moyo_version: str | None = None,
) -> str:
    """Return a canonical dated basename for the requested artifact role."""
    iso_date = _iso_date(artifact_date)
    role_value = str(role)
    if role_value == ArtifactRole.geo_opt_analysis:
        if symprec is None or moyo_version is None:
            raise ValueError(
                "symprec and moyo_version are required for geo_opt_analysis"
            )
        if not MOYO_VERSION_PATTERN.fullmatch(moyo_version):
            raise ValueError(f"Invalid moyo version {moyo_version!r}")
        suffix = (
            f"geo-opt-symprec={canonical_scientific_notation(symprec)}"
            f"-moyo={moyo_version}.csv.gz"
        )
    else:
        if symprec is not None or moyo_version is not None:
            raise ValueError("symprec and moyo_version are only for geo_opt_analysis")
        if (suffix := ARTIFACT_SUFFIXES.get(role_value)) is None:
            raise ValueError(f"Unknown artifact role {role_value!r}")
    return f"{iso_date}-{suffix}"


def parse_artifact_filename(filename: str) -> ArtifactRole:
    """Validate a canonical artifact filename or path and return its role."""
    basename = os.path.basename(filename)
    if not ISO_DATE_PATTERN.match(basename[:10]) or basename[10:11] != "-":
        raise ValueError(f"Not a canonical model artifact filename: {filename!r}")
    artifact_date, suffix = basename[:10], basename[11:]
    _iso_date(artifact_date)
    for role_value, expected_suffix in ARTIFACT_SUFFIXES.items():
        if suffix == expected_suffix:
            return ArtifactRole(role_value)
    if match := _GEO_OPT_ANALYSIS_SUFFIX.fullmatch(suffix):
        symprec, moyo_version = match.groups()
        if (
            artifact_filename(
                artifact_date,
                ArtifactRole.geo_opt_analysis,
                symprec=symprec,
                moyo_version=moyo_version,
            )
            == basename
        ):
            return ArtifactRole.geo_opt_analysis
    raise ValueError(f"Not a canonical model artifact filename: {filename!r}")
