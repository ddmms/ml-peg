"""Schemas for materials-discovery reference and prediction artifacts."""

from __future__ import annotations

from enum import Enum
from numbers import Real
from typing import Final

import numpy as np
import pandas as pd

from ml_peg.data.artifacts import material_id_index, validate_required_columns

MATERIAL_ID: Final = "material_id"
E_ABOVE_HULL: Final = "e_above_hull_mp2020_corrected_ppd_mp"
REFERENCE_FORMATION_ENERGY: Final = "e_form_per_atom_mp2020_corrected"
UNIQUE_PROTOTYPE: Final = "unique_prototype"
PREDICTED_FORMATION_ENERGY: Final = "e_form_per_atom"

REFERENCE_COLUMNS: Final[tuple[str, ...]] = (
    E_ABOVE_HULL,
    REFERENCE_FORMATION_ENERGY,
    UNIQUE_PROTOTYPE,
)
PREDICTION_COLUMNS: Final[tuple[str, ...]] = (PREDICTED_FORMATION_ENERGY,)


class DiscoverySubset(str, Enum):
    """Subsets reported by materials-discovery evaluation."""

    full_test_set = "full_test_set"
    unique_prototypes = "unique_prototypes"
    most_stable_10k = "most_stable_10k"

    def __str__(self) -> str:
        """
        Return the serialized subset key.

        Returns
        -------
        str
            Serialized subset key.
        """
        return self.value


def index_by_material_id(
    dataframe: pd.DataFrame,
    *,
    artifact_name: str,
) -> pd.DataFrame:
    """
    Return a copy indexed by validated material identifiers.

    Parameters
    ----------
    dataframe
        Artifact data containing material identifiers.
    artifact_name
        Artifact label used in error messages.

    Returns
    -------
    pandas.DataFrame
        Copy indexed by material ID.
    """
    identifiers = material_id_index(
        dataframe, id_column=MATERIAL_ID, artifact_name=artifact_name
    )
    identifiers = identifiers.astype(str)
    if identifiers.has_duplicates:
        duplicate_ids = identifiers[identifiers.duplicated()].unique().tolist()
        raise ValueError(
            f"{artifact_name} contains duplicate {MATERIAL_ID!r} values after "
            f"string conversion: {duplicate_ids!r}"
        )
    indexed_dataframe = dataframe.drop(columns=MATERIAL_ID, errors="ignore").copy()
    indexed_dataframe.index = identifiers
    return indexed_dataframe


def _validated_frame(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
    artifact_name: str,
) -> pd.DataFrame:
    """
    Validate required columns and IDs, returning an indexed copy.

    Parameters
    ----------
    dataframe
        Artifact data to validate.
    required_columns
        Column names that must be present.
    artifact_name
        Artifact label used in error messages.

    Returns
    -------
    pandas.DataFrame
        Validated copy indexed by material ID.
    """
    validate_required_columns(dataframe, required_columns, artifact_name=artifact_name)
    return index_by_material_id(dataframe, artifact_name=artifact_name)


def _validated_reference_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a discovery reference and return its indexed copy.

    Parameters
    ----------
    dataframe
        Discovery reference data.

    Returns
    -------
    pandas.DataFrame
        Validated copy indexed by material ID.
    """
    indexed_dataframe = _validated_frame(
        dataframe, REFERENCE_COLUMNS, "discovery reference"
    )
    for energy_column in (E_ABOVE_HULL, REFERENCE_FORMATION_ENERGY):
        numeric_values = pd.to_numeric(
            indexed_dataframe[energy_column], errors="coerce"
        )
        invalid_mask = ~np.isfinite(numeric_values.to_numpy(dtype=float))
        if invalid_mask.any():
            invalid_values = indexed_dataframe.loc[invalid_mask, energy_column].tolist()
            raise ValueError(
                f"{energy_column!r} values must be finite, got {invalid_values!r}"
            )
    unique_prototype_flags = indexed_dataframe[UNIQUE_PROTOTYPE]
    invalid_flags = ~unique_prototype_flags.map(
        lambda value: (
            isinstance(value, (bool, np.bool_))
            or (isinstance(value, Real) and value in (0, 1))
        )
    )
    if invalid_flags.any():
        invalid_values = unique_prototype_flags[invalid_flags].tolist()
        raise ValueError(
            f"{UNIQUE_PROTOTYPE!r} values must be boolean, got {invalid_values!r}"
        )
    return indexed_dataframe


def validate_reference_frame(dataframe: pd.DataFrame) -> None:
    """
    Validate discovery reference columns, IDs, energies, and flags.

    Parameters
    ----------
    dataframe
        Discovery reference data.
    """
    _validated_reference_frame(dataframe)


def validate_prediction_frame(dataframe: pd.DataFrame) -> None:
    """
    Validate discovery prediction columns and material IDs.

    Parameters
    ----------
    dataframe
        Discovery prediction data.
    """
    _validated_frame(dataframe, PREDICTION_COLUMNS, "discovery predictions")


def prediction_series(dataframe: pd.DataFrame) -> pd.Series:
    """
    Return validated formation-energy predictions indexed by material ID.

    Parameters
    ----------
    dataframe
        Discovery prediction data.

    Returns
    -------
    pandas.Series
        Formation-energy predictions indexed by material ID.
    """
    indexed_dataframe = _validated_frame(
        dataframe, PREDICTION_COLUMNS, "discovery predictions"
    )
    return indexed_dataframe[PREDICTED_FORMATION_ENERGY]
