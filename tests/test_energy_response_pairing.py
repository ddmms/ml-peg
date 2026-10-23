"""Unit tests for energy response field/zero-field pairing."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write
import numpy as np
import pytest

from ml_peg.analysis.electric_field.energy_response.analyse_energy_response import (
    pair_by_formula,
)

FIELD = [0.0, 0.0, 1e-3]


def build_struct(formula: str, field: list[float] | None) -> Atoms:
    """
    Build a structure with a recorded external field.

    Parameters
    ----------
    formula
        Chemical formula of the structure.
    field
        External field to record, or `None` to record none at all.

    Returns
    -------
    Atoms
        Structure with the requested formula and recorded field.
    """
    struct = Atoms(formula)
    if field is not None:
        struct.info["external_field"] = field
    return struct


def write_dataset(path: Path, structs: list[Atoms]) -> Path:
    """
    Write structures to an extended xyz dataset.

    Parameters
    ----------
    path
        Path to write the dataset to.
    structs
        Structures to write, in the order given.

    Returns
    -------
    Path
        Path the dataset was written to.
    """
    write(path, structs, format="extxyz")
    return path


def test_pairs_by_formula_not_file_order(tmp_path: Path) -> None:
    """Test pairing is by formula, not by position in the file."""
    # Field and zero-field structures deliberately in different relative orders
    dataset = write_dataset(
        tmp_path / "SHUFFLED.xyz",
        [
            build_struct("C4H10", [0.0, 0.0, 0.0]),
            build_struct("C2H6", FIELD),
            build_struct("C2H6", [0.0, 0.0, 0.0]),
            build_struct("C4H10", FIELD),
        ],
    )

    pairs = pair_by_formula(dataset)

    assert list(pairs) == ["C2H6", "C4H10"]
    for formula, (no_field, field) in pairs.items():
        assert no_field.get_chemical_formula() == formula
        assert field.get_chemical_formula() == formula
        assert not np.any(no_field.info["external_field"])
        assert np.any(field.info["external_field"])


def test_pairs_ordered_by_size(tmp_path: Path) -> None:
    """Test pairs are ordered by increasing number of atoms, not alphabetically."""
    formulas = ["C2H6", "C4H10", "C10H22"]
    structs = [build_struct(f, [0.0, 0.0, 0.0]) for f in reversed(formulas)]
    structs += [build_struct(f, FIELD) for f in formulas]
    dataset = write_dataset(tmp_path / "ALKANES.xyz", structs)

    assert list(pair_by_formula(dataset)) == formulas


def test_unpaired_structure_dropped(tmp_path: Path) -> None:
    """Test a structure with no counterpart is dropped with a warning."""
    dataset = write_dataset(
        tmp_path / "UNPAIRED.xyz",
        [
            build_struct("C2H6", [0.0, 0.0, 0.0]),
            build_struct("C2H6", FIELD),
            build_struct("C4H10", FIELD),
        ],
    )

    with pytest.warns(UserWarning, match="C4H10.*no field or zero-field counterpart"):
        pairs = pair_by_formula(dataset)

    assert list(pairs) == ["C2H6"]


def test_duplicate_formula_warns(tmp_path: Path) -> None:
    """Test a repeated formula in the same field state is reported and dropped."""
    dataset = write_dataset(
        tmp_path / "DUPLICATE.xyz",
        [
            build_struct("C2H6", [0.0, 0.0, 0.0]),
            build_struct("C2H6", FIELD),
            build_struct("C2H6", FIELD),
        ],
    )

    with pytest.warns(UserWarning, match="Duplicate C2H6"):
        pairs = pair_by_formula(dataset)

    assert list(pairs) == ["C2H6"]


def test_missing_external_field_skipped(tmp_path: Path) -> None:
    """Test a structure with no recorded external field is skipped."""
    dataset = write_dataset(
        tmp_path / "NO_FIELD_INFO.xyz",
        [
            build_struct("C2H6", [0.0, 0.0, 0.0]),
            build_struct("C2H6", FIELD),
            build_struct("C4H10", None),
        ],
    )

    with pytest.warns(UserWarning, match="no recorded external_field"):
        pairs = pair_by_formula(dataset)

    assert list(pairs) == ["C2H6"]
