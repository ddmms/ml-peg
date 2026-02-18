"""Unit tests for low-dimensional relaxation utilities."""

from __future__ import annotations

from ase import Atoms
import pytest

from ml_peg.calcs.bulk_crystal.low_dimensional_relaxation.calc_low_dimensional_relaxation import (  # noqa: E501
    get_area_per_atom,
    get_length_per_atom,
)


def test_get_area_per_atom() -> None:
    """Test get_area_per_atom for a simple 2D structure."""
    # 2-atom structure with 3x4 in-plane cell → area = 12, area/atom = 6
    atoms = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[3, 4, 20], pbc=True)
    assert get_area_per_atom(atoms) == pytest.approx(6.0)


def test_get_length_per_atom() -> None:
    """Test get_length_per_atom for a simple 1D structure."""
    # 2-atom chain with a=5 Å → length/atom = 2.5
    atoms = Atoms("H2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 20, 20], pbc=True)
    assert get_length_per_atom(atoms) == pytest.approx(2.5)
