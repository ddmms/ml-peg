"""Tests for the YBCO lattice-vs-oxygen benchmark helpers."""

from __future__ import annotations

from ase import Atoms
import pytest

from ml_peg.analysis.bulk_crystal.YBCO_lattice_vs_O import (
    analyse_ybco_lattice_vs_O as A,  # noqa: N812
)
from ml_peg.models.mock import MockCalculator


def test_reference_shapes() -> None:
    """The DFT reference has 11 oxygen contents for each of a, b, c."""
    # YBCO6.00, 6.10, ... 7.00 = 11 points; 11 x 3 params = 33 flattened values
    assert len(A.CONC) == 11
    assert len(A.DFT["a"]) == 11
    assert len(A.REF_FLAT) == 33


def test_orthorhombic_trend() -> None:
    """From YBCO6 to YBCO7 the a axis shrinks and b grows (orthorhombic splitting)."""
    # physics benchmark: adding oxygen splits a and b apart
    assert A.DFT["a"][0] > A.DFT["a"][-1]
    assert A.DFT["b"][0] < A.DFT["b"][-1]


def test_mock_calculator_runs() -> None:
    """The mock calculator returns a finite (zero) energy for a YBCO-like cell."""
    atoms = Atoms(
        "YBaCuO",
        positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
        cell=[6, 6, 6],
        pbc=True,
    )
    atoms.calc = MockCalculator()
    assert atoms.get_potential_energy() == pytest.approx(0.0)
