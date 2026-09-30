"""Tests for the MC500 molecular-crystal benchmark."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.molecular_crystal.MC500.analyse_MC500 import (
    rms_cartesian_displacement,
)


def test_rmscd_identical_structures() -> None:
    """Identical structures have zero RMSCD."""
    atoms = Atoms(
        "CH",
        scaled_positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        cell=[5.0, 6.0, 7.0],
    )
    atoms.pbc = True

    rmscd, rmscd_no_h, displacements = rms_cartesian_displacement(atoms, atoms)

    assert rmscd == pytest.approx(0.0)
    assert rmscd_no_h == pytest.approx(0.0)
    assert displacements == pytest.approx([0.0, 0.0])


def test_rmscd_averages_reference_and_relaxed_cells() -> None:
    """The Cartesian displacement is averaged over the two cells."""
    reference = Atoms("C", scaled_positions=[[0.1, 0.0, 0.0]], cell=[2.0] * 3)
    relaxed = Atoms("C", scaled_positions=[[0.2, 0.0, 0.0]], cell=[4.0] * 3)
    reference.pbc = True
    relaxed.pbc = True

    rmscd, rmscd_no_h, displacements = rms_cartesian_displacement(reference, relaxed)

    assert rmscd == pytest.approx(0.3)
    assert rmscd_no_h == pytest.approx(0.3)
    assert displacements == pytest.approx([0.3])


def test_rmscd_rejects_different_atom_order() -> None:
    """RMSCD requires matching atoms in matching order."""
    reference = Atoms("CH", positions=np.zeros((2, 3)), cell=[5.0] * 3)
    relaxed = Atoms("HC", positions=np.zeros((2, 3)), cell=[5.0] * 3)

    with pytest.raises(ValueError, match="different atom order"):
        rms_cartesian_displacement(reference, relaxed)
