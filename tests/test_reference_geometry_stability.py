"""Regression tests for the MLIP Audit geometry benchmark."""

from __future__ import annotations

from types import SimpleNamespace

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
import numpy as np
import pytest

pytest.importorskip("mlipaudit")
from ml_peg.calcs.molecular.reference_geometry_stability import (
    calc_reference_geometry_stability as calc,
)


def test_minimization_saves_final_atoms():
    """An optimization converging between log intervals records its last step."""
    molecule = SimpleNamespace(
        atom_symbols=["Ar", "Ar"], coordinates=[[0, 0, 0], [1.4, 0, 0]], charge=0
    )
    expected = Atoms(molecule.atom_symbols, positions=molecule.coordinates)
    expected.calc = LennardJones()
    with BFGS(expected, logfile=None) as optimizer:
        assert optimizer.run(fmax=0.01, steps=1000)
    state = calc.minimize_molecule(molecule, LennardJones())
    assert 0 < state.step == optimizer.nsteps < 1000
    np.testing.assert_allclose(state.positions[-1], expected.positions)
    np.testing.assert_allclose(state.forces[-1], expected.get_forces())
