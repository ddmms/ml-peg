"""Regression tests for the MLIP Audit geometry benchmark."""

from __future__ import annotations

from types import SimpleNamespace

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
import numpy as np
import pytest

pytest.importorskip("mlipaudit")
from ml_peg.analysis.molecular.reference_geometry_stability import (
    analyse_reference_geometry_stability as analysis,
)
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


def test_geometry_metrics_include_each_molecule():
    """Failed molecules score zero, while invalid archives have no score."""
    values = {
        "complete": [0.0] * 200 + [0.3] * 20,
        "failed": [None, None],
        "partial": [0.0, None],
        "invalid": None,
    }
    averages = analysis.get_avg_rmsd.__wrapped__(values)
    scores = analysis.get_score.__wrapped__(values)
    assert averages["complete"] == pytest.approx(0.3 / 11)
    assert np.isnan(averages["partial"])
    assert scores["failed"] == 0
    assert scores["partial"] == 0.5
    assert np.isnan(scores["invalid"])


def test_stale_step_zero_archive_is_rejected():
    """Legacy unconverged coordinates cannot receive a perfect score."""
    record = SimpleNamespace(
        molecule_name="mol_0",
        failed=False,
        simulation_state=SimpleNamespace(forces=[[[2, 0, 0]]]),
    )
    with pytest.raises(ValueError, match="convergence tolerance"):
        analysis._subset_rmsds({"mol_0": None}, [record])
