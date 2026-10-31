"""Tests for the convergence record of carbon lattice parameter relaxations."""

from __future__ import annotations

from collections.abc import Callable

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
import numpy as np
import pytest

from ml_peg.calcs.carbon.lattice_parameters import calc_lattice_parameters

FMAX_EV_PER_ANGSTROM = 1e-3


def strained_diamond() -> Atoms:
    """Build a cubic diamond cell with zero forces but non-zero stress."""
    atoms = bulk("C", "diamond", a=3.75, cubic=True)
    atoms.calc = EMT()
    assert np.abs(atoms.get_forces()).max() < 1e-10
    assert np.abs(atoms.get_stress()).max() > 0.01
    return atoms


def stretched_dimer() -> Atoms:
    """Build a C2 dimer stretched past its Lennard-Jones minimum."""
    return Atoms("C2", positions=[[0, 0, 0], [0, 0, 1.6]])


@pytest.mark.parametrize(
    ("system", "make", "calc", "is_molecule"),
    [
        ("Diamond", strained_diamond, EMT, False),
        ("C60", stretched_dimer, LennardJones, True),
    ],
)
@pytest.mark.parametrize(("steps", "converged"), [(1, False), (500, True)])
def test_converged_flag_matches_optimiser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    system: str,
    make: Callable[[], Atoms],
    calc: type[Calculator],
    is_molecule: bool,
    steps: int,
    converged: bool,
) -> None:
    """The converged flag and residual force follow the optimiser, cell included."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(calc_lattice_parameters, "MAX_STEPS", steps)
    atoms = make()
    ref = {
        "is_molecule": is_molecule,
        "fmax": FMAX_EV_PER_ANGSTROM,
        "cell_params": atoms.cell.cellpar(),
        "shell_1": 1.4,
        "shell_2": 2.4,
    }
    calc_lattice_parameters.relax_system(atoms, calc(), system, ref)
    assert atoms.info["converged"] is converged
    ratio = atoms.info["max_force_ev_per_angstrom"] / atoms.info["fmax_ev_per_angstrom"]
    assert (ratio < 1) is converged
