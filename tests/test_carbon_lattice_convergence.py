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
    """
    Build a strained diamond cell with zero atomic forces but non-zero stress.

    Returns
    -------
    Atoms
        Cubic diamond cell at a = 3.75 Angstrom.
    """
    atoms = bulk("C", "diamond", a=3.75, cubic=True)
    atoms.calc = EMT()
    assert np.abs(atoms.get_forces()).max() < 1e-10
    assert np.abs(atoms.get_stress()).max() > 0.01
    return atoms


def stretched_dimer() -> Atoms:
    """
    Build a C2 dimer away from its Lennard-Jones minimum.

    Returns
    -------
    Atoms
        Two carbon atoms 1.6 Angstrom apart.
    """
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
    """
    Test the flag and residual force ratio follow the optimiser, cell included.

    Parameters
    ----------
    monkeypatch
        Pytest fixture to cap the relaxation step count.
    tmp_path
        Temporary directory to run the relaxation in.
    system
        System name, selecting the periodic or molecular branch.
    make
        Factory for the starting structure.
    calc
        Calculator class to relax with.
    is_molecule
        Whether the system is relaxed without a cell filter.
    steps
        Maximum number of optimiser steps.
    converged
        Expected convergence of the relaxation.
    """
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
