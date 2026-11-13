"""Configure tests."""

from __future__ import annotations

from itertools import combinations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import pytest

from ml_peg import analysis, models
from ml_peg.analysis.utils import decorators


class PairSpringCalculator(Calculator):
    """
    Toy calculator that is permutation invariant and parity equivariant.

    The energy is a sum of harmonic springs over all atom pairs,
    E = sum_{i<j} 0.5 * (r_ij - 1)^2, so it depends only on the set of
    interatomic distances. Relabelling identical atoms therefore leaves the
    energy unchanged and permutes the forces with the atoms, and inverting the
    structure leaves the energy unchanged and inverts the forces. Periodic
    images are ignored.
    """

    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
        **kwargs,
    ) -> None:
        """
        Calculate pair-spring energy and analytic forces.

        Parameters
        ----------
        atoms
            Atoms object to calculate properties for.
        properties
            List of properties to calculate.
        system_changes
            List of system changes to consider for calculation.
        **kwargs
            Any additional keyword arguments.
        """
        super().calculate(atoms, properties, system_changes, **kwargs)
        positions = self.atoms.positions
        energy = 0.0
        forces = np.zeros((len(positions), 3))
        for i, j in combinations(range(len(positions)), 2):
            r_ij = positions[j] - positions[i]
            r = np.linalg.norm(r_ij)
            force_vector = (r - 1.0) * r_ij / r
            energy += 0.5 * (r - 1.0) ** 2
            forces[i] += force_vector
            forces[j] -= force_vector
        self.results["energy"] = energy
        self.results["forces"] = forces


@pytest.fixture
def pair_spring_calc() -> PairSpringCalculator:
    """
    Get a toy calculator that respects permutation and parity symmetry.

    Returns
    -------
    PairSpringCalculator
        Pair-spring calculator, shared by the symmetry benchmark tests.
    """
    return PairSpringCalculator()


@pytest.fixture
def fake_models(monkeypatch):
    """
    Replace the model registry with two fake models.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    """

    def get_model_names(models=None, filepath=None):
        """
        Get fake model names.

        Parameters
        ----------
        models
            Models to select, as a comma-separated list. Default is `None`,
            corresponding to all models.
        filepath
            Unused path to model definitions.

        Returns
        -------
        list[str]
            Selected model names, in the order they are defined.
        """
        all_models = ["model_1", "model_2"]
        if models is None:
            return all_models

        selected = set(models.split(","))

        return [model for model in all_models if model in selected]

    monkeypatch.setattr(decorators, "get_model_names", get_model_names)
    monkeypatch.setattr(
        decorators,
        "load_model_configs",
        lambda mlips, filepath=None: (
            {mlip: {} for mlip in mlips},
            dict.fromkeys(mlips),
        ),
    )
    monkeypatch.setattr(models, "current_models", None)
    monkeypatch.setattr(analysis, "update_results", False)


@pytest.fixture
def update_model_2(monkeypatch, fake_models):
    """
    Set up an update run, analysing only `model_2`.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    fake_models
        Fixture replacing the model registry with two fake models.
    """
    monkeypatch.setattr(models, "current_models", "model_2")
    monkeypatch.setattr(analysis, "update_results", True)


@pytest.fixture
def update_missing_model_1(monkeypatch, update_model_2):
    """
    Set up an update run with a models file that no longer defines `model_1`.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    update_model_2
        Fixture setting up an update run, analysing only `model_2`.
    """
    monkeypatch.setattr(
        decorators, "get_model_names", lambda models=None, filepath=None: ["model_2"]
    )
