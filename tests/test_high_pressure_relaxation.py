"""Tests for the high-pressure relaxation interface.

Relaxations use a cheap analytic calculator, as the benchmark's results depend on
the model and optimiser, which are not ML-PEG's to test.
"""

from __future__ import annotations

import bz2
import json
from pathlib import Path

from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.units import GPa
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
import pytest

from ml_peg.calcs.bulk_crystal.high_pressure_relaxation import (
    calc_high_pressure_relaxation as high_pressure,
)

PRESSURE = 50


class FailingCalculator(Calculator):
    """Calculator that fails, as an unsupported structure would."""

    implemented_properties = ("energy", "forces", "stress")

    def calculate(self, *args, **kwargs):
        """
        Fail to calculate any property.

        Parameters
        ----------
        *args
            Positional arguments from ASE.
        **kwargs
            Keyword arguments from ASE.

        Raises
        ------
        RuntimeError
            Always, in place of a calculation.
        """
        raise RuntimeError("Calculation failed")


def copper(displacement: float = 0.0):
    """
    Build a copper structure with an analytic calculator attached.

    Parameters
    ----------
    displacement
        Distance to displace one atom by, along x.

    Returns
    -------
    Atoms
        Copper structure, displaced from equilibrium by `displacement`.
    """
    atoms = bulk("Cu", cubic=True)
    atoms.positions[0] += [displacement, 0, 0]
    atoms.calc = EMT()

    return atoms


def write_entries(path: Path, structures: dict[str, Structure]) -> None:
    """
    Write structures in the format downloaded from Alexandria.

    Parameters
    ----------
    path
        Path to write the compressed entries to.
    structures
        Structures to write, keyed by material ID.
    """
    entries = [
        ComputedStructureEntry(
            structure, energy=-4.0 * len(structure), data={"mat_id": mat_id}
        ).as_dict()
        for mat_id, structure in structures.items()
    ]
    with bz2.open(path, "wt") as file:
        json.dump({"entries": entries}, file)


@pytest.fixture
def cached_data(tmp_path: Path, monkeypatch) -> dict[str, Structure]:
    """
    Provide cached pressure data, in place of downloading it.

    Parameters
    ----------
    tmp_path
        Temporary directory to write the data to.
    monkeypatch
        Pytest monkeypatch fixture.

    Returns
    -------
    dict[str, Structure]
        Structures written for each material ID.
    """
    structures = {
        "agm001": Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]]),
        "agm002": Structure(Lattice.cubic(4.0), ["Ni", "Ni"], [[0, 0, 0], [0.5, 0, 0]]),
        "agm003": Structure(Lattice.cubic(4.1), ["Al"], [[0, 0, 0]]),
    }

    for label in ("P000", "P050"):
        write_entries(tmp_path / f"{label}.json.bz2", structures)

    monkeypatch.setattr(
        high_pressure,
        "download_pressure_data",
        lambda label: tmp_path / f"{label}.json.bz2",
    )

    return structures


def test_enthalpy_is_energy_and_pressure_volume_per_atom():
    """Enthalpy per atom is calculated from the relaxed structure."""
    relaxed, converged, enthalpy = high_pressure.relax_with_pressure(
        copper(), PRESSURE, max_steps=50
    )

    assert converged
    expected = relaxed.get_potential_energy() + PRESSURE * GPa * relaxed.get_volume()
    assert enthalpy == pytest.approx(expected / len(relaxed))


def test_pressure_compresses_structure():
    """The requested pressure is applied to the structure."""
    at_zero, _, _ = high_pressure.relax_with_pressure(copper(), 0, max_steps=50)
    compressed, _, _ = high_pressure.relax_with_pressure(
        copper(), PRESSURE, max_steps=50
    )

    assert compressed.get_volume() < at_zero.get_volume()


# Displaced so that the largest force component is negative, which a convergence
# check on the signed force would mistake for a relaxed structure.
def test_unconverged_relaxation_is_reported():
    """A relaxation that runs out of steps reports no result."""
    relaxed, converged, enthalpy = high_pressure.relax_with_pressure(
        copper(displacement=0.3), PRESSURE, max_steps=1
    )

    assert relaxed is None
    assert not converged
    assert np.isnan(enthalpy)


def test_failed_relaxation_is_reported():
    """A relaxation that raises reports no result, rather than failing the run."""
    atoms = bulk("Cu")
    atoms.calc = FailingCalculator()

    with pytest.warns(UserWarning, match="Relaxation failed"):
        relaxed, converged, enthalpy = high_pressure.relax_with_pressure(
            atoms, PRESSURE
        )

    assert relaxed is None
    assert not converged
    assert np.isnan(enthalpy)


def test_structures_are_loaded_with_references(cached_data: dict[str, Structure]):
    """Structures are loaded with the reference values for the pressure."""
    structures = high_pressure.load_structures(
        "P050", n_structures=2, random_select=False
    )

    assert [structure["mat_id"] for structure in structures] == ["agm001", "agm002"]
    assert [len(structure["atoms"]) for structure in structures] == [1, 2]

    # References are per atom, so the two-atom structure halves its entry's values
    nickel = structures[1]
    reference = cached_data["agm002"]
    assert nickel["ref_energy_per_atom"] == pytest.approx(-4.0)
    assert nickel["ref_volume_per_atom"] == pytest.approx(reference.volume / 2)


def test_structure_selection_is_reproducible(cached_data: dict[str, Structure]):
    """The structures selected at random are the same for every run."""
    selected = [
        [structure["mat_id"] for structure in high_pressure.load_structures("P050", 2)]
        for _ in range(2)
    ]

    assert selected[0] == selected[1]


def test_too_many_structures_requested(cached_data: dict[str, Structure]):
    """Requesting more structures than are available is reported."""
    with pytest.raises(ValueError, match="only 3 available"):
        high_pressure.load_structures("P050", n_structures=4)
