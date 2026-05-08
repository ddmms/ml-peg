"""LAMMPS data → ASE conversion helper for the polymer-cell builder."""

from __future__ import annotations

import pathlib
import typing as ty

import ase
import ase.data
import ase.io.lammpsdata
import numpy as np


def _atomic_numbers_from_masses(atoms: ase.Atoms) -> ty.Sequence[int]:
    """
    Recover atomic numbers by matching each atom's mass to ``ase.data``.

    LAMMPS data files identify atom types by mass. EMC writes its own masses,
    which can differ slightly from ``ase.data.atomic_masses``; we pick the
    closest standard mass for each atom.
    """
    return [
        int(np.argmin(np.abs(ase.data.atomic_masses - m))) for m in atoms.get_masses()
    ]


def lammps_data_to_atoms(lammps_data_path: pathlib.Path) -> ase.Atoms:
    """
    Convert an EMC LAMMPS data file into an :class:`ase.Atoms` object.

    Strips the LAMMPS-specific per-atom arrays that ``read_lammps_data``
    injects (atom IDs, type indices, molecule IDs, partial charges, per-atom
    masses, and the bonded-topology strings ``bonds`` / ``angles`` /
    ``dihedrals`` / ``impropers``) so the resulting extxyz carries only
    chemical species and positions. The LAMMPS ``comment`` info entry is
    preserved.

    Parameters
    ----------
    lammps_data_path
        Path to an ``atom_style full``, ``units real`` LAMMPS data file
        (the one EMC produces by default).

    Returns
    -------
    ase.Atoms
        Atoms object with atomic numbers recovered from the LAMMPS masses.
    """
    atoms = ase.io.lammpsdata.read_lammps_data(
        str(lammps_data_path), atom_style="full", units="real"
    )
    if not isinstance(atoms, ase.Atoms):
        raise TypeError(f"Expected a single Atoms object, got {type(atoms).__name__}")

    atoms.set_atomic_numbers(_atomic_numbers_from_masses(atoms))

    for key in (
        "id",
        "type",
        "mol-id",
        "initial_charges",
        "mmcharges",
        "masses",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
    ):
        atoms.arrays.pop(key, None)

    return atoms
