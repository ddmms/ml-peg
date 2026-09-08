"""
Run calculations for carbon nanotube formation energies benchmark.

The 2021 ``nanotubes_formation_energy/test.py`` scores each tube against a
hardcoded isolated-atom reference (``single_atom_energy = 0.94664775`` eV, the
spin-unpolarised carbon atom) on the reference side, and the model's own
isolated-atom energy on the model side. That convention was self-consistent for
GAP-20, the model it was built around: the same offset appeared on both sides.
It does not generalise to an arbitrary model under test, whose own isolated-atom
energy carries whatever spin convention that model was trained on rather than
the reference's spin-unpolarised one. The published archive also no longer
ships an isolated atom to compute either side from.

This benchmark instead reports strain energy relative to graphene, with each
side using its own graphene energy (the model's graphene for the model series,
the DFT graphene for the reference series), exactly as ``defect_energies`` uses
each side's own pristine host:

``strain_energy_ev_per_atom = E_tube / n_tube - E_graphene / n_graphene``

The isolated-atom term cancels within each side and never appears. Graphene is
a fixed reference value, not a scored system, and no output structure is
written for it.

Every reference geometry is a single-point DFT calculation (``NSW = 1`` in all
20 shipped INCARs); the model is evaluated as a single point at each shipped
geometry, with no relaxation.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"

GITHUB_URI = "https://raw.githubusercontent.com/patrickwrowe/Carbon_GAP/main/ml_peg_benchmark_data"

GRAPHENE_SYSTEM = "Graphene"


def chirality_of(system: str) -> str:
    """
    Get a nanotube's chirality, from its ``Nanotube_<n>_<m>`` name.

    Parameters
    ----------
    system
        Nanotube system name, e.g. "Nanotube_9_9" or "Nanotube_9_0".

    Returns
    -------
    str
        "Armchair" for an (n, n) tube, "Zigzag" for an (n, 0) tube.

    Raises
    ------
    ValueError
        If `system` is neither an armchair nor a zigzag tube name.
    """
    _, n_index, m_index = system.split("_")
    if n_index == m_index:
        return "Armchair"
    if m_index == "0":
        return "Zigzag"
    raise ValueError(f"Unrecognised nanotube chirality: {system}")


def tube_diameter_angstrom(atoms: Atoms) -> float:
    """
    Measure a nanotube's diameter from its geometry.

    The shipped cell places each tube's axis along the third cell vector, with
    100 Å of vacuum on the other two; atoms near the vacuum edge wrap to the
    far side of the cell rather than sitting together, so the diameter is
    measured from positions wrapped to lie within half a cell of their
    in-plane centroid, not from the raw Cartesian spread.

    Parameters
    ----------
    atoms
        Nanotube structure to measure.

    Returns
    -------
    float
        Tube diameter in Å, twice the mean in-plane radius about the
        wrapped centroid.
    """
    scaled = atoms.get_scaled_positions(wrap=True)
    scaled[:, :2] = (scaled[:, :2] + 0.5) % 1.0 - 0.5
    cartesian = scaled @ atoms.cell[:]
    centre = cartesian[:, :2].mean(axis=0)
    radii = np.linalg.norm(cartesian[:, :2] - centre, axis=1)
    return float(2 * radii.mean())


def energy_at(atoms: Atoms, calc: Calculator, label: str) -> float:
    """
    Get the single-point potential energy of a copy of `atoms` with `calc` attached.

    Parameters
    ----------
    atoms
        Reference structure to evaluate, unmodified.
    calc
        ASE calculator to attach.
    label
        Description of the structure being evaluated, for warning messages.

    Returns
    -------
    float
        Potential energy in eV, or `np.nan` on failure.
    """
    struct = atoms.copy()
    struct.info.setdefault("charge", 0)
    struct.info.setdefault("spin", 1)
    struct.calc = copy(calc)
    try:
        return struct.get_potential_energy()
    except Exception as exc:
        warn(f"Error computing energy for {label}: {exc}", stacklevel=2)
        return np.nan


@pytest.mark.parametrize("mlip", MODELS.items())
def test_nanotube_formation_energies(mlip: tuple[str, Any]) -> None:
    """
    Run armchair and zigzag nanotube strain energy benchmark.

    Parameters
    ----------
    mlip
        Name of model and model instance to get calculator from.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_dir = (
        download_github_data(
            filename="nanotube_formation_energies.zip", github_uri=GITHUB_URI
        )
        / "nanotube_formation_energies"
    )

    with open(data_dir / "list") as file:
        systems = file.read().splitlines()
    tubes = [system for system in systems if system != GRAPHENE_SYSTEM]

    graphene = read(data_dir / GRAPHENE_SYSTEM / "reference.xyz", index=0)
    n_graphene_atoms = len(graphene)
    e_graphene_ref = graphene.info["REF_energy"]

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    variant_calcs = (
        (calc,)
        if model.trained_on_dispersion
        else (calc, model.add_d3_calculator(copy(calc)))
    )
    for variant_index, variant_calc in enumerate(variant_calcs):
        e_graphene_model = energy_at(graphene, variant_calc, GRAPHENE_SYSTEM)

        for system in tubes:
            tube = read(data_dir / system / "reference.xyz", index=0)
            n_tube_atoms = len(tube)
            e_tube_model = energy_at(tube, variant_calc, system)

            atoms = tube.copy()
            atoms.info["system"] = system
            atoms.info["chirality"] = chirality_of(system)
            atoms.info["diameter_angstrom"] = tube_diameter_angstrom(tube)
            atoms.info["strain_energy_ev_per_atom"] = (
                e_tube_model / n_tube_atoms - e_graphene_model / n_graphene_atoms
            )
            atoms.info["reference_strain_energy_ev_per_atom"] = (
                tube.info["REF_energy"] / n_tube_atoms
                - e_graphene_ref / n_graphene_atoms
            )

            write(write_dir / f"{system}.extxyz", atoms, append=variant_index > 0)
            if len(variant_calcs) == 1:
                write(write_dir / f"{system}.extxyz", atoms, append=True)
