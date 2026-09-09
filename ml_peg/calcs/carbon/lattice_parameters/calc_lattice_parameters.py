"""Run calculations for carbon lattice parameters benchmark."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"

GITHUB_URI = "https://raw.githubusercontent.com/patrickwrowe/Carbon_GAP/main/ml_peg_benchmark_data"

REPEAT_FACTORS = {
    "Graphite": {"a": 6, "c": 2},
    "Graphene": {"a": 5},
    "Diamond": {"a": 3},
    "Lonsdaleite": {"a": 2, "c": 2},
    "Nanotube-(9,0)": {"c": 5},
}

MOLECULES = ("C60", "C100")

FMAX = 1e-3
FMAX_GRAPHITE = 1e-4
MAX_STEPS = 500

FIRST_SHELL_MAX_ANGSTROM = 1.6
SECOND_SHELL_MIN_ANGSTROM = 2.2
SECOND_SHELL_MAX_ANGSTROM = 2.6


def get_shell_means(atoms: Atoms) -> tuple[float, float]:
    """
    Get mean first and second neighbour shell bond lengths.

    Parameters
    ----------
    atoms
        Structure to measure interatomic distances for.

    Returns
    -------
    tuple[float, float]
        Mean first-shell and second-shell bond lengths.
    """
    distances = atoms.get_all_distances()
    first = distances[(distances > 0) & (distances < FIRST_SHELL_MAX_ANGSTROM)]
    second = distances[
        (distances >= SECOND_SHELL_MIN_ANGSTROM)
        & (distances <= SECOND_SHELL_MAX_ANGSTROM)
    ]
    return first.mean(), second.mean()


def prepare_reference(data_dir: Path, system: str) -> dict[str, Any]:
    """
    Read reference geometry and energy for one system.

    Parameters
    ----------
    data_dir
        Path to the extracted lattice_parameters benchmark data.
    system
        Name of the carbon allotrope system.

    Returns
    -------
    dict[str, Any]
        Reference values needed to relax and score `system`.
    """
    reference = read(data_dir / system / "reference.xyz", index=0)
    is_molecule = system in MOLECULES
    ref: dict[str, Any] = {
        "is_molecule": is_molecule,
        "fmax": FMAX_GRAPHITE if system == "Graphite" else FMAX,
        "energy_per_atom": reference.info["REF_energy"] / len(reference),
    }
    if is_molecule:
        ref["shell_1"], ref["shell_2"] = get_shell_means(reference)
    else:
        ref["cell_params"] = reference.cell.cellpar()
    return ref


def relax_system(
    atoms: Atoms, calc: Calculator, system: str, ref: dict[str, Any]
) -> float:
    """
    Relax one system with one calculator and record its geometry and convergence.

    Parameters
    ----------
    atoms
        Reference structure to relax in place.
    calc
        ASE calculator to attach.
    system
        Name of the carbon allotrope system.
    ref
        Reference values for `system`, from `prepare_reference`.

    Returns
    -------
    float
        Relaxed potential energy per atom in eV, or `np.nan` on failure.
    """
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    atoms.info["system"] = system
    atoms.calc = copy(calc)

    relaxed = True
    try:
        if ref["is_molecule"]:
            GeomOpt(
                struct=atoms, fmax=ref["fmax"], steps=MAX_STEPS, filter_class=None
            ).run()
        else:
            GeomOpt(struct=atoms, fmax=ref["fmax"], steps=MAX_STEPS).run()
    except Exception as exc:
        warn(f"Error relaxing {system}: {exc}", stacklevel=2)
        relaxed = False

    energy_per_atom = np.nan
    converged = False
    if relaxed:
        try:
            converged = bool(np.abs(atoms.get_forces()).max() < ref["fmax"])
            energy_per_atom = atoms.get_potential_energy() / len(atoms)
        except Exception as exc:
            warn(f"Error calculating energy for {system}: {exc}", stacklevel=2)

    atoms.info["converged"] = converged

    if ref["is_molecule"]:
        shell_1, shell_2 = get_shell_means(atoms) if relaxed else (np.nan, np.nan)
        atoms.info["bond_length_1_angstrom"] = shell_1
        atoms.info["bond_length_2_angstrom"] = shell_2
        atoms.info["ref_bond_length_1_angstrom"] = ref["shell_1"]
        atoms.info["ref_bond_length_2_angstrom"] = ref["shell_2"]
    else:
        factors = REPEAT_FACTORS[system]
        cell_params = atoms.cell.cellpar() if relaxed else None
        if "a" in factors:
            atoms.info["lattice_a_angstrom"] = (
                cell_params[0] / factors["a"] if relaxed else np.nan
            )
            atoms.info["ref_lattice_a_angstrom"] = ref["cell_params"][0] / factors["a"]
        if "c" in factors:
            atoms.info["lattice_c_angstrom"] = (
                cell_params[2] / factors["c"] if relaxed else np.nan
            )
            atoms.info["ref_lattice_c_angstrom"] = ref["cell_params"][2] / factors["c"]

    return energy_per_atom


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_lattice_parameters(mlip: tuple[str, Any]) -> None:
    """
    Run lattice parameter, bond length, and energy above graphite benchmark.

    Parameters
    ----------
    mlip
        Name of model and model instance to get calculator from.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_dir = (
        download_github_data(filename="lattice_parameters.zip", github_uri=GITHUB_URI)
        / "lattice_parameters"
    )

    with open(data_dir / "list") as file:
        systems = file.read().splitlines()

    references = {system: prepare_reference(data_dir, system) for system in systems}
    ref_graphite_energy = references["Graphite"]["energy_per_atom"]

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    variant_calcs = (
        (calc,)
        if model.trained_on_dispersion
        else (calc, model.add_d3_calculator(copy(calc)))
    )
    for variant_index, variant_calc in enumerate(variant_calcs):
        energy_per_atom = {}
        frames = {}
        for system in systems:
            atoms = read(data_dir / system / "reference.xyz", index=0)
            energy_per_atom[system] = relax_system(
                atoms, variant_calc, system, references[system]
            )
            frames[system] = atoms

        graphite_energy = energy_per_atom["Graphite"]
        for system, atoms in frames.items():
            atoms.info["energy_above_graphite_ev_per_atom"] = (
                energy_per_atom[system] - graphite_energy
            )
            atoms.info["ref_energy_above_graphite_ev_per_atom"] = (
                references[system]["energy_per_atom"] - ref_graphite_energy
            )
            write(write_dir / f"{system}.extxyz", atoms, append=variant_index > 0)
            if len(variant_calcs) == 1:
                write(write_dir / f"{system}.extxyz", atoms, append=True)
