"""Run calculations for EOS tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.filters import ExpCellFilter
from ase.lattice.cubic import (
    BodyCenteredCubic,
    FaceCenteredCubic,
    FaceCenteredCubicFactory,
    SimpleCubicFactory,
)
from ase.lattice.hexagonal import HexagonalClosedPacked, HexagonalFactory
from ase.optimize import LBFGS
import numpy as np
import pandas as pd
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "../../../../inputs/bulk_crystal/equation_of_state/"
OUT_PATH = Path(__file__).parent / "outputs"


class A15Factory(SimpleCubicFactory):
    """A factory for creating A15 lattices."""

    xtal_name = "A15"
    bravais_basis = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.5, 0.25, 0.0],
        [0.5, 0.75, 0.0],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75],
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
    ]


class C15Factory(FaceCenteredCubicFactory):
    """
    A factory for creating C15 (MgCu2-type) Laves lattices.

    Contains 2 atoms on 8a sites and 4 atoms on 16d sites in the primitive basis.
    """

    xtal_name = "C15"

    # 6-atom basis (2 'A' sites and 4 'B' sites)
    # When multiplied by 4 FCC translations, you get 24 atoms total.
    bravais_basis = [
        # A sites (8a)
        [0.000, 0.000, 0.000],
        [0.250, 0.250, 0.250],
        # B sites (16d)
        [0.625, 0.625, 0.625],
        [0.625, 0.875, 0.875],
        [0.875, 0.625, 0.875],
        [0.875, 0.875, 0.625],
    ]


class OmegaFactory(HexagonalFactory):
    """A factory for creating Omega lattices."""

    xtal_name = "Omega"
    bravais_basis = [
        [0, 0, 0],
        [1 / 3, 2 / 3, 0.5],
        [2 / 3, 1 / 3, 0.5],
    ]


A15 = A15Factory()
C15 = C15Factory()
Omega = OmegaFactory()

lattices = {
    "BCC": BodyCenteredCubic,  # cubic
    "FCC": FaceCenteredCubic,
    "A15": A15,
    "C15": C15,
    "HCP": HexagonalClosedPacked,  # hexagonal
    "OMEGA": Omega,
}


def get_lattice_constants(lattice, volume_per_atom, symbol, calc):
    """
    Calculate lattice constants for a given lattice and volume per atom.

    Parameters
    ----------
    lattice
        ASE lattice class to use for structure generation.
    volume_per_atom
        Array of volumes per atom to sample for the EOS curve.
    symbol
        Chemical symbol of the element to use for structure generation.
    calc
        ASE calculator to use for energy calculations.

    Returns
    -------
    list[float] for cubic lattices or list[tuple[float, float]] for hexagonal lattices
        List of lattice constants for each volume per atom.
    """
    if lattice in [HexagonalClosedPacked, Omega]:
        if lattice == Omega:
            # assuming ideal c/a ratio for omega is 0.6123
            ideal_ratio = 0.6323
            a0 = (
                lattice.calc_num_atoms()
                * volume_per_atom[len(volume_per_atom) // 2]
                / (ideal_ratio * np.sqrt(3) / 2)
            ) ** (1 / 3)
            lattice_name = "Omega"
        elif lattice == HexagonalClosedPacked:
            ideal_ratio = 1.63
            # create a cell assuming ideal c/a ratio of 1.63 for HCP
            a0 = (
                lattice.calc_num_atoms()
                * volume_per_atom[len(volume_per_atom) // 2]
                / (np.sqrt(2))
            ) ** (1 / 3)
            lattice_name = "HCP"
        try:
            unit_cell = lattice(symbol=symbol, latticeconstant=(a0, a0 * ideal_ratio))
            unit_cell.calc = calc
            uc_filter = ExpCellFilter(unit_cell, constant_volume=True)
            opt = LBFGS(uc_filter)
            converged = opt.run(fmax=1e-3, steps=25)
            a, c = unit_cell.cell[0, 0], unit_cell.cell[2, 2]
            ratio = c / a
            print(f"Optimized c/a ratio for {symbol} for {lattice_name}: {ratio:.4f}")
            print(
                "Difference from ideal c/a ratio: "
                + f"{(ratio - ideal_ratio) / ideal_ratio * 100:.1f} %"
            )
        except Exception as e:
            print(
                f"Error optimizing {lattice_name} structure for {symbol}: {e}, "
                + "using ideal c/a ratio instead."
            )
            ratio = ideal_ratio
        if not converged:
            print(
                f"Cell shape optimization for {symbol} "
                + f"{lattice_name} did not converge!"
            )
        if lattice == HexagonalClosedPacked:
            lattice_constants = (
                lattice.calc_num_atoms() * volume_per_atom / (np.sqrt(2))
            ) ** (1 / 3)
        elif lattice == Omega:
            lattice_constants = (
                lattice.calc_num_atoms() * volume_per_atom / (ratio * np.sqrt(3) / 2)
            ) ** (1 / 3)
        return [(a, a * ratio) for a in lattice_constants]
    # assuming cubic lattice
    return (lattice.calc_num_atoms() * volume_per_atom) ** (1 / 3)


def equation_of_state(
    calc,
    lattice,
    volumes_per_atoms,
    symbol="W",
    size=(2, 2, 2),
):
    """
    Compute the equation of state for a given element and lattice.

    Parameters
    ----------
    calc
        ASE calculator to use for energy calculations.
    lattice
        ASE lattice class to use for structure generation.
    volumes_per_atoms
        Array of volumes per atom to sample for the EOS curve.
    symbol
        Chemical symbol of the element to use for structure generation.
    size
        Size of the supercell to generate for each volume per atom.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lattice constants (A) and energies (eV/atom) arrays.
    """
    # dummy call to have calc_num_atoms available
    if lattice in [HexagonalClosedPacked, Omega]:
        lattice(symbol="W", latticeconstant=(3.16, 3.16 * 1.6123))
    else:
        lattice(symbol="W", latticeconstant=3.16)

    lattice_constants = get_lattice_constants(lattice, volumes_per_atoms, symbol, calc)

    structures = [
        lattice(latticeconstant=lc, size=size, symbol=symbol)
        for lc in lattice_constants
    ]
    for structure in structures:
        structure.calc = calc

    energies = [
        structure.get_potential_energy() / len(structure) for structure in structures
    ]

    return np.array(lattice_constants), np.array(energies)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_equation_of_state(mlip: tuple[str, Any]) -> None:
    """
    Test equation of state calculation for three BCC metals.

    Parameters
    ----------
    mlip
        Tuple of (model_name, model) as provided by pytest parametrize.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    filenames = list(DATA_PATH.glob("*DFT*"))

    for filename in filenames:
        element = filename.name.split("_")[0]
        print(f"Starting EOS calculations for {element} with model {model_name}")

        dft_data = pd.read_csv(filename, comment="#")

        volumes_per_atoms = np.linspace(
            np.round(dft_data[dft_data.columns[0]].min() * 0.95),
            np.round(dft_data[dft_data.columns[0]].max() * 1.05),
            50,
            endpoint=False,
        )
        results = {"V/atom": volumes_per_atoms}

        phases = [col.split("_")[1] for col in dft_data.columns if "Delta" in col]

        for phase in phases:
            assert phase in lattices, f"Lattice {phase} not implemented for EOS test."
            lattice = lattices[phase]

            _, energies = equation_of_state(
                calc,
                lattice,
                volumes_per_atoms,
                symbol=element,
            )

            results[f"{phase}_E"] = energies

        write_dir = OUT_PATH / model_name
        df = pd.DataFrame(results)
        output_file = write_dir / f"{element}_eos_results.csv"
        write_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
