"""Run calculations for EOS tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic, SimpleCubicFactory
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


A15 = A15Factory()

lattices = {"BCC": BodyCenteredCubic, "FCC": FaceCenteredCubic, "A15": A15}


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
    lattice(symbol="W", latticeconstant=3.16)
    lattice_constants = (volumes_per_atoms * lattice.calc_num_atoms()) ** (1 / 3)

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

    fns = list(DATA_PATH.glob("*DFT*"))

    for fn in fns:
        element = fn.name.split("_")[0]
        print(f"Starting EOS calculations for {element} with model {model_name}")

        dft_data = pd.read_csv(fn, comment="#")

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
            """'
            start_time = datetime.now()
            print(f"Start time for {phase} @ {model_name}: {start_time}")
            """
            lattice_constants, energies = equation_of_state(
                calc,
                lattice,
                volumes_per_atoms,
                symbol=element,
            )
            """
            end_time = datetime.now()
            duration = end_time - start_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"End time for {phase} @ {model_name}: {end_time}")
            print(
                f"Duration for {phase} @ {model_name}: "
                f"{hours} hours {minutes} minutes {seconds} seconds"
            )
            print(duration)
            """
            results[f"{phase}_a"] = lattice_constants
            results[f"{phase}_E"] = energies

        write_dir = OUT_PATH / model_name
        df = pd.DataFrame(results)
        output_file = write_dir / f"{element}_eos_results.csv"
        write_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
