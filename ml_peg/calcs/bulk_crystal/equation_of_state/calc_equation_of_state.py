"""Run calculations for EOS tests."""

from __future__ import annotations
from copy import copy
from pathlib import Path
from typing import Any
from datetime import datetime

from ase import units
from ase.io import read, write

import pandas as pd
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models


from ase.lattice.cubic import SimpleCubicFactory, \
                              FaceCenteredCubic, BodyCenteredCubic

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"



class A15Factory(SimpleCubicFactory):
    "A factory for creating A15 lattices."
    xtal_name = "A15"
    bravais_basis = [[0, 0, 0],
                     [0.5, 0.5, 0.5],
                     [0.5, 0.25, 0.0],
                     [0.5, 0.75, 0.0],
                     [0.0, 0.5, 0.25],
                     [0.0, 0.5, 0.75],
                     [0.25, 0.0, 0.5],
                     [0.75, 0.0, 0.5]]


A15 = A15Factory()

lattices = {"BCC": BodyCenteredCubic,
            "FCC": FaceCenteredCubic,
            "A15": A15}


def equation_of_state(calc, lattice, symbol="W", size = (2, 2, 2),
                      volumes_per_atoms=np.linspace(12, 22, 10, endpoint=False)):
    """Compute the equation of state for a given element and lattice.
    """
    
    # dummy call to have calc_num_atoms available
    lattice(symbol="W", latticeconstant=3.16) 
    lattice_constants = (volumes_per_atoms * lattice.calc_num_atoms()) ** (1 / 3)

    structures = [lattice(latticeconstant=lc, size=size, symbol=symbol) for lc in lattice_constants]
    for structure in structures:
        structure.calc = calc

    energies = [structure.get_potential_energy() / len(structure) for structure in structures]

    return np.array(lattice_constants), np.array(energies)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_equation_of_state(mlip: tuple[str, Any]) -> None:
    """Test equation of state calculation.
    For the moment only for three BCC metals"""

    model_name, model = mlip
    calc = model.get_calculator()

    volumes_per_atoms = np.linspace(12, 22, 100, endpoint=False)
    results = {"V/atom": volumes_per_atoms}

    elements = ["W", "Mo", "Nb"]

    for element in elements:
        for lattice_name, lattice in lattices.items():
            start_time = datetime.now()
            print(f"Start time for {lattice_name} @ {model_name}: {start_time}")
            lattice_constants, energies = equation_of_state(
                calc, lattice, symbol=element, volumes_per_atoms=volumes_per_atoms
            )
            end_time = datetime.now()
            duration = end_time - start_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"End time for {lattice_name} @ {model_name}: {end_time}")
            print(f"Duration for {lattice_name} @ {model_name}: {hours} hours {minutes} minutes {seconds} seconds")
            print(duration)


            results[f"{element}_{lattice_name}_a"] = lattice_constants
            results[f"{element}_{lattice_name}_E"] = energies

    write_dir = OUT_PATH / model_name
    df = pd.DataFrame(results)
    output_file = write_dir / "eos_results.csv"
    write_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)