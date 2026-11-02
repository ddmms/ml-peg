"""Calculation of energies in the Surface Alloy Reorganisation dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_surface_alloy_reorganisation(mlip: tuple[str, Any]) -> None:
    """
    Run energy tests for surface alloy reorganisation.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    # Download X23 dataset
    surface_alloy_reorganisation_dir = (
        download_s3_data(
            key="inputs/surface_alloy_reorganisation/surface_alloy_reorganisation.zip",
            filename="surface_alloy_reorganisation.zip",
        )
        / "surface_alloy_reorganisation"
    )

    hosts_list = ["Ag", "Cu"]
    mechanisms_list = [
        "diffusion_over_step",
        "step_lift_in",
        "terrace_pop_in",
        "terrace_slide_in",
        "vacancy_mediated_diffusion",
    ]
    states_list = ["initial_state", "final_state", "transition_state"]

    for host in hosts_list:
        for mechanism in mechanisms_list:
            # Ignore trivial calculations of reaction energy

            for state in states_list:
                if host == "Cu" and mechanism == "diffusion_over_step":
                    elements_list = ["Ag", "Cu", "Nb", "Pd", "Rh", "Ru", "Tc", "Zr"]
                elif host == "Ag" and mechanism == "diffusion_over_step":
                    elements_list = [
                        "Ag",
                        "Cu",
                        "Mo",
                        "Nb",
                        "Pd",
                        "Rh",
                        "Ru",
                        "Tc",
                        "Zr",
                    ]
                elif host == "Cu" and mechanism == "step_lift_in":
                    elements_list = [
                        "Cu",
                        "Mo",
                        "Nb",
                        "Pd",
                        "Rh",
                        "Ru",
                        "Tc",
                        "Y",
                        "Zr",
                    ]
                else:
                    elements_list = [
                        "Ag",
                        "Cu",
                        "Mo",
                        "Nb",
                        "Pd",
                        "Rh",
                        "Ru",
                        "Tc",
                        "Y",
                        "Zr",
                    ]

                for element in elements_list:
                    path_init = (
                        surface_alloy_reorganisation_dir
                        / host
                        / mechanism
                        / state
                        / element
                    )

                    structure_path = path_init / "POSCAR"
                    initial_energy_ref_path = path_init / "pot_en"

                    slab = read(structure_path, index=0, format="vasp")

                    slab.info["charge"] = 0
                    slab.info["spin"] = 1
                    slab.calc = calc

                    try:
                        slab.get_potential_energy()
                    except Exception as exc:
                        warn(f"Error calculating {model_name}: {exc}", stacklevel=2)
                        slab.info["energy"] = np.nan

                    ref = np.loadtxt(initial_energy_ref_path)

                    filename = f"{host}_{mechanism}_{state}_{element}"

                    identifier = f"{host}_{mechanism}_{element}"

                    slab.info["ref"] = ref
                    slab.info["system"] = identifier

                    # Write output structures
                    write_dir = OUT_PATH / model_name
                    write_dir.mkdir(parents=True, exist_ok=True)
                    write(write_dir / f"{filename}.xyz", slab)
