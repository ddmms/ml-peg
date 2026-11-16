"""Calculation of energies in the Surface Alloy Reorganisation dataset."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

HOSTS = ["Ag", "Cu"]
MECHANISMS = [
    "diffusion_over_step",
    "step_lift_in",
    "terrace_pop_in",
    "terrace_slide_in",
    "vacancy_mediated_diffusion",
]
STATES = ["initial_state", "final_state", "transition_state"]
ALL_ELEMENTS = ["Ag", "Cu", "Mo", "Nb", "Pd", "Rh", "Ru", "Tc", "Y", "Zr"]
# Host/mechanism pairs for which only a subset of elements is available
ELEMENTS_SUBSETS = {
    ("Cu", "diffusion_over_step"): ["Ag", "Cu", "Nb", "Pd", "Rh", "Ru", "Tc", "Zr"],
    ("Ag", "diffusion_over_step"): [
        "Ag",
        "Cu",
        "Mo",
        "Nb",
        "Pd",
        "Rh",
        "Ru",
        "Tc",
        "Zr",
    ],
    ("Cu", "step_lift_in"): ["Cu", "Mo", "Nb", "Pd", "Rh", "Ru", "Tc", "Y", "Zr"],
}


def get_systems() -> list[tuple[str, str, str, str]]:
    """
    Get all (host, mechanism, state, element) combinations in the dataset.

    Returns
    -------
    list[tuple[str, str, str, str]]
        Host, mechanism, state and element for every structure to evaluate.
    """
    return [
        (host, mechanism, state, element)
        for host, mechanism, state in product(HOSTS, MECHANISMS, STATES)
        for element in ELEMENTS_SUBSETS.get((host, mechanism), ALL_ELEMENTS)
    ]


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

    # Download surface alloy reorganisation dataset
    surface_alloy_reorganisation_dir = (
        download_s3_data(
            key="inputs/surface_alloy_reorganisation/surface_alloy_reorganisation.zip",
            filename="surface_alloy_reorganisation.zip",
        )
        / "surface_alloy_reorganisation"
    )

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(get_systems(), desc=f"{model_name} surface alloy reorganisation")
    for host, mechanism, state, element in progress:
        filename = f"{host}_{mechanism}_{state}_{element}"
        progress.set_postfix_str(filename)

        path_init = surface_alloy_reorganisation_dir / host / mechanism / state / element

        # Missing or unreadable inputs leave no output file, which analysis maps
        # to NaN for every observable that depends on this structure.
        try:
            slab = read(path_init / "POSCAR", index=0, format="vasp")
        except Exception as exc:
            warn(
                f"Failed to read structure for {model_name} {filename}: {exc}",
                stacklevel=2,
            )
            continue

        try:
            ref = float(np.loadtxt(path_init / "pot_en"))
        except Exception as exc:
            warn(
                f"Failed to read reference energy for {model_name} {filename}: {exc}",
                stacklevel=2,
            )
            ref = np.nan

        slab.info["charge"] = 0
        slab.info["spin"] = 1
        slab.calc = calc

        try:
            slab.get_potential_energy()
        except Exception as exc:
            warn(f"Error calculating {model_name} {filename}: {exc}", stacklevel=2)
            slab.calc = None
            slab.info["energy"] = np.nan

        slab.info["ref"] = ref
        slab.info["system"] = f"{host}_{mechanism}_{element}"

        # Write output structures
        write(write_dir / f"{filename}.xyz", slab)
