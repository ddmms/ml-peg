"""Calculation of energies in the Surface Alloy Reorganisation dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.surfaces.surface_alloy_reorganisation.reactions import (
    get_systems,
)
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


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

        path_init = (
            surface_alloy_reorganisation_dir / host / mechanism / state / element
        )
        out_file = write_dir / f"{filename}.xyz"

        # Remove any output from a previous run, so a failure in this run leaves no
        # output file, which analysis maps to NaN for every dependent observable.
        try:
            out_file.unlink(missing_ok=True)
        except OSError as exc:
            warn(
                f"Failed to remove previous output for {model_name} {filename}, "
                f"skipping: {exc}",
                stacklevel=2,
            )
            continue

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

        # Write to a temporary file and publish only once the write succeeds, so a
        # failed write cannot leave a truncated output file
        tmp_file = out_file.with_name(f".{out_file.name}.tmp")
        try:
            write(tmp_file, slab, format="extxyz")
            tmp_file.replace(out_file)
        except Exception as exc:
            warn(f"Failed to write {model_name} {filename}: {exc}", stacklevel=2)
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                warn(
                    f"Failed to remove temporary file for {model_name} {filename}: "
                    f"{cleanup_exc}",
                    stacklevel=2,
                )
