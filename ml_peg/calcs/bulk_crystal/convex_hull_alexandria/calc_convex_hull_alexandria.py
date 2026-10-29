"""Run calculations for the Alexandria convex-hull benchmark."""

from __future__ import annotations

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
OUT_PATH = Path(__file__).parent / "outputs"


def get_data_path() -> Path:
    """
    Download the benchmark inputs.

    Returns
    -------
    Path
        Path to the extracted input directory.
    """
    return (
        download_s3_data(
            key="inputs/bulk_crystal/convex_hull_alexandria/convex_hull_alexandria.zip",
            filename="convex_hull_alexandria.zip",
        )
        / "convex_hull_alexandria"
    )


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_convex_hull_alexandria(mlip: tuple[str, Any]) -> None:
    """
    Compute single-point energies of the Alexandria unaries and binaries.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    structures = read(get_data_path() / "structures.extxyz", index=":")

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for atoms in tqdm(structures, desc=model_name):
        atoms.info.update({"charge": 0, "spin": 1})
        atoms.calc = calc
        try:
            energy = atoms.get_potential_energy()
        except Exception as exc:
            warn(
                f"Error calculating {atoms.info['mat_id']} with {model_name}: {exc}",
                stacklevel=2,
            )
            energy = np.nan
        atoms.calc = None
        atoms.info["mlip_energy"] = energy
        results.append(atoms)

    write(write_dir / "structures.extxyz", results)
