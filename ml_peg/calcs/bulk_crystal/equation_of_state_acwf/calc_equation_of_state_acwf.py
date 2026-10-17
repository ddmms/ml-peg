"""Run calculations for the ACWF equation-of-state benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"

DATASETS = ("unaries", "oxides")

# ACWF protocol: seven volumes, from 0.94 to 1.06 times the central volume.
VOLUME_SCALE_FACTORS = tuple(np.round(np.linspace(0.94, 1.06, 7), 2))


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
            key="inputs/bulk_crystal/equation_of_state_acwf/equation_of_state_acwf.zip",
            filename="equation_of_state_acwf.zip",
        )
        / "equation_of_state_acwf"
    )


def scaled_structures(atoms: Atoms, label: str) -> list[Atoms]:
    """
    Get the structures at each volume of the ACWF protocol.

    Parameters
    ----------
    atoms
        Structure at the central volume.
    label
        Structure label, ``{element}-{configuration}``, such as ``Fe-BCC``. A single
        label is stored because extended XYZ would read an element ``F`` as ``False``.

    Returns
    -------
    list[Atoms]
        Structures with the cell volume scaled by each factor.
    """
    structures = []
    for scale in VOLUME_SCALE_FACTORS:
        scaled = atoms.copy()
        scaled.set_cell(atoms.cell * scale ** (1 / 3), scale_atoms=True)
        scaled.info = {
            "label": label,
            "volume_scale": float(scale),
            "charge": 0,
            "spin": 1,
        }
        structures.append(scaled)
    return structures


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_equation_of_state_acwf(mlip: tuple[str, Any]) -> None:
    """
    Compute E(V) for the ACWF unaries and oxides.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    data_path = get_data_path()

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        results = []
        filenames = sorted((data_path / "structures" / dataset).glob("*.xsf"))
        for filename in tqdm(filenames, desc=f"{model_name} {dataset}"):
            atoms = read(filename)

            for struct in scaled_structures(atoms, filename.stem):
                struct.calc = calc
                try:
                    energy = struct.get_potential_energy()
                except Exception as exc:
                    warn(
                        f"Error calculating {filename.stem} at "
                        f"{struct.info['volume_scale']} V0 with {model_name}: {exc}",
                        stacklevel=2,
                    )
                    energy = np.nan
                struct.calc = None
                struct.info["mlip_energy"] = energy
                results.append(struct)

        write(write_dir / f"{dataset}.extxyz", results)
