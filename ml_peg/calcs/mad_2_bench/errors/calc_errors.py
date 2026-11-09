"""Run calculations for the MAD-2 test benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
from ase.io import read
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

S3_KEY = "inputs/mad_2_bench/errors/mad-2-bench.zip"
S3_FILENAME = "mad-2-bench.zip"

LABEL_FILES = (
    "mad2_r2scan",
    "omat24",
    "matpes_pbe",
    "matpes_r2scan",
    "mptrj",
    "alexandria",
)

OUT_PATH = Path(__file__).parent / "outputs"


def structure_key(atoms: Atoms) -> bytes:
    """
    Get a key that is equal only for identical model inputs.

    Parameters
    ----------
    atoms
        Structure of one labelled frame.

    Returns
    -------
    bytes
        Atomic numbers, cell, positions and periodic boundary conditions.
    """
    # adding 0.0 maps -0.0 to 0.0
    return b"".join(
        np.ascontiguousarray(array).tobytes()
        for array in (
            atoms.numbers,
            atoms.cell.array + 0.0,
            atoms.positions + 0.0,
            atoms.pbc,
        )
    )


def load_structures(data_path: Path) -> list[dict[str, Any]]:
    """
    Collect the distinct structures of all label files.

    Parameters
    ----------
    data_path
        Directory of the extracted inputs zip.

    Returns
    -------
    list[dict[str, Any]]
        Each structure with the (file, frame id) pairs it labels, and whether any
        of them is a bulk frame.
    """
    structures: dict[bytes, dict[str, Any]] = {}

    for name in LABEL_FILES:
        for atoms in read(data_path / f"{name}.xyz", ":"):
            entry = structures.setdefault(
                structure_key(atoms),
                {"atoms": atoms, "frames": [], "bulk": False},
            )
            entry["frames"].append([name, atoms.info["id"]])
            entry["bulk"] = entry["bulk"] or atoms.info["class"] == "bulk"

    return list(structures.values())


def predict(atoms: Atoms, calc: Calculator, with_stress: bool) -> dict[str, Any]:
    """
    Predict the energy, forces and, for bulk frames, stress of one structure.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    calc
        Calculator of the model.
    with_stress
        Whether to request the stress.

    Returns
    -------
    dict[str, Any]
        Predicted properties in eV, eV/Å and eV/Å^3, or the error message.
    """
    structure = Atoms(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    structure.info["charge"] = 0
    structure.info["spin"] = 1
    structure.calc = calc

    result: dict[str, Any] = {"natoms": len(structure)}

    try:
        result["energy"] = float(structure.get_potential_energy())
        result["forces"] = structure.get_forces().tolist()
    except Exception as error:
        return {"natoms": len(structure), "error": f"{type(error).__name__}: {error}"}

    if with_stress:
        try:
            result["stress"] = structure.get_stress(voigt=True).tolist()
        except PropertyNotImplementedError:
            pass
        except Exception as error:
            result["stress_error"] = f"{type(error).__name__}: {error}"

    return result


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_mad_2_bench_predictions(mlip: tuple[str, Any]) -> None:
    """
    Predict every distinct structure of the MAD-2 test frames with one model.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_path = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "mad-2-bench"
    structures = load_structures(data_path)

    out_file = OUT_PATH / model_name / "predictions.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # predictions already written are kept, so an interrupted run continues
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            done.update(tuple(frame) for frame in json.loads(line)["frames"])

    with out_file.open("a") as handle:
        for entry in tqdm(structures, desc=f"MAD-2 structures for {model_name}"):
            if all(tuple(frame) in done for frame in entry["frames"]):
                continue

            result = {
                "frames": entry["frames"],
                **predict(entry["atoms"], calc, entry["bulk"]),
            }
            handle.write(json.dumps(result) + "\n")
            handle.flush()
