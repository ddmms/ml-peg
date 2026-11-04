"""Evaluate AM26 single points using the complete verified reference dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import copy, deepcopy
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.amorphous_materials.AM26.data import get_reference_path
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"
REFERENCE_COUNTS = {
    "carbon": 100,
    "silicon": 324,
    "silica": 80,
    "gst": 290,
    "lips": 140,
}


def load_reference_structures(data_path: str | Path | None = None) -> list[Atoms]:
    """
    Read the complete, hash-verified AM26 dataset without modifying it.

    Parameters
    ----------
    data_path
        Optional local file, read in place; otherwise resolve through the standard
        cached GitHub source or the explicit ``AM26_DATA_PATH`` override.

    Returns
    -------
    list[Atoms]
        All 934 DFT-labelled input frames in their original order.
    """
    structures = read(get_reference_path(data_path), index=":", format="extxyz")
    counts = Counter(atoms.info["system"] for atoms in structures)
    if counts != REFERENCE_COUNTS:
        raise ValueError("AM26 must contain all 934 frames with the audited counts.")
    labels = [atoms.info["label"] for atoms in structures]
    if len(set(labels)) != len(labels):
        raise ValueError("AM26 input labels must be unique.")
    return structures


def preserve_reference(atoms: Atoms, input_index: int) -> Atoms:
    """
    Copy a DFT frame and preserve its labels before calculator replacement.

    Parameters
    ----------
    atoms
        Input frame with a DFT SinglePointCalculator and original AM26 metadata.
    input_index
        Zero-based position of this frame in the input dataset.

    Returns
    -------
    Atoms
        Independent output frame with explicit reference fields and no calculator.
        Energies are in eV (total) and eV/atom; forces are in eV/angstrom.
    """
    if not isinstance(atoms.calc, SinglePointCalculator):
        raise ValueError("AM26 references must have a DFT SinglePointCalculator.")
    required = {"label", "system", "energy_per_at", "virial", "cell_origin"}
    if not required <= atoms.info.keys():
        raise ValueError("The AM26 frame is missing required reference metadata.")
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float).copy()
    if (
        not np.isfinite(energy)
        or not np.isfinite(atoms.info["energy_per_at"])
        or forces.shape != (len(atoms), 3)
        or not np.isfinite(forces).all()
    ):
        raise ValueError(
            "AM26 DFT energy and force references must be valid and finite."
        )

    output = atoms.copy()
    output.info = deepcopy(atoms.info)
    output.info["ref_energy"] = energy
    output.info["ref_energy_per_atom"] = atoms.info["energy_per_at"]
    output.set_array("ref_forces", forces)
    output.info["input_index"] = input_index
    return output


def _record_failure(
    output: Atoms, model_name: str, property_name: str, exc: Exception
) -> None:
    """Record an independent property failure with extxyz-safe context."""
    message = " ".join(f"{type(exc).__name__}: {exc}".split())
    output.info[f"{property_name}_failed"] = True
    output.info[f"{property_name}_failure"] = message
    warn(
        f"{model_name}: {output.info['system']}/{output.info['label']} "
        f"(input_index={output.info['input_index']}) {property_name} failed: {message}",
        stacklevel=2,
    )


def evaluate_structures(
    structures: Sequence[Atoms], calculator: Calculator, model_name: str
) -> list[Atoms]:
    """
    Evaluate raw model energies and forces, retaining every reference frame.

    Parameters
    ----------
    structures
        DFT-labelled frames in the original input order.
    calculator
        Successfully constructed ML-PEG calculator, requested at high precision.
    model_name
        Model identifier used in progress messages and failure warnings.

    Returns
    -------
    list[Atoms]
        Output frames in input order, each with reference fields, property failure
        flags, and predicted energy/forces in an ASE SinglePointCalculator.
    """
    outputs = []
    for index, atoms in enumerate(tqdm(structures, desc=model_name)):
        output = preserve_reference(atoms, index)
        output.info["energy_failed"] = False
        output.info["forces_failed"] = False

        # Isolate the input/output geometry and metadata from backend mutations.
        evaluated = atoms.copy()
        evaluated.info = deepcopy(atoms.info)
        evaluated.calc = copy(calculator)
        evaluated.calc.reset()

        energy = np.nan
        forces = np.full((len(atoms), 3), np.nan)
        try:
            energy = float(evaluated.get_potential_energy())
            if not np.isfinite(energy):
                raise ValueError("Non-finite predicted energy.")
        except Exception as exc:
            energy = np.nan
            _record_failure(output, model_name, "energy", exc)

        try:
            predicted_forces = np.asarray(evaluated.get_forces(), dtype=float)
            if predicted_forces.shape != (len(atoms), 3):
                raise ValueError("Predicted forces must have shape (atom_count, 3).")
            if not np.isfinite(predicted_forces).all():
                raise ValueError("Non-finite predicted forces.")
            forces = predicted_forces.copy()
        except Exception as exc:
            _record_failure(output, model_name, "forces", exc)

        # A later property call may clear the live calculator's earlier results.
        # Snapshot independent local values, including failures, only after both calls.
        output.calc = SinglePointCalculator(output, energy=energy, forces=forces)
        outputs.append(output)
    return outputs


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_am26(mlip: tuple[str, Any]) -> None:
    """
    Run the full AM26 single-point calculation for one model.

    Parameters
    ----------
    mlip
        Model identifier and standard ML-PEG model wrapper. Calculator construction
        errors propagate normally and do not produce a success artifact.
    """
    model_name, model = mlip
    structures = load_reference_structures()
    calculator = model.get_calculator(precision="high")
    outputs = evaluate_structures(structures, calculator, model_name)
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / "am26_results.extxyz", outputs, format="extxyz")
