"""Run calculations for Al3Ti polymorph tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
import shutil
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import LBFGS
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

STRUCTURE_NAMES = ("Al_fcc", "Ti_hcp", "D022", "D023", "L12")

MAX_STEPS = 1000
MAX_STEP_ANGSTROM = 0.1
FORCE_TOLERANCE_EV_A = 1.0e-6

_S3_KEY = "inputs/alloy_metallurgy/al3ti_polymorphs/al3ti_polymorphs.zip"
_S3_FILENAME = "al3ti_polymorphs.zip"


@pytest.fixture
def data_path() -> Path:
    """
    Download and cache the benchmark input data from S3.

    Returns
    -------
    Path
        Path to the extracted ``al3ti_polymorphs`` directory.
    """
    path = download_s3_data(key=_S3_KEY, filename=_S3_FILENAME) / "al3ti_polymorphs"
    shutil.copytree(path / "references", OUT_PATH / "references", dirs_exist_ok=True)

    return path


def load_structure(structure: str, data_path: Path) -> Atoms | None:
    """
    Load a structure from a cif file.

    Parameters
    ----------
    structure
        Name of the structure to be loaded.
    data_path
        Path to the root of the data directory.

    Returns
    -------
    Atoms
        Loaded structure.
    None
        If loading failed.
    """
    structure_path = data_path / f"{structure}.cif"
    try:
        atoms = read(structure_path)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        return atoms
    except Exception as exc:
        warn(f"Error loading structure {structure}: {exc}", stacklevel=2)
        return None


def relax_structure(structure: str, atoms: Atoms, calc: Calculator) -> Atoms | None:
    """
    Relax a structure.

    Parameters
    ----------
    structure
        Name of the structure to be relaxed.
    atoms
        Structure to be relaxed.
    calc
        ASE calculator used to relax the structure.

    Returns
    -------
    Atoms
        Relaxed structures.
    None
        If the relaxation failed.
    """
    try:
        atoms.calc = copy(calc)
        optimizer = LBFGS(
            FrechetCellFilter(atoms), logfile=None, maxstep=MAX_STEP_ANGSTROM
        )
        converged = optimizer.run(
            fmax=FORCE_TOLERANCE_EV_A,
            steps=MAX_STEPS,
        )
        if not converged:
            warn(f"Relaxation did not converge for structure {structure}", stacklevel=2)
            return None
        atoms.get_potential_energy()
        return atoms
    except Exception as exc:
        warn(f"Error relaxing structure {structure}: {exc}", stacklevel=2)
        return None


def write_structure(structure: str, atoms: Atoms, output_dir: Path) -> bool:
    """
    Write a structure to a xyz file.

    Parameters
    ----------
    structure
        Name of the structure to be written.
    atoms
        Structure to be written.
    output_dir
        Path to the output directory.

    Returns
    -------
    bool
        True if the structure has been written successfully. False if writing failed.
    """
    try:
        write(output_dir / f"{structure}.xyz", atoms)
        return True
    except Exception as exc:
        warn(f"Error writing structure {structure}: {exc}", stacklevel=2)
        return False


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_al3ti_polymorphs(mlip: tuple[str, Any], data_path: Path) -> None:
    """
    Run the polymorph relaxation.

    Parameters
    ----------
    mlip
        Model name and model instance used to get an ASE calculator.
    data_path
        Path to the root of the data directory.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    output_dir = OUT_PATH / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for structure in STRUCTURE_NAMES:
        atoms = load_structure(structure, data_path)
        if atoms is None:
            failures.append(f"{structure}: load")
            continue
        atoms = relax_structure(structure, atoms, calc)
        if atoms is None:
            failures.append(f"{structure}: relax")
            continue
        if not write_structure(structure, atoms, output_dir):
            failures.append(f"{structure}: write")

    if failures:
        pytest.fail("Polymorph calculation failed: " + "; ".join(failures))
