"""Run the MC500 molecular-crystal relaxation benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS
import numpy as np
import pytest
import spglib
from tqdm import tqdm

# from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Temporary local source. Replace this with the commented S3 download in
# ``test_mc500_relaxation`` once the archive has been uploaded.
LOCAL_DATA_PATH = Path(__file__).parents[4] / "data_mol_crys_michal" / "MC500_CIF"

FMAX = 0.01
MAX_STEPS = 1000
SYMPREC = 0.01


def get_spacegroup_number(atoms: Atoms, symprec: float = SYMPREC) -> int | None:
    """
    Determine the space-group number of a structure.

    Parameters
    ----------
    atoms
        Periodic structure to inspect.
    symprec
        Symmetry tolerance in Angstrom.

    Returns
    -------
    int | None
        International space-group number, or ``None`` if it cannot be determined.
    """
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return None if dataset is None else int(dataset.number)


def relax_crystal(
    atoms: Atoms,
    calculator: Any,
    *,
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
    symprec: float = SYMPREC,
) -> tuple[Atoms, bool, int, float]:
    """
    Relax atomic positions and cell while preserving the starting symmetry.

    Parameters
    ----------
    atoms
        Starting molecular-crystal structure.
    calculator
        ASE-compatible calculator.
    fmax
        Maximum force tolerance in eV/Angstrom.
    max_steps
        Maximum number of BFGS steps.
    symprec
        Symmetry tolerance in Angstrom.

    Returns
    -------
    tuple[Atoms, bool, int, float]
        Relaxed structure, convergence status, number of optimization steps, and
        final maximum atomic force in eV/Angstrom.
    """
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = calculator
    atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))

    filtered = FrechetCellFilter(atoms)
    optimizer = BFGS(filtered, logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
    max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

    relaxed = atoms.copy()
    relaxed.calc = None
    relaxed.set_constraint()
    return relaxed, converged, optimizer.nsteps, max_force


def _write_trajectory(
    path: Path,
    reference: Atoms,
    relaxed: Atoms,
    metadata: dict[str, Any],
) -> None:
    """
    Write reference and relaxed structures as a two-frame trajectory.

    Parameters
    ----------
    path
        Output trajectory path.
    reference
        Reference crystal structure.
    relaxed
        Relaxed crystal structure.
    metadata
        Metadata to store on both trajectory frames.
    """
    reference = reference.copy()
    relaxed = relaxed.copy()
    reference.info = metadata | {"frame": "r2SCAN+MBD reference"}
    relaxed.info = metadata | {"frame": "MLIP relaxed"}
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, [reference, relaxed])


@pytest.mark.parametrize("mlip", MODELS.items())
def test_mc500_relaxation(mlip: tuple[str, Any]) -> None:
    """
    Relax the 500 molecular crystals in the MC500 dataset.

    Parameters
    ----------
    mlip
        Model name and model wrapper.
    """
    model_name, model = mlip
    calculator = model.get_calculator(precision="high")
    calculator = model.add_d3_calculator(calculator)

    # cif_dir = (
    #     download_s3_data(
    #         key="inputs/molecular_crystal/MC500/MC500.zip",
    #         filename="MC500.zip",
    #     )
    #     / "MC500_CIF"
    # )
    cif_dir = LOCAL_DATA_PATH
    if not cif_dir.is_dir():
        raise FileNotFoundError(f"MC500 input directory not found: {cif_dir}")

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {cif_dir}")

    model_out = OUT_PATH / model_name
    traj_out = model_out / "structures"
    model_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for cif_file in tqdm(cif_files, desc=f"MC500: {model_name}"):
        structure_id = cif_file.stem
        refcode = structure_id.split("_", maxsplit=1)[-1]
        failure = None
        converged = False
        steps = 0
        max_force = np.nan
        output_spacegroup = None

        try:
            atoms = read(cif_file, reader="pycodcif")
            reference = atoms.copy()
            reference.info = {}
            input_spacegroup = get_spacegroup_number(reference)

            relaxed, converged, steps, max_force = relax_crystal(atoms, calculator)
            output_spacegroup = get_spacegroup_number(relaxed)

            metadata = {
                "structure_id": structure_id,
                "refcode": refcode,
                "converged": converged,
                "steps": steps,
                "max_force": max_force,
                "input_spacegroup": input_spacegroup,
                "output_spacegroup": output_spacegroup,
            }
            _write_trajectory(
                traj_out / f"{structure_id}.xyz", reference, relaxed, metadata
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            warn(f"MC500 relaxation failed for {structure_id}: {exc}", stacklevel=2)
            input_spacegroup = None

        rows.append(
            {
                "structure_id": structure_id,
                "refcode": refcode,
                "converged": converged,
                "steps": steps,
                "max_force": max_force,
                "input_spacegroup": input_spacegroup,
                "output_spacegroup": output_spacegroup,
                "failure": failure,
            }
        )

    with (model_out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
