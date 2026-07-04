"""Evaluate MLIPs on small atomistic clusters using reference forces."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from ase.atoms import Atoms
from ase.io import iread, write
import numpy as np
import pytest
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"
BENCHMARK_FILENAME = "cluster_forces.extxyz"
S3_KEY = "inputs/clusters/cluster_forces/cluster_forces.zip"
S3_FILENAME = "cluster_forces.zip"
INPUT_FILENAME = "clusters.xyz"

MAD2_FORCES_KEY = "mad2_forces"
OMOL25_FORCES_KEY = "omol25_forces"
MAD2_REF_FORCES_KEY = "mad2_ref_forces"
OMOL25_REF_FORCES_KEY = "omol25_ref_forces"
PRED_FORCES_KEY = "pred_forces"


def _cluster_data_path() -> Path:
    """
    Locate the downloaded cluster extended XYZ file.

    Returns
    -------
    Path
        Path to the cluster input file.
    """
    data_dir = download_s3_data(key=S3_KEY, filename=S3_FILENAME)
    candidates = [
        data_dir / INPUT_FILENAME,
        data_dir / "cluster_forces" / INPUT_FILENAME,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {INPUT_FILENAME} after downloading {S3_KEY}. "
        f"Searched:\n{searched}"
    )


def _count_extxyz_frames(path: Path) -> int:
    """
    Count structures in an extended XYZ file.

    Parameters
    ----------
    path
        Extended XYZ file.

    Returns
    -------
    int
        Number of structures in the file.
    """
    count = 0
    with path.open(encoding="utf8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                n_atoms = int(stripped)
            except ValueError as err:
                raise ValueError(
                    f"Expected atom-count line while reading {path}: {stripped!r}"
                ) from err
            comment = file.readline()
            if not comment:
                raise ValueError(
                    f"Unexpected end of file after frame {count} in {path}"
                )
            for _ in range(n_atoms):
                atom_line = file.readline()
                if not atom_line:
                    raise ValueError(
                        f"Unexpected end of file in atom block for frame {count} "
                        f"in {path}"
                    )
            count += 1
    return count


def _spin_multiplicity(atoms: Atoms) -> int:
    """
    Return singlet/doublet spin multiplicity for a neutral cluster.

    Parameters
    ----------
    atoms
        Cluster structure.

    Returns
    -------
    int
        Spin multiplicity.
    """
    n_electrons = int(np.sum(atoms.get_atomic_numbers()))
    return 1 if n_electrons % 2 == 0 else 2


def _set_charge_and_spin(atoms: Atoms) -> None:
    """
    Set charge and spin metadata in-place.

    Parameters
    ----------
    atoms
        Cluster structure to update.
    """
    spin_multiplicity = _spin_multiplicity(atoms)
    atoms.info["charge"] = 0
    atoms.info["spin"] = spin_multiplicity
    atoms.info["spin_multiplicity"] = spin_multiplicity


def _reference_forces(atoms: Atoms, reference_key: str) -> np.ndarray:
    """
    Get reference forces from an input cluster.

    Parameters
    ----------
    atoms
        Cluster structure.
    reference_key
        Name of the reference-force array.

    Returns
    -------
    np.ndarray
        Reference forces.
    """
    if reference_key not in atoms.arrays:
        raise KeyError(f"Missing '{reference_key}' in cluster input.")
    return np.asarray(atoms.arrays[reference_key], dtype=float)


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_cluster_forces(mlip: tuple[str, Any]) -> None:
    """
    Evaluate MLIP forces for comparison to MAD2 and OMOL25 references.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip

    calc = model.get_calculator(precision="high")
    data_path = _cluster_data_path()
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = _count_extxyz_frames(data_path)
    frames = iread(data_path, index=":")
    if sys.stderr.isatty():
        frames = tqdm(
            frames,
            desc=f"{model_name} cluster forces",
            total=n_clusters,
            unit="cluster",
        )

    results: list[Atoms] = []
    failed_calculations = 0
    for structure_index, atoms in enumerate(frames):
        atoms_pred = atoms.copy()
        _set_charge_and_spin(atoms_pred)
        atoms_pred.calc = calc

        try:
            pred_forces = np.asarray(atoms_pred.get_forces(), dtype=float)
            calculation_failed = False
            calculation_error = ""
        except Exception as err:
            pred_forces = np.full((len(atoms_pred), 3), np.nan, dtype=float)
            calculation_failed = True
            calculation_error = f"{type(err).__name__}: {err}"
            failed_calculations += 1

        out_atoms = atoms.copy()
        _set_charge_and_spin(out_atoms)
        out_atoms.info["structure_index"] = structure_index
        out_atoms.info["calculation_failed"] = calculation_failed
        if calculation_error:
            out_atoms.info["calculation_error"] = calculation_error[:500]

        out_atoms.arrays[MAD2_REF_FORCES_KEY] = _reference_forces(
            out_atoms, MAD2_FORCES_KEY
        )
        out_atoms.arrays[OMOL25_REF_FORCES_KEY] = _reference_forces(
            out_atoms, OMOL25_FORCES_KEY
        )
        out_atoms.arrays[PRED_FORCES_KEY] = pred_forces
        out_atoms.arrays.pop(MAD2_FORCES_KEY, None)
        out_atoms.arrays.pop(OMOL25_FORCES_KEY, None)
        results.append(out_atoms)

    if not results:
        raise ValueError(f"No clusters found in {data_path}.")

    write(out_dir / BENCHMARK_FILENAME, results)
    print(
        f"{model_name}: wrote {len(results)} clusters; "
        f"{failed_calculations} force evaluations failed and were stored as NaNs."
    )
