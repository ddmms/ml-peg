"""Evaluate MLIPs on Si interstitial NEB DFT singlepoints (energies + forces)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@dataclass(frozen=True)
class _Case:
    """Definition of a single Si interstitial NEB dataset."""

    key: str
    ref_file: str


CASES: tuple[_Case, ...] = (
    _Case(key="64_atoms", ref_file="64_atoms.extxyz"),
    _Case(key="216_atoms", ref_file="216_atoms.extxyz"),
    _Case(key="216_atoms_di_to_single", ref_file="216_atoms_di_to_single.extxyz"),
)


def _read_frames(path: Path) -> list[Atoms]:
    """
    Read all frames from an extxyz trajectory.

    Parameters
    ----------
    path
        Path to extxyz trajectory.

    Returns
    -------
    list[ase.atoms.Atoms]
        Frames as ASE atoms objects.
    """
    frames = read(path, index=":")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def _ref_energy_ev(atoms: Atoms) -> float:
    """
    Extract reference (DFT) energy in eV from a frame.

    Parameters
    ----------
    atoms
        Frame with ``ref_energy_ev`` in ``atoms.info``.

    Returns
    -------
    float
        Reference energy in eV.
    """
    if "ref_energy_ev" not in atoms.info:
        raise KeyError("Missing ref_energy_ev in reference trajectory.")
    return float(atoms.info["ref_energy_ev"])


def _ref_forces(atoms: Atoms) -> np.ndarray:
    """
    Extract reference (DFT) forces in eV/Å from a frame.

    Parameters
    ----------
    atoms
        Frame with ``ref_forces`` in ``atoms.arrays``.

    Returns
    -------
    numpy.ndarray
        Reference forces array with shape ``(n_atoms, 3)``.
    """
    if "ref_forces" not in atoms.arrays:
        raise KeyError("Missing ref_forces in reference trajectory.")
    return np.asarray(atoms.arrays["ref_forces"], dtype=float)


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_si_defect_neb_singlepoints(mlip: tuple[str, Any]) -> None:
    """
    Compare MLIP energies/forces to DFT along fixed NEB images.

    Outputs per-case trajectories containing:
    - ref_energy_ev, ref_forces
    - pred_energy_ev, pred_forces

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    for case in CASES:
        frames = _read_frames(DATA_PATH / case.ref_file)
        out_dir = OUT_PATH / case.key / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[Atoms] = []
        for atoms in frames:
            ref_energy_ev = _ref_energy_ev(atoms)
            ref_forces = _ref_forces(atoms)

            atoms_pred = atoms.copy()
            atoms_pred.calc = calc

            out_atoms = atoms.copy()
            out_atoms.info["ref_energy_ev"] = ref_energy_ev
            out_atoms.arrays["ref_forces"] = ref_forces
            out_atoms.info["pred_energy_ev"] = float(atoms_pred.get_potential_energy())
            out_atoms.arrays["pred_forces"] = np.asarray(
                atoms_pred.get_forces(), dtype=float
            )
            results.append(out_atoms)

        write(out_dir / "si_defect_neb_singlepoints.extxyz", results)
