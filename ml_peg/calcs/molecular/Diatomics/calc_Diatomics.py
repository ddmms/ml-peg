"""Run calculations for homo- and heteronuclear diatomics benchmark."""

from __future__ import annotations

from collections.abc import Iterable
import itertools
from pathlib import Path

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols, covalent_radii, vdw_alvarez
from ase.io import write
from mlipx.utils import freeze_copy_atoms
import numpy as np
import pandas as pd
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory for calculator outputs
OUT_PATH = Path(__file__).parent / "outputs"

# Benchmark configuration
ELEMENTS: list[str] = [symbol for symbol in chemical_symbols if symbol]
INCLUDE_HETERONUCLEAR = True
DISTANCE_MODE = "covalent-radius"
MIN_DISTANCE = 0.8
MAX_DISTANCE = 6.0
N_POINTS = 200

# Distance fallbacks (Å)
DISTANCE_MIN_FALLBACK = 0.5


def _is_uma_omol_calc(calc_obj) -> bool:
    """
    Return whether the calculator corresponds to the UMA oMOL task.

    Parameters
    ----------
    calc_obj
        Calculator instance obtained from the model.

    Returns
    -------
    bool
        ``True`` when the calculator is the UMA oMOL predictor, else ``False``.
    """
    try:
        module_name = getattr(calc_obj.__class__, "__module__", "").lower()
        if "fairchem" not in module_name:
            return False
        task = getattr(calc_obj, "task_name", None)
        if task is None and hasattr(calc_obj, "predictor"):
            task = getattr(calc_obj.predictor, "task_name", None)
        return task == "omol"
    except Exception:
        return False


def _vdw_radius(element: str) -> float:
    """
    Return the van der Waals radius (Å) for an element.

    Parameters
    ----------
    element
        Chemical symbol.

    Returns
    -------
    float
        Van der Waals radius in Ångström, with a conservative fallback.
    """
    z = atomic_numbers[element]
    if z < len(vdw_alvarez.vdw_radii):
        value = vdw_alvarez.vdw_radii[z]
        if not np.isnan(value):
            return float(value)
    return 2.0  # conservative fallback


def _distance_grid(
    element1: str,
    element2: str,
    mode: str,
    min_distance: float,
    max_distance: float,
    n_points: int,
) -> np.ndarray:
    """
    Return distance grid for a diatomic pair.

    Parameters
    ----------
    element1, element2
        Chemical symbols of the atoms forming the diatomic.
    mode
        Distance sampling mode. ``"covalent-radius"`` uses element radii, any
        other value falls back to an evenly spaced grid between ``min_distance``
        and ``max_distance``.
    min_distance, max_distance
        Bounds of the distance grid when ``mode`` is not ``"covalent-radius"``.
    n_points
        Number of points for the uniform grid fallback.

    Returns
    -------
    np.ndarray
        Monotonic array of bond lengths in Ångström.
    """
    if mode == "covalent-radius":
        cov1 = covalent_radii[atomic_numbers[element1]]
        cov2 = covalent_radii[atomic_numbers[element2]]
        rmin = 0.9 * float(cov1 + cov2) / 2.0

        rvdw = 0.5 * (_vdw_radius(element1) + _vdw_radius(element2))
        rmax = max(rmin + 0.5, 3.1 * rvdw)
        rmin = max(DISTANCE_MIN_FALLBACK, rmin)
        step = 0.01
        n_samples = max(int(np.ceil((rmax - rmin) / step)), 2)
        return np.linspace(rmin, rmax, n_samples, dtype=float)

    return np.linspace(min_distance, max_distance, n_points, dtype=float)


def _generate_pairs(
    elements: Iterable[str], include_hetero: bool
) -> Iterable[tuple[str, str]]:
    """
    Yield element pairs for the benchmark.

    Parameters
    ----------
    elements
        Iterable of element symbols.
    include_hetero
        Whether to include heteronuclear combinations.

    Yields
    ------
    tuple[str, str]
        Element pairs.
    """
    sorted_elements = sorted(set(elements))
    for element in sorted_elements:
        yield element, element
    if include_hetero:
        yield from itertools.combinations(sorted_elements, 2)


def _safe_register_torch_slice() -> None:
    """Allow torch safe loader to serialise ``slice`` objects."""
    try:
        import torch

        if hasattr(torch, "serialization") and hasattr(
            torch.serialization, "add_safe_globals"
        ):
            torch.serialization.add_safe_globals([slice])
    except Exception:
        pass


def _project_force(forces: np.ndarray, bond_vector: np.ndarray) -> float:
    """
    Project the second atom's force onto the bond vector.

    Parameters
    ----------
    forces
        Array of forces for each atom.
    bond_vector
        Vector pointing from atom 0 to atom 1.

    Returns
    -------
    float
        Parallel component of the force acting on the second atom.
    """
    norm = np.linalg.norm(bond_vector)
    if norm == 0.0:
        return 0.0
    direction = bond_vector / norm
    return float(np.dot(forces[1], direction))


def run_diatomics(model_name: str, model) -> None:
    """
    Evaluate diatomic curves for a single model.

    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    model
        Model wrapper providing ``get_calculator``.
    """
    _safe_register_torch_slice()

    calc = model.get_calculator()
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float]] = []

    for element1, element2 in _generate_pairs(ELEMENTS, INCLUDE_HETERONUCLEAR):
        distances = _distance_grid(
            element1,
            element2,
            DISTANCE_MODE,
            MIN_DISTANCE,
            MAX_DISTANCE,
            N_POINTS,
        )

        pair_label = f"{element1}-{element2}"
        structures: list[Atoms] = []

        for distance in distances:
            atoms = Atoms(
                [element1, element2],
                positions=[(0.0, 0.0, 0.0), (0.0, 0.0, float(distance))],
            )
            if _is_uma_omol_calc(calc):
                atoms.info.setdefault("spin", 1)

            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()

            bond_vector = atoms.positions[1] - atoms.positions[0]
            force_parallel = _project_force(forces, bond_vector)

            atoms_copy = freeze_copy_atoms(atoms)
            atoms_copy.calc = None
            atoms_copy.info.update(
                {
                    "pair": pair_label,
                    "distance": float(distance),
                    "energy": energy,
                    "force_parallel": force_parallel,
                    "model": model_name,
                }
            )
            atoms_copy.arrays["forces"] = forces
            structures.append(atoms_copy)

            records.append(
                {
                    "pair": pair_label,
                    "element_1": element1,
                    "element_2": element2,
                    "distance": float(distance),
                    "energy": energy,
                    "force_parallel": force_parallel,
                }
            )

        if structures:
            write(write_dir / f"{pair_label}.xyz", structures, format="extxyz")

    if records:
        df = pd.DataFrame.from_records(records)
        df.to_csv(write_dir / "diatomics.csv", index=False)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_diatomics(model_name: str) -> None:
    """
    Run diatomics benchmark for each registered model.

    Parameters
    ----------
    model_name
        Name of the model to evaluate.
    """
    run_diatomics(model_name, MODELS[model_name])
