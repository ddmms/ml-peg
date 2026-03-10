"""Run calculations for homo- and heteronuclear diatomics benchmark."""

from __future__ import annotations

from collections.abc import Iterable
import itertools
import json
from pathlib import Path

from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write
import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory for calculator outputs
OUT_PATH = Path(__file__).parent / "outputs"

# Benchmark configuration (matches historical benchmark settings)
ELEMENTS: list[str] = [symbol for symbol in chemical_symbols if symbol]
INCLUDE_HETERONUCLEAR = True
MIN_DISTANCE = 0.18
MAX_DISTANCE = 6.0
# for testing, reduce the number of points to e.g. 5
N_POINTS = 100


def _distance_grid(
    min_distance: float,
    max_distance: float,
    n_points: int,
) -> np.ndarray:
    """
    Return a uniformly spaced distance grid for a diatomic pair.

    Parameters
    ----------
    min_distance, max_distance
        Bounds of the distance grid.
    n_points
        Number of points in the grid.

    Returns
    -------
    np.ndarray
        Monotonic array of bond lengths in Ångström.
    """
    return np.linspace(min_distance, max_distance, n_points, dtype=float)


def _generate_pairs(
    elements: Iterable[str], include_hetero: bool
) -> Iterable[tuple[str, str]]:
    """
    Yield element pairs for the benchmark.

    Homonuclear combinations are yielded first to mirror the historical
    calculation order before any heteronuclear pairs.

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
    homonuclear = [(element, element) for element in sorted_elements]
    yield from homonuclear
    if include_hetero:
        yield from itertools.combinations(sorted_elements, 2)


def _safe_register_torch_slice() -> None:
    """
    Register slice objects as safe for PyTorch deserialization.

    Some torch.compile or JIT models serialize slice objects in weights.
    This prevents deserialization errors when loading such models.
    """
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
    return float(forces[1] @ direction)


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
    traj_dir = write_dir / "diatomics"
    traj_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float]] = []
    supported_pairs: set[str] = set()
    supported_elements: set[str] = set()
    failed_pairs: dict[str, str] = {}

    pairs = list(_generate_pairs(ELEMENTS, INCLUDE_HETERONUCLEAR))

    for element1, element2 in tqdm(pairs, desc=f"{model_name} diatomics", unit="pair"):
        pair_label = f"{element1}-{element2}"
        try:
            distances = _distance_grid(
                MIN_DISTANCE,
                MAX_DISTANCE,
                N_POINTS,
            )
        except Exception as exc:
            failed_pairs[pair_label] = f"distance grid failed: {exc}"
            print(f"[{model_name}] Skipping {pair_label}: {exc}")
            continue

        structures: list[Atoms] = []

        try:
            for distance in distances:
                atoms = Atoms(
                    [element1, element2],
                    positions=[(0.0, 0.0, 0.0), (0.0, 0.0, float(distance))],
                )
                # Set default charge and spin
                atoms.info.setdefault("charge", 0)
                atoms.info.setdefault("spin", 1)

                atoms.calc = calc
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()

                bond_vector = atoms.positions[1] - atoms.positions[0]
                force_parallel = _project_force(forces, bond_vector)

                atoms_copy = atoms.copy()
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
        except Exception as exc:
            failed_pairs[pair_label] = str(exc)
            print(f"[{model_name}] Skipping {pair_label}: {exc}")
            continue

        if not structures:
            failed_pairs.setdefault(pair_label, "no trajectory generated")
            continue

        write(traj_dir / f"{pair_label}.xyz", structures, format="extxyz")
        supported_pairs.add(pair_label)
        supported_elements.update({element1, element2})

    if records:
        df = pd.DataFrame.from_records(records)
        df.to_csv(write_dir / "diatomics.csv", index=False)

    metadata = {
        "supported_pairs": sorted(supported_pairs),
        "supported_elements": sorted(supported_elements),
        "failed_pairs": failed_pairs,
        "config": {
            "include_heteronuclear": INCLUDE_HETERONUCLEAR,
            "min_distance": MIN_DISTANCE,
            "max_distance": MAX_DISTANCE,
            "n_points": N_POINTS,
        },
    }
    (write_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


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
