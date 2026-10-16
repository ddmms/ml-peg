"""Run calculation for permutation symmetry (invariance) physicality test."""

from __future__ import annotations

from itertools import permutations, product
from math import factorial, prod
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.build import bulk, graphene, molecule
from ase.io import write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Maximum number of (non-identity) permutations evaluated per structure. If a
# structure has more permutations of identical atoms than this, this many are
# drawn at random (with PERMUTATION_SEED) instead of enumerating them all.
MAX_PERMUTATIONS = 10
PERMUTATION_SEED = 42


# Graphene is made fully periodic (pbc=True, with vacuum in z) rather than a 2D
# slab: the ORB and UMA calculators reject periodicity along a subset of axes,
# and the large z cell decouples the images so the forces are identical to a
# true 2D (pbc=[True, True, False]) sheet.
_graphene = graphene(formula="C2", a=2.46, thickness=0.0, vacuum=20.0)
_graphene.pbc = True

# Diamond, rattled with a fixed seed so the atoms carry forces of a realistic
# magnitude: a perfect crystal's true forces vanish by symmetry, so any
# violation that scales with the size of the forces is understated there.
# Matches the rattle used in the rotational symmetry physicality test.
_diamond = bulk("C", "diamond", a=3.567)
_diamond.rattle(stdev=0.05, seed=42)

# Ten diverse, low-cost structures: varied bond orders, elements, geometries,
# and two periodic systems (3D bulk, 2D sheet). Elements are restricted to
# those supported by every registered model (including molecule-only models),
# so a missing score always means a genuine failure rather than an
# out-of-scope element.
STRUCTURES = {
    "H2O": molecule("H2O"),
    "CH4": molecule("CH4"),
    "NH3": molecule("NH3"),
    "C2H4": molecule("C2H4"),
    "C2H2": molecule("C2H2"),
    "SO2": molecule("SO2"),
    "CH3OH": molecule("CH3OH"),
    "C6H6": molecule("C6H6"),
    "C_diamond": _diamond,
    "graphene": _graphene,
}


def generate_permutations(
    struct: Atoms,
    max_permutations: int = MAX_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> list[np.ndarray]:
    """
    Generate permutations of identical atoms (same atomic number) in `struct`.

    If the number of non-identity permutations is at most `max_permutations`,
    all of them are returned. Otherwise, `max_permutations` distinct, random,
    non-identity permutations are drawn using `seed`.

    Parameters
    ----------
    struct
        Structure to permute.
    max_permutations
        Maximum number of permutations to return.
    seed
        Seed for the random number generator used to draw permutations.

    Returns
    -------
    list[np.ndarray]
        Index arrays `perm`, such that `struct[perm]` is the permuted structure
        (atom `i` of the permuted structure is atom `perm[i]` of `struct`). The
        identity permutation is excluded.
    """
    numbers = struct.get_atomic_numbers()
    groups = [np.flatnonzero(numbers == z) for z in np.unique(numbers)]
    identity = np.arange(len(struct))
    n_permutations = prod(factorial(len(group)) for group in groups) - 1

    if n_permutations <= max_permutations:
        perms = []
        for group_perms in product(*(permutations(group) for group in groups)):
            perm = identity.copy()
            for group, group_perm in zip(groups, group_perms, strict=True):
                perm[group] = group_perm
            if not np.array_equal(perm, identity):
                perms.append(perm)
        return perms

    rng = np.random.default_rng(seed)
    seen = {tuple(identity)}
    perms = []
    while len(perms) < max_permutations:
        perm = identity.copy()
        for group in groups:
            perm[group] = rng.permutation(group)
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            perms.append(perm)
    return perms


def evaluate(
    calc, struct: Atoms, permutation_index: int, permutation: np.ndarray
) -> Atoms:
    """
    Evaluate energy and forces for `struct`, storing them as info/arrays.

    Parameters
    ----------
    calc
        ASE calculator to evaluate energy and forces with.
    struct
        Structure to evaluate. Modified in place.
    permutation_index
        Index of the permutation already applied to `struct`. 0 for the
        unpermuted original.
    permutation
        Permutation already applied to `struct`: atom `i` of `struct` is atom
        `permutation[i]` of the original structure.

    Returns
    -------
    Atoms
        `struct`, with "energy"/"permutation_index" info and "forces" and
        "permutation" arrays attached.
    """
    struct.calc = calc

    try:
        energy = struct.get_potential_energy()
        forces = struct.get_forces()
    except Exception as exc:
        warn(f"Error calculating energy/forces: {exc}  (In permutation symmetry calculation)", category=RuntimeWarning, stacklevel=2)
        energy = np.nan
        forces = np.full((len(struct), 3), np.nan)

    struct.info["energy"] = energy
    struct.info["permutation_index"] = permutation_index
    struct.new_array("forces", forces)
    struct.new_array("permutation", np.asarray(permutation, dtype=int))
    struct.calc = None
    return struct


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_permutation_symmetry(mlip: tuple[str, Any]) -> None:
    """
    Run permutation symmetry test.

    Evaluates energy and forces for ten diverse structures, then again for
    each permutation of identical atoms in each structure (up to
    MAX_PERMUTATIONS, drawn at random if there are more). Comparing permuted to
    original is done during analysis, not here.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    for struct_name, struct in tqdm(STRUCTURES.items(), desc=model_name):
        original = struct.copy()
        original.info["charge"] = 0
        original.info["spin"] = 1
        original = evaluate(
            calc, original, permutation_index=0, permutation=np.arange(len(struct))
        )

        frames = [original]

        for i, perm in enumerate(generate_permutations(struct), start=1):
            permuted = struct[perm]
            permuted.info["charge"] = 0
            permuted.info["spin"] = 1
            frames.append(
                evaluate(calc, permuted, permutation_index=i, permutation=perm)
            )

        write(write_dir / f"{struct_name}.xyz", frames)
