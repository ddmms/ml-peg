"""Run calculations for rotational symmetry (equivariance) physicality test."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.build import bulk, graphene, molecule
from ase.io import write
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# The rotations form a single cumulative walk through orientation space: each
# step composes a rotation onto the previous orientation, so the relative
# rotation between successive frames is exactly the step applied. Successive
# orientations are compared pairwise during analysis, and two uniformly
# random orientations are almost never within 40° of each other, so the first
# steps cover the small and intermediate relative angles (a smooth model
# must fail gently at small relative angles; an abrupt small-angle violation
# signals an implementation artifact rather than learned approximate
# symmetry). The remaining steps are uniformly random rotations (Haar measure
# on SO(3)), which sample orientation space rather than trusting hand-picked
# axes and angles, and generically avoid the special rotations (90°, 180°
# about a coordinate axis) that permute or negate coordinates exactly in
# floating point and can cancel real violations out of the comparison.
# Everything is seeded, so every model sees the identical walk.
N_RANDOM = 100
STEP_ANGLES = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    14.0,
    20.0,
    28.0,
    40.0,
)


def _build_rotations() -> Rotation:
    """
    Build the seeded walk of orientations applied to every structure.

    Each step composes a rotation onto the previous orientation: STEP_ANGLES
    rotations about random axes, then N_RANDOM uniformly random rotations.

    Returns
    -------
    Rotation
        The walk's orientations, in order.
    """
    rng = np.random.default_rng(42)
    axes = rng.normal(size=(len(STEP_ANGLES), 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    steps = Rotation.concatenate(
        [
            Rotation.from_rotvec(np.radians(STEP_ANGLES)[:, None] * axes),
            Rotation.random(N_RANDOM, random_state=42),
        ]
    )

    orientations = []
    current = Rotation.identity()
    for step in steps:
        current = step * current
        orientations.append(current)

    walk = Rotation.concatenate(orientations)

    # An imperfect R would masquerade as a model symmetry violation.
    for matrix in walk.as_matrix():
        assert np.abs(matrix @ matrix.T - np.eye(3)).max() < 1e-12
        assert abs(np.linalg.det(matrix) - 1) < 1e-12

    return walk


ROTATIONS = _build_rotations()
N_ROTATIONS = len(STEP_ANGLES) + N_RANDOM


# Graphene is made fully periodic (pbc=True, with vacuum in z) rather than a 2D
# slab: the ORB and UMA calculators reject periodicity along a subset of axes,
# and the resulting 40 Angstrom z cell decouples the images so the forces
# (and hence the rotational symmetry) are identical to a true 2D
# (pbc=[True, True, False]) sheet. The vacuum matches the translational
# symmetry test, keeping the two benchmarks' structure sets aligned.
_graphene = graphene(formula="C2", a=2.46, thickness=0.0, vacuum=20.0)
_graphene.pbc = True

# Diamond, rattled with a fixed seed so the atoms carry nonzero forces: a
# perfect crystal's forces vanish by symmetry, which would leave the periodic
# structures probing energies only. Carbon is also supported by every model,
# unlike fcc aluminium, which is outside the element coverage of the
# molecular models.
_diamond = bulk("C", "diamond", a=3.567)
_diamond.rattle(stdev=0.05, seed=42)

# Ten diverse, low-cost structures: varied bond orders, elements, geometries,
# and two periodic systems (3D crystal, 2D sheet), matching the systems used
# in the translational symmetry physicality test.
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


def evaluate(calc, struct: Atoms, angle: float, rot: np.ndarray) -> Atoms:
    """
    Evaluate energy and forces for `struct`, storing them as info/arrays.

    Alongside the raw forces, the forces rotated back into the original
    (unrotated) frame are stored as "forces_original_frame": forces are
    equivariant, so for a perfect model this array is identical for every
    frame. Comparing these stored arrays lets the .xyz format's 8-decimal
    truncation cancel out of the analysis, rather than setting a spurious
    ~1e-5 meV/Å floor under every model.

    Parameters
    ----------
    calc
        ASE calculator to evaluate energy and forces with.
    struct
        Structure to evaluate. Modified in place.
    angle
        Angle of the rotation already applied to `struct`, in degrees. 0.0
        for the unrotated original.
    rot
        Rotation matrix already applied to `struct`. Identity for the
        unrotated original.

    Returns
    -------
    Atoms
        `struct`, with "energy"/"angle" info and "forces"/
        "forces_original_frame" arrays attached.
    """
    struct.calc = calc

    try:
        energy = struct.get_potential_energy()
        forces = struct.get_forces()
    except Exception as exc:
        warn(f"Error calculating energy/forces: {exc}", stacklevel=2)
        energy = np.nan
        forces = np.full((len(struct), 3), np.nan)

    struct.info["energy"] = energy
    struct.info["angle"] = angle
    struct.new_array("forces", forces)
    struct.new_array("forces_original_frame", forces @ rot)
    struct.calc = None
    return struct


@pytest.mark.parametrize("mlip", MODELS.items())
def test_rotational_symmetry(mlip: tuple[str, Any]) -> None:
    """
    Run rotational symmetry test.

    Evaluates energy and forces for ten diverse structures, then again after
    each of N_ROTATIONS random rigid rotations about the origin. For periodic
    structures the cell is rotated together with the positions, preserving
    fractional coordinates: a rotation of the positions alone is not a
    symmetry of a periodic crystal. Forces are equivariant rather than
    invariant, so each frame also stores its forces rotated back into the
    original frame; comparing rotated to original is done during analysis,
    not here.

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
        original = evaluate(calc, original, angle=0, rot=np.eye(3))

        frames = [original]
        for rotation in ROTATIONS:
            rot = rotation.as_matrix()
            angle = float(np.degrees(np.linalg.norm(rotation.as_rotvec())))
            rotated = struct.copy()
            rotated.info["charge"] = 0
            rotated.info["spin"] = 1
            rotated.set_cell(rotated.cell[:] @ rot.T, scale_atoms=False)
            rotated.positions = rotated.positions @ rot.T
            frames.append(evaluate(calc, rotated, angle=angle, rot=rot))

        write(write_dir / f"{struct_name}.xyz", frames)
