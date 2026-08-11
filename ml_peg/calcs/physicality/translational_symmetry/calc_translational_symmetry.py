"""Run calculations for translational symmetry (invariance) physicality test."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.build import bulk, graphene, molecule
from ase.io import write
import numpy as np
import pytest

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Fixed, non-axis-aligned direction, scanned at a small and a large magnitude
# (Å). Testing both distinguishes "any shift at all breaks it" from "large
# shifts break it worse", which is not the same thing for every model.
DIRECTION = np.array([37.2, -18.9, 24.6])
DIRECTION = DIRECTION / np.linalg.norm(DIRECTION)
MAGNITUDES = (1, 40, 1000)

# Graphene is made fully periodic (pbc=True, with vacuum in z) rather than a 2D
# slab: the ORB and UMA calculators reject periodicity along a subset of axes,
# and the resulting 40 Angstrom z cell decouples the images so the forces
# (and hence the translational symmetry) are identical to a true 2D
# (pbc=[True, True, False]) sheet. This vacuum is larger than the 10 Angstrom
# used elsewhere (e.g. jacobian_symmetry) specifically so the 40 Angstrom
# translation below doesn't wrap back on itself in z: with only 10 Angstrom
# vacuum (a 20 Angstrom z cell), the effective post-wrap shift is under 1
# Angstrom, defeating the point of testing a large translation.
_graphene = graphene(formula="C2", a=2.46, thickness=0.0, vacuum=20.0)
_graphene.pbc = True

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
    "C_diamond": bulk("C", "diamond", a=3.567),
    "graphene": _graphene,
}


def evaluate(calc, struct: Atoms, shift_magnitude: float) -> Atoms:
    """
    Evaluate energy and forces for `struct`, storing them as info/arrays.

    Parameters
    ----------
    calc
        ASE calculator to evaluate energy and forces with.
    struct
        Structure to evaluate. Modified in place.
    shift_magnitude
        Magnitude of the translation already applied to `struct`, in Å. 0.0
        for the untranslated original.

    Returns
    -------
    Atoms
        `struct`, with "energy"/"shift_magnitude" info and "forces" array
        attached.
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
    struct.info["shift_magnitude"] = shift_magnitude
    struct.new_array("forces", forces)
    struct.calc = None
    return struct


@pytest.mark.parametrize("mlip", MODELS.items())
def test_translational_symmetry(mlip: tuple[str, Any]) -> None:
    """
    Run translational symmetry test.

    Evaluates energy and forces for ten diverse structures, then again after
    rigidly translating every atom along a fixed direction, at each of
    MAGNITUDES. Comparing translated to original is done during analysis,
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

    for struct_name, struct in STRUCTURES.items():
        original = struct.copy()
        original.info["charge"] = 0
        original.info["spin"] = 1
        original = evaluate(calc, original, shift_magnitude=0)

        frames = [original]
        for magnitude in MAGNITUDES:
            translated = struct.copy()
            translated.info["charge"] = 0
            translated.info["spin"] = 1
            translated.translate(DIRECTION * magnitude)
            frames.append(evaluate(calc, translated, shift_magnitude=magnitude))

        write(write_dir / f"{struct_name}.xyz", frames)
