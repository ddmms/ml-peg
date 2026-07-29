"""Run calculations for Jacobian symmetry (lambda) physicality test."""

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

EPS = 1e-3  # finite-difference step size, Angstrom

# Ten diverse, low-cost structures: varied bond orders, elements and
# geometries, plus two periodic systems (an fcc metal and a 2D sheet) to check
# the metric generalises beyond isolated molecules.
#
# Graphene is made fully periodic (pbc=True, with vacuum in z) rather than a 2D
# slab: the ORB and UMA calculators reject periodicity along a subset of axes,
# and the ~20 Angstrom vacuum decouples the z-images so the forces (and hence
# the Jacobian) are identical to a true 2D (pbc=[True, True, False]) sheet.
_graphene = graphene(formula="C2", a=2.46, thickness=0.0, vacuum=10.0)
_graphene.pbc = True

STRUCTURES = {
    "H2O": molecule("H2O"),
    "CH4": molecule("CH4"),
    "NH3": molecule("NH3"),
    "C2H4": molecule("C2H4"),
    "C2H2": molecule("C2H2"),
    "SO2": molecule("SO2"),
    "CH3OH": molecule("CH3OH"),
    "C6H6": molecule("C6H6"),
    "Al_fcc": bulk("Al", "fcc", a=4.05, cubic=True),
    "graphene": _graphene,
}


def perturbed_structures(calc, struct: Atoms, eps: float = EPS) -> list[Atoms]:
    """
    Build perturbed configurations of `struct` with forces evaluated at each.

    Parameters
    ----------
    calc
        ASE calculator to evaluate forces with.
    struct
        Structure to perturb.
    eps
        Finite-difference step size, in Angstrom.

    Returns
    -------
    list[Atoms]
        One structure per perturbation (2 per degree of freedom), with forces
        and dof/sign metadata attached.
    """
    positions = struct.get_positions()
    n_dof = 3 * len(struct)

    frames = []
    for dof in range(n_dof):
        atom_idx, coord_idx = divmod(dof, 3)

        for sign in (1, -1):
            frame = struct.copy()
            new_positions = positions.copy()
            new_positions[atom_idx, coord_idx] += sign * eps
            frame.set_positions(new_positions)
            frame.calc = calc

            try:
                forces = frame.get_forces()
            except Exception as exc:
                warn(f"Error calculating forces: {exc}", stacklevel=2)
                forces = np.full((len(frame), 3), np.nan)

            frame.new_array("forces", forces)
            frame.info["dof"] = dof
            frame.info["sign"] = sign
            frame.calc = None
            frames.append(frame)

    return frames


@pytest.mark.parametrize("mlip", MODELS.items())
def test_jacobian_symmetry(mlip: tuple[str, Any]) -> None:
    """
    Run Jacobian symmetry test.

    Perturbs each degree of freedom of ten diverse structures by +/- EPS,
    evaluating forces at each perturbed configuration. The finite-difference
    Jacobian itself is built during analysis, not here.

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
        struct = struct.copy()
        struct.info["charge"] = 0
        struct.info["spin"] = 1

        frames = perturbed_structures(calc, struct)
        write(write_dir / f"{struct_name}.xyz", frames)
