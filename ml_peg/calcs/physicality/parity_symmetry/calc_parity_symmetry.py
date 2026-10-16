"""Run calculation for parity (inversion) symmetry physicality test."""

from __future__ import annotations

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

RATTLE_STDEV = 0.05
RATTLE_SEED = 42


def rattled(struct: Atoms) -> Atoms:
    """
    Get a copy of `struct` with its atoms randomly displaced.

    A generic displacement removes every point-group symmetry, so the structure
    becomes chiral and its inverted copy is a true mirror image, rather than a
    rotated and/or permuted copy of the original. Only then can the energy
    distinguish a model that is rotationally but not parity equivariant. A fixed
    seed ensures every model sees the same structure.

    Parameters
    ----------
    struct
        Structure to rattle. Not modified.

    Returns
    -------
    Atoms
        Rattled copy of `struct`.
    """
    struct = struct.copy()
    struct.rattle(stdev=RATTLE_STDEV, seed=RATTLE_SEED)
    return struct


# Lowest-energy conformer (alpha_002) of the Glucose205 benchmark (J. Chem.
# Theory Comput. 2016, 12, 6157-6168; DOI: 10.1021/acs.jctc.6b00876),
# hard-coded so the test needs no data download. With five stereocentres it is
# chiral even without rattling.
_glucose = Atoms(
    "C6O6H12",
    positions=[
        [-0.10540, 2.42602, -2.57449],
        [0.10423, 1.48258, -1.40924],
        [1.12005, 2.00415, -0.40810],
        [1.37451, 0.94712, 0.64018],
        [1.84176, -0.33444, -0.02095],
        [0.83530, -0.77254, -1.07489],
        [0.56712, 0.25413, -1.99441],
        [-0.31618, -1.18208, -0.36715],
        [2.07106, -1.35371, 0.93805],
        [2.35933, 1.45410, 1.52803],
        [0.57414, 3.17824, 0.16739],
        [1.08131, 2.61188, -3.32454],
        [-0.39764, 3.40026, -2.19237],
        [-0.91005, 2.03416, -3.20206],
        [-0.84700, 1.32060, -0.89547],
        [2.05625, 2.22399, -0.92998],
        [0.43860, 0.75809, 1.17719],
        [2.79697, -0.14827, -0.51055],
        [1.23272, -1.59307, -1.67196],
        [-0.90558, -1.62349, -0.98624],
        [1.19860, -1.69353, 1.17190],
        [2.60925, 0.72555, 2.10735],
        [1.15256, 3.41202, 0.90172],
        [1.37578, 1.73224, -3.58101],
    ],
)

# Periodic structures use rattled supercells: any two-atom cell (such as the
# primitive diamond and graphene cells) is centrosymmetric about the midpoint
# of its two atoms, so its inverted copy is merely a translated and permuted
# original, which tests translation and permutation symmetry rather than
# parity.
_diamond = rattled(bulk("C", "diamond", a=3.567, cubic=True))

# Graphene is made fully periodic (pbc=True, with vacuum in z) rather than a 2D
# slab: the ORB and UMA calculators reject periodicity along a subset of axes,
# and the large z cell decouples the images so the forces are identical to a
# true 2D (pbc=[True, True, False]) sheet. The rattle includes out-of-plane
# displacements, which break the mirror symmetry of the flat sheet.
_graphene = graphene(formula="C2", a=2.46, size=(2, 2, 1), thickness=0.0, vacuum=20.0)
_graphene.pbc = True
_graphene = rattled(_graphene)

# Diverse, low-cost structures: varied bond orders, elements and geometries,
# and two periodic systems (3D bulk, 2D sheet). Elements are restricted to
# those supported by every registered model (including molecule-only models),
# so a missing score always means a genuine failure rather than an
# out-of-scope element.
#
# The unrattled molecules are all achiral, so for them inversion equals a proper
# rotation combined with a permutation of identical atoms: their energies are
# unchanged by inversion for any rotationally and permutationally invariant
# model, and parity breaking can appear only in the forces, along vibrational
# modes that are odd under improper operations (e.g. the C2H4 torsion). The
# rattled and chiral structures probe parity breaking in the energy directly.
STRUCTURES = {
    "H2O": molecule("H2O"),
    "CH4": molecule("CH4"),
    "NH3": molecule("NH3"),
    "C2H4": molecule("C2H4"),
    "C2H2": molecule("C2H2"),
    "SO2": molecule("SO2"),
    "CH3OH": molecule("CH3OH"),
    "C6H6": molecule("C6H6"),
    "glucose": _glucose,
    "CH3OH_rattled": rattled(molecule("CH3OH")),
    "C6H6_rattled": rattled(molecule("C6H6")),
    "glucose_rattled": rattled(_glucose),
    "C_diamond": _diamond,
    "graphene": _graphene,
}


def evaluate(
    calc,
    struct: Atoms,
    inverted: bool = False,
    model_name: str = "unknown",
    struct_name: str = "unknown",
) -> Atoms:
    """
    Evaluate energy and forces for `struct`, storing them as info/arrays.

    Parameters
    ----------
    calc
        ASE calculator to evaluate energy and forces with.
    struct
        Structure to evaluate. Modified in place.
    inverted
        Whether `struct` is the parity-transformed (inverted, r -> -r) structure,
        rather than the original.
    model_name
        Name of the model, used to identify a failed evaluation.
    struct_name
        Name of the structure, used to identify a failed evaluation.

    Returns
    -------
    Atoms
        `struct`, with "energy" and "inverted" info and "forces" array attached.
    """
    struct.calc = calc

    try:
        energy = struct.get_potential_energy()
        forces = struct.get_forces()
    except Exception as exc:
        warn(
            f"Error calculating energy/forces for model {model_name}, structure "
            f"{struct_name} (inverted={inverted}) in parity symmetry "
            f"calculation: {exc}",
            category=RuntimeWarning,
            stacklevel=2,
        )
        energy = np.nan
        forces = np.full((len(struct), 3), np.nan)

    struct.info["energy"] = energy
    struct.info["inverted"] = inverted
    struct.new_array("forces", forces)
    struct.calc = None
    return struct


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_parity_symmetry(mlip: tuple[str, Any]) -> None:
    """
    Run parity symmetry test.

    Evaluates energy and forces for 14 diverse structures, then again for
    their parity-transformed (inverted, r -> -r) versions. For periodic
    structures the cell is left unchanged: the lattice is itself inversion
    symmetric, so negating the positions inverts the whole crystal. Comparing
    inverted to original is done during analysis, not here.

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
            calc,
            original,
            inverted=False,
            model_name=model_name,
            struct_name=struct_name,
        )

        inverted = struct.copy()
        inverted.positions *= -1
        inverted.info["charge"] = 0
        inverted.info["spin"] = 1
        inverted = evaluate(
            calc,
            inverted,
            inverted=True,
            model_name=model_name,
            struct_name=struct_name,
        )

        write(write_dir / f"{struct_name}.xyz", [original, inverted])
