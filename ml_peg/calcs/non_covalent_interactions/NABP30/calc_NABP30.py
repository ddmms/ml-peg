"""
Calculate energies of 30 hydrogen-bonded nucleic acid base pairs (NABP30).

For each base pair, the pair and its two bases are calculated at the input geometry
(interaction energy) and relaxed with the model (binding energy).

J. Sponer, P. Jurecka, P. Hobza, J. Am. Chem. Soc. 2004, 126, 10142-10151.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.constraints import FixedPlane
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.optimize import BFGS
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"

# Frames written for each base pair, in this order: the pair (AB) and its two bases
# (A, B), first at the input geometry (bases as they are in the pair), then relaxed
FRAMES = [
    (geometry, fragment)
    for geometry in ("input", "relaxed")
    for fragment in ("AB", "A", "B")
]

# Geometry optimisation settings
FMAX = 0.002  # eV/Å; residual energy error well below 0.01 kcal/mol
MAX_STEPS = 3000
MAX_STEP = 0.1  # Å, largest displacement of any atom per optimiser step
# Nonplanar pairs given as planar structures start at a saddle point; a small random
# displacement lets them relax to the minimum.
RATTLE_STDEV = 0.01  # Å
RATTLE_SEED = 42
# Isolated bases start from their (often planar) geometry in a base pair, a saddle point
# for pyramidal amino groups. Each is relaxed from several randomly displaced starts and
# the lowest energy is used, as a single small displacement does not always lead off it.
BASE_RATTLE_STDEV = 0.05  # Å
BASE_STARTS = 3
# Atoms within this distance of the base-pair plane are kept in it for planar (Cs)
# pairs; atoms further out, such as methyl hydrogens, relax freely.
PLANE_TOL = 0.3  # Å
# Covalent bond cutoff, as a multiple of the sum of covalent radii. Bonds in the input
# structures are at most 1.05 times the sum, and non-bonded atoms at least 1.39 times.
BOND_MULT = 1.2


def get_monomers(dimer: Atoms) -> tuple[Atoms, Atoms]:
    """
    Split a dimer into its two monomers, at their geometries in the dimer.

    Parameters
    ----------
    dimer
        Dimer with per-atom fragment index in ``arrays["frag"]`` (0 or 1) and
        fragment charges in ``info["frag_charges"]``.

    Returns
    -------
    tuple[Atoms, Atoms]
        Monomers A (fragment 0) and B (fragment 1), with charge and spin set.
    """
    monomers = []
    for frag_id in (0, 1):
        monomer = dimer[dimer.arrays["frag"] == frag_id]
        monomer.info = {
            "charge": int(dimer.info["frag_charges"][frag_id]),
            "spin": 1,
        }
        monomers.append(monomer)
    return monomers[0], monomers[1]


def get_bonds(atoms: Atoms) -> set[tuple[int, int]]:
    """
    Get covalent bonds from interatomic distances.

    Parameters
    ----------
    atoms
        Structure to analyse.

    Returns
    -------
    set[tuple[int, int]]
        Pairs of bonded atom indices (i < j).
    """
    first, second = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=BOND_MULT))
    return {(i, j) for i, j in zip(first, second, strict=True) if i < j}


def constrain_to_plane(atoms: Atoms) -> None:
    """
    Make a base pair planar and keep it planar during relaxation.

    The structure is rotated so its best-fit plane is the xy-plane. Atoms within
    ``PLANE_TOL`` of the plane are moved into it and may then only move within it.

    Parameters
    ----------
    atoms
        Base pair to constrain, modified in place.
    """
    centred = atoms.positions - atoms.positions.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    axes[2] = np.cross(axes[0], axes[1])  # proper rotation, not a reflection
    positions = centred @ axes.T
    in_plane = np.flatnonzero(np.abs(positions[:, 2]) < PLANE_TOL)
    positions[in_plane, 2] = 0.0
    atoms.positions = positions
    atoms.set_constraint(FixedPlane(in_plane, direction=(0, 0, 1)))


def single_point(atoms: Atoms, calc: Calculator) -> Atoms:
    """
    Calculate the energy of a structure with the model, without relaxing it.

    Parameters
    ----------
    atoms
        Structure to calculate. A copy is used.
    calc
        Calculator to use.

    Returns
    -------
    Atoms
        Copy of the structure, with ``model_energy`` (eV) in ``info``.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.info["model_energy"] = atoms.get_potential_energy()
    atoms.calc = None
    return atoms


def relax(
    atoms: Atoms,
    calc: Calculator,
    planar: bool = False,
    rattle: float = 0.0,
    seed: int = RATTLE_SEED,
) -> Atoms:
    """
    Relax a structure with the model.

    Parameters
    ----------
    atoms
        Structure to relax. A copy is relaxed.
    calc
        Calculator to use.
    planar
        Whether to keep the structure planar. Default is False.
    rattle
        Standard deviation (Å) of random displacements applied before relaxing.
        Default is 0, no displacement.
    seed
        Random seed for the displacements. Default is ``RATTLE_SEED``.

    Returns
    -------
    Atoms
        Relaxed structure, with ``model_energy`` (eV), ``converged``,
        ``bonds_unchanged`` (no covalent bonds broken or formed compared with the input
        structure) and ``n_steps`` in ``info``.
    """
    atoms = atoms.copy()
    # Bonds of the input structure, before it is displaced.
    bonds = get_bonds(atoms)
    if planar:
        constrain_to_plane(atoms)
    if rattle:
        atoms.rattle(stdev=rattle, seed=seed)

    atoms.calc = calc
    optimizer = BFGS(atoms, maxstep=MAX_STEP, logfile=None)
    converged = optimizer.run(fmax=FMAX, steps=MAX_STEPS)
    atoms.info.update(
        model_energy=atoms.get_potential_energy(),
        converged=bool(converged),
        bonds_unchanged=get_bonds(atoms) == bonds,
        n_steps=optimizer.nsteps,
    )
    atoms.calc = None
    atoms.set_constraint()
    return atoms


def is_ok(atoms: Atoms) -> bool:
    """
    Check whether a relaxation converged without changing covalent bonds.

    Parameters
    ----------
    atoms
        Structure returned by ``relax``.

    Returns
    -------
    bool
        Whether the relaxation converged and kept the covalent bonds.
    """
    return atoms.info["converged"] and atoms.info["bonds_unchanged"]


def failed(atoms: Atoms) -> Atoms:
    """
    Get a copy of a structure marked as a failed relaxation.

    Parameters
    ----------
    atoms
        Structure whose relaxation failed.

    Returns
    -------
    Atoms
        Copy with NaN energy and ``converged`` and ``bonds_unchanged`` False.
    """
    atoms = atoms.copy()
    atoms.info.update(
        model_energy=np.nan, converged=False, bonds_unchanged=False, n_steps=0
    )
    return atoms


def relax_base(base: Atoms, calc: Calculator) -> Atoms:
    """
    Relax an isolated base from several starts and keep the lowest-energy result.

    Parameters
    ----------
    base
        Base at its geometry in a base pair.
    calc
        Calculator to use.

    Returns
    -------
    Atoms
        Lowest-energy relaxed structure among those that converged with unchanged
        covalent bonds (among all, if none did), with the energies (eV) of all starts
        in ``info["start_energies"]``.
    """
    relaxed = [
        relax(base, calc, rattle=BASE_RATTLE_STDEV, seed=RATTLE_SEED + start)
        for start in range(BASE_STARTS)
    ]
    best = min(
        [atoms for atoms in relaxed if is_ok(atoms)] or relaxed,
        key=lambda atoms: atoms.info["model_energy"],
    )
    best.info["start_energies"] = [atoms.info["model_energy"] for atoms in relaxed]
    return best


def heavy_atom_rmsd(atoms: Atoms, reference: Atoms) -> float:
    """
    Get the heavy-atom RMSD between two structures after optimal superposition.

    Parameters
    ----------
    atoms
        Structure to compare.
    reference
        Reference structure, with the same atoms in the same order.

    Returns
    -------
    float
        Root-mean-square deviation of the heavy atoms in Å.
    """
    heavy = reference.numbers > 1
    target, moved = reference[heavy], atoms[heavy]
    minimize_rotation_and_translation(target, moved)
    return float(
        np.sqrt(((moved.positions - target.positions) ** 2).sum(axis=1).mean())
    )


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_nabp30(mlip: tuple[str, Any]) -> None:
    """
    Run NABP30 base pair interaction and binding energy benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    data_path = (
        download_s3_data(
            filename="NABP30.zip",
            key="inputs/non_covalent_interactions/NABP30/NABP30.zip",
        )
        / "NABP30"
    )
    dimers = read(data_path / "dimers.extxyz", index=":")

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Each distinct base is relaxed once, from its geometry in the first base pair that
    # contains it, and used for every base pair that contains it
    bases = {}
    for dimer in dimers:
        for name, monomer in zip(
            dimer.info["frag_names"], get_monomers(dimer), strict=True
        ):
            bases.setdefault(name, monomer)
    relaxed_bases = {}
    for name, base in tqdm(bases.items()):
        try:
            relaxed_bases[name] = relax_base(base, calc)
        except Exception as exc:
            warn(f"{model_name}, base {name}: relaxation failed: {exc}", stacklevel=1)
            relaxed_bases[name] = failed(base)
        else:
            if not is_ok(relaxed_bases[name]):
                warn(
                    f"{model_name}, base {name}: relaxation did not converge or "
                    "changed covalent bonds",
                    stacklevel=1,
                )

    for dimer in tqdm(dimers):
        label = dimer.info["label"]
        planar = dimer.info["planar"]
        dimer.info["charge"] = int(dimer.info["charge"])
        dimer.info["spin"] = int(dimer.info["spin"])

        # Base pair and its two bases at the input geometry, not relaxed
        unrelaxed = [dimer, *get_monomers(dimer)]
        try:
            unrelaxed = [single_point(atoms, calc) for atoms in unrelaxed]
        except Exception as exc:
            warn(
                f"{model_name}, {label}: calculation at the input geometry failed: "
                f"{exc}",
                stacklevel=1,
            )
            unrelaxed = [atoms.copy() for atoms in unrelaxed]
            for atoms in unrelaxed:
                atoms.info["model_energy"] = np.nan

        try:
            pair = relax(
                dimer, calc, planar=planar, rattle=0 if planar else RATTLE_STDEV
            )
        except Exception as exc:
            warn(f"{model_name}, {label}: relaxation failed: {exc}", stacklevel=1)
            pair = failed(dimer)
        else:
            if not is_ok(pair):
                warn(
                    f"{model_name}, {label}: relaxation did not converge or changed "
                    "covalent bonds",
                    stacklevel=1,
                )
        # How far the pair moved from the input (e.g. to a different H-bond motif)
        pair.info["rmsd_to_input"] = heavy_atom_rmsd(pair, dimer)

        # Interaction and binding energies are calculated in the analysis, from the
        # three frames at the input geometry and the three relaxed frames
        frames = [
            *unrelaxed,
            pair,
            *(relaxed_bases[name].copy() for name in dimer.info["frag_names"]),
        ]
        for (geometry, fragment), frame in zip(FRAMES, frames, strict=True):
            frame.info["geometry"] = geometry
            frame.info["fragment"] = fragment
        # The reference binding energy includes the deformation energy of the bases,
        # the reference interaction energy (bases as they are in the pair) does not
        ref_binding_energy = dimer.info["ref_binding_energy_kcal"]
        frames[0].info["ref_interaction_energy"] = (
            ref_binding_energy - dimer.info["ref_E_def_kcal"]
        ) * KCAL_TO_EV
        frames[0].info["ref_binding_energy"] = ref_binding_energy * KCAL_TO_EV
        write(write_dir / f"{label}.xyz", frames)
