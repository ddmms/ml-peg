"""Run calculations for locality benchmark."""

from __future__ import annotations

from copy import copy
from pathlib import Path

from ase import Atoms
from ase.io import write
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

SEED = 42


def smiles_to_ase(smiles: str) -> Atoms:
    """
    Prepare solute structure.

    Parameters
    ----------
    smiles
        SMILES string of solute to create.

    Returns
    -------
    Atoms
        Solute structure.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=SEED)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    return Atoms(symbols=symbols, positions=positions, info={"charge": 0, "spin": 1})


def prepare_ghost_system(solute: Atoms, num_ne: int, ghost_dist: float) -> Atoms:
    """
    Create combined system of solute and ghost Ne atoms.

    Parameters
    ----------
    solute
        Solute structure to add Ne atoms to.
    num_ne
        Number of Ne atoms to add to solute.
    ghost_dist
        Minimum distance to place ghost atoms.

    Returns
    -------
    Atoms
        Combined solute and ghost Ne atoms system.
    """
    # Place ghost Ne atoms at least ghost_dist from solute center of mass
    com = solute.get_center_of_mass()
    ghost_positions = []
    rng = np.random.default_rng(SEED)

    # Find positions for Ne atoms
    while len(ghost_positions) < num_ne:
        pos = rng.uniform(low=0, high=60, size=3)
        if np.linalg.norm(pos - com) >= ghost_dist:
            ghost_positions.append(pos)

    ghost_atoms = Atoms(
        symbols=["Ne"] * num_ne,
        positions=ghost_positions,
        info={"charge": 0, "spin": 1},
    )
    return solute.copy() + ghost_atoms


def add_random_h(
    solute: Atoms, min_dist: float, max_dist: float, rng: np.random.Generator
) -> Atoms:
    """
    Add random hydrogens around solute.

    Parameters
    ----------
    solute
        Solute structure to add random hydrogens to.
    min_dist
        Minimum distance of sphere shell to add hydrogens to.
    max_dist
        Maximum distance of sphere shell to add hydrogens to.
    rng
        Random number generator, used to place hydrogen atoms.

    Returns
    -------
    Atoms
        Combined solute and randomly placed hydrogens.
    """
    com = solute.get_center_of_mass()
    for _ in range(100):
        # Random point on sphere shell between min_dist and max_dist
        r = rng.uniform(min_dist, max_dist)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        x = com[0] + r * np.sin(phi) * np.cos(theta)
        y = com[1] + r * np.sin(phi) * np.sin(theta)
        z = com[2] + r * np.cos(phi)
        pos = np.array([x, y, z])
        # check min distance to existing atoms
        dists = np.linalg.norm(solute.positions - pos, axis=1)
        if np.all(dists > 1.0):
            return solute.copy() + Atoms(
                "H", positions=[pos], info={"charge": 0, "spin": 1}
            )

    # Fallback: add H at min_dist along x axis
    pos = com + np.array([min_dist, 0, 0])
    return solute.copy() + Atoms("H", positions=[pos], info={"charge": 0, "spin": 1})


@pytest.fixture(scope="module")
def prepared_solute() -> dict[str, Atoms]:
    """
    Prepare solute ASE structure from SMILES.

    Returns
    -------
    dict[str, Atoms]
        Solute with forces calcualted, for each model.
    """
    solute_smiles = "CC(=O)C"  # SMILES (RDKit required)
    box_length = 60.0  # Edge length of cubic cell

    # Generate solute structure
    solute = smiles_to_ase(solute_smiles)
    solute.set_cell([box_length, box_length, box_length])
    solute.center()
    solute.pbc = True

    solutes = {}
    for model_name, calc in MODELS.items():
        solute = solute.copy()
        try:
            calc.default_dtype = "float64"
            solute.calc = calc.get_calculator()
            solute.get_forces()
            solutes[model_name] = solute
        # If a model fails, don't block other model tests
        except (ModuleNotFoundError, RuntimeError, TypeError):
            continue
    return solutes


@pytest.mark.parametrize("model_name", MODELS)
def test_ghost_atoms(prepared_solute: dict[str, Atoms], model_name: str) -> None:
    """
    Run ghost atom tests.

    Parameters
    ----------
    prepared_solute
        Solute structure to add ghost atoms to for each model.
    model_name
        Name of model use and model to get calculator.
    """
    solute = prepared_solute[model_name]

    ghost_num = 20  # number of ghost atoms
    ghost_dist = 40.0  # place all Ne ≥ this many Angstrom from solute COM

    system_ghost = prepare_ghost_system(solute, ghost_num, ghost_dist)
    system_ghost.calc = copy(solute.calc)

    system_ghost.get_forces()

    # Write output structures
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / "system_ghost.xyz", [solute, system_ghost])


@pytest.mark.parametrize("model_name", MODELS)
def test_rand_h(prepared_solute: dict[str, Atoms], model_name: str) -> None:
    """
    Run test adding random hydrogens to solute.

    Parameters
    ----------
    prepared_solute
        Solute structure to add hydrogens to for each model.
    model_name
        Name of model use and model to get calculator.
    """
    solute = prepared_solute[model_name]

    rand_trials = 30
    rand_min_dist = 20.0  # inner shell radius for random H
    rand_max_dist = 50.0  # outer shell radius
    rng = np.random.default_rng(SEED)

    rand_h_structures = [solute]

    for _ in range(rand_trials):
        system_rand_h = add_random_h(solute, rand_min_dist, rand_max_dist, rng)
        system_rand_h.calc = copy(solute.calc)
        system_rand_h.get_forces()
        rand_h_structures.append(system_rand_h)

    # Write output structures
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / "system_random_H.xyz", rand_h_structures)
