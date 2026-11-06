"""
Build each Landscape17 landscape from scratch and see how much is found.

The stationary points benchmark hands the model the DFT structures and asks
what survives contact with it. This asks the harder question: given a few starting
conformer and no further reference information, how much of the landscape does
the model discover on its own, compared to DFT.

Nothing here reads the reference network. The search starts from a single
geometry and runs basin hopping to collect minima, then pairs them up and looks
for the transition states between them. Comparing the result with the reference
is the analysis module's job, which is also where a model inventing structures
DFT does not have becomes visible.

Notes
-----
``NEIGHBOURS = 6`` is derived rather than tuned. Pair selection runs once, so
each minimum is paired with its six nearest and that is the whole extent of the
search. Six is the worst-case neighbour rank over the DFT reference -- the
furthest a genuine reference saddle ever sits down the neighbour list of the
minima it joins. Measured against the published setting of two: six with five
seeds recovers 56 of 63 reference saddles where two with twenty seeds recovers
44, and costs 5.8 CPU-hours against 10.0. One seed at six is worth twenty at
two, so the budget belongs in neighbours and not in seeds.

Five seeds because coverage saturates there. The transition state search is the
part that benefits from breadth, and breadth comes from pairing, not from
repeating the basin hopping.

The eigenvector following is looser than the stationary points benchmark uses
-- a hundred steps against its hundred and fifty, and a much larger uphill step
-- because the two are asking different questions. There, a search is one
measurement of one reference saddle and is worth pursuing. Here it is one
attempt among hundreds, and a failed attempt costs only that pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any
from warnings import warn

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
import pytest
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import chdir, download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

pytest.importorskip("topsearch", reason="Please install the `topsearch` extra")

from topsearch.data.coordinates import MolecularCoordinates  # noqa: E402
from topsearch.data.kinetic_transition_network import (  # noqa: E402
    KineticTransitionNetwork,
)
from topsearch.global_optimisation.basin_hopping import BasinHopping  # noqa: E402
from topsearch.global_optimisation.perturbations import (  # noqa: E402
    MolecularPerturbation,
)
from topsearch.potentials.potential import Potential  # noqa: E402
from topsearch.sampling.exploration import NetworkSampling  # noqa: E402
from topsearch.similarity.molecular_similarity import (  # noqa: E402
    MolecularSimilarity,
)
from topsearch.transition_states.hybrid_eigenvector_following import (  # noqa: E402
    HybridEigenvectorFollowing,
)
from topsearch.transition_states.nudged_elastic_band import (  # noqa: E402
    NudgedElasticBand,
)
from topsearch.utils.random import set_global_seed  # noqa: E402

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"
DATA_URI = (
    "https://github.com/VladCarare/mlp-landscapes/releases/download/landscape17-data-v1"
)
DATA_FILENAME = "Landscape17.zip"
STARTS_FILENAME = "landscape17_starts.zip"

CHARGE = 0
SPIN = 1

# Search settings. See the module docstring for why neighbours and seeds are
# what they are; the rest follow the published landscape runs.
# Each search starts from a different rMD17 frame rather than from the same
# geometry with a different random seed. Seeds that share a starting point
# differ only in where their perturbations take them, which makes them
# correlated; different conformers are independent samples of the question the
# benchmark asks, which is what a model finds when handed a structure.
#
# The archive holds twenty per molecule, evenly spaced through the trajectory
# and reordered to Landscape17's atom ordering.
STARTS = 5
NEIGHBOURS = 6
# 100, which is what the published runs used and we find is enough to find all minima.
BASIN_HOPPING_STEPS = 100
# The random seed still varies between runs, so two searches from the same
# conformer are not identical either.
SEED_OFFSETS = (3000, 3017, 3034, 3051, 3068)
BASIN_HOPPING_TEMPERATURE = 100.0
MINIMISE_CONV_CRIT = 1e-3
# topsearch's own default is 200, at which 55% of basin hopping minimisations
# on aspirin ran out of iterations and were discarded wholesale. Raising it to
# 1000 converges all of them, quadrupling the minima found for ~12% more
# L-BFGS iterations, because the capped work was previously spent and binned.
MAX_MINIMISATION_STEPS = 1000
MAX_CONNECTION_ATTEMPTS = 3
# An RMSD, so 0.3 A per atom means the same thing whatever the molecule size.
SAME_STRUCTURE = 0.3
ENERGY_CRITERION = 5e-3
# Symmetry copies are kept apart rather than merged into one node. A pair of
# them is joined by a real saddle, a degenerate rearrangement, which cannot be
# proposed while they are the same node because pair selection never pairs a
# minimum with itself. Azobenzene is the case that shows it: three of its four
# reference saddles are such rearrangements, and without this the search finds
# none of them. This departs from the published runs, which merged them.
KEEP_SYMMETRY_COPIES = True

MAX_DISPLACEMENT = 180.0
MAX_BONDS = 2
TS_CONV_CRIT = 1e-2
TS_STEPS = 100
PUSHOFF = 0.8
MAX_UPHILL_STEP_SIZE = 0.3
POSITIVE_EIGENVALUE_STEP = 0.1
STEEPEST_DESCENT_CONV_CRIT = 1e-3
EIGENVALUE_CONV_CRIT = 5e-2
NEB_FORCE_CONSTANT = 50.0
NEB_IMAGE_DENSITY = 15.0
NEB_MAX_IMAGES = 20
NEB_CONV_CRIT = 1e-2


@dataclass(frozen=True)
class _Molecule:
    """
    One Landscape17 molecule.

    Attributes
    ----------
    key
        Name used for outputs and metrics.
    directory
        Directory name inside the archive.
    xyz
        Basename of the structure the search starts from. It does not always
        match the directory name.
    """

    key: str
    directory: str
    xyz: str


MOLECULES: tuple[_Molecule, ...] = (
    _Molecule("ethanol", "ethanol", "ethanol.xyz"),
    _Molecule("malonaldehyde", "malonaldehyde", "malonaldehyde.xyz"),
    _Molecule("salicylic_acid", "salicylic_acid", "salicylic.xyz"),
    _Molecule("paracetamol", "paracetamol", "paracetamol.xyz"),
    _Molecule("azobenzene", "azobenzene", "azobenzene.xyz"),
    _Molecule("aspirin", "aspirin", "aspirin.xyz"),
)


# Cached because the shared downloader re-extracts the whole archive on
# every call rather than checking whether it is already unpacked. This is
# called once per molecule per model, which without the cache meant 36 full
# extractions, and the better part of a quarter hour, in one analysis run.
@cache
def _data_dir() -> Path:
    """
    Fetch and extract the Landscape17 archive.

    Returns
    -------
    pathlib.Path
        Directory holding the six molecule directories.
    """
    return download_github_data(filename=DATA_FILENAME, github_uri=DATA_URI) / (
        "Landscape17"
    )


@cache
def _starts_dir() -> Path:
    """
    Fetch and extract the starting conformers.

    Returns
    -------
    pathlib.Path
        Directory holding one file of frames per molecule.
    """
    return (
        download_github_data(filename=STARTS_FILENAME, github_uri=DATA_URI)
        / "landscape17_starts"
    )


class _AseCalculatorPotential(Potential):
    """
    A topsearch potential backed by any ASE calculator.

    The same adapter the stationary points benchmark uses: topsearch supplies
    the search algorithms, ml-peg supplies the model.

    Parameters
    ----------
    symbols
        Chemical symbols, fixing the element order for every position.
    calc
        ASE calculator to evaluate with.
    """

    def __init__(self, symbols: list[str], calc) -> None:
        """
        Bind the calculator to a structure of the given composition.

        Parameters
        ----------
        symbols
            Chemical symbols, fixing the element order for every position.
        calc
            ASE calculator to evaluate with.
        """
        super().__init__()
        self.atomistic = True
        self.atoms = Atoms(symbols, positions=np.zeros((len(symbols), 3)))
        self.atoms.info.update(charge=CHARGE, spin=SPIN)
        self.atoms.calc = calc

    def function(self, position: np.ndarray) -> float:
        """
        Evaluate the energy at a position.

        Parameters
        ----------
        position
            Flattened coordinates.

        Returns
        -------
        float
            Potential energy in eV.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        return float(self.atoms.get_potential_energy())

    def gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient at a position.

        Parameters
        ----------
        position
            Flattened coordinates.

        Returns
        -------
        numpy.ndarray
            Flattened gradient, the negative of the forces.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        return -np.asarray(self.atoms.get_forces(), dtype=float).flatten()

    def function_gradient(self, position: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluate energy and gradient together, which is what the optimisers use.

        Parameters
        ----------
        position
            Flattened coordinates.

        Returns
        -------
        tuple[float, numpy.ndarray]
            Energy in eV and the flattened gradient.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        forces = np.asarray(self.atoms.get_forces(), dtype=float)
        return float(self.atoms.get_potential_energy()), -forces.flatten()


def _search(symbols: list[str], start: np.ndarray, potential: Potential, seed: int):
    """
    Run one landscape search and return the network it found.

    Parameters
    ----------
    symbols
        Chemical symbols.
    start
        Flattened starting coordinates, the single conformer the search is given.
    potential
        Potential to search on.
    seed
        Seed for this run.

    Returns
    -------
    topsearch.data.kinetic_transition_network.KineticTransitionNetwork
        The network of minima and transition states found.
    """
    # topsearch dumps the network and opens a log file in the working
    # directory without being asked, so the search is run in a scratch
    # directory to keep those out of the user's. Nothing the search
    # reads is addressed by a relative path.
    with TemporaryDirectory() as scratch, chdir(Path(scratch)):
        # Building a model can reseed the global generators -- fairchem's does --
        # so the seed is set after the potential exists, not before.
        set_global_seed(seed)

        comparer = MolecularSimilarity(
            distance_criterion=SAME_STRUCTURE,
            energy_criterion=ENERGY_CRITERION,
            weighted=False,
            keep_symmetry_copies=KEEP_SYMMETRY_COPIES,
        )
        network = KineticTransitionNetwork()
        coords = MolecularCoordinates(symbols, start.copy())

        explorer = NetworkSampling(
            ktn=network,
            coords=coords,
            global_optimiser=BasinHopping(
                ktn=network,
                potential=potential,
                similarity=comparer,
                step_taking=MolecularPerturbation(
                    max_displacement=MAX_DISPLACEMENT, max_bonds=MAX_BONDS
                ),
                # topsearch minimises through this class's function_gradient,
                # the same route the stationary point benchmark takes, so a
                # minimum here is converged by the same criterion as one there.
                # Its "ase" alternative would instead drive ASE's own LBFGS to
                # an fmax, which is a different quantity from the projected
                # gradient this tolerance refers to.
                opt_method="scipy",
                max_minimisation_steps=MAX_MINIMISATION_STEPS,
            ),
            single_ended_search=HybridEigenvectorFollowing(
                potential=potential,
                ts_conv_crit=TS_CONV_CRIT,
                ts_steps=TS_STEPS,
                pushoff=PUSHOFF,
                max_uphill_step_size=MAX_UPHILL_STEP_SIZE,
                positive_eigenvalue_step=POSITIVE_EIGENVALUE_STEP,
                steepest_descent_conv_crit=STEEPEST_DESCENT_CONV_CRIT,
                eigenvalue_conv_crit=EIGENVALUE_CONV_CRIT,
            ),
            double_ended_search=NudgedElasticBand(
                potential=potential,
                force_constant=NEB_FORCE_CONSTANT,
                image_density=NEB_IMAGE_DENSITY,
                max_images=NEB_MAX_IMAGES,
                neb_conv_crit=NEB_CONV_CRIT,
            ),
            similarity=comparer,
            max_connection_attempts_per_pair=MAX_CONNECTION_ATTEMPTS,
        )

        explorer.get_minima(
            coords=coords,
            n_steps=BASIN_HOPPING_STEPS,
            conv_crit=MINIMISE_CONV_CRIT,
            temperature=BASIN_HOPPING_TEMPERATURE,
            test_valid=True,
        )
        explorer.get_transition_states(
            method="ClosestEnumeration",
            neighbours=NEIGHBOURS,
            remove_bounds_minima=False,
        )
        return network


def _frames(network, symbols: list[str], molecule: str, seed: int) -> list[Atoms]:
    """
    Convert a found network into structures to write out.

    Every minimum and every transition state the search produced, whether or
    not the reference has anything like it. What the model invents is as much
    the result as what it recovers.

    Parameters
    ----------
    network
        The network found by one search.
    symbols
        Chemical symbols.
    molecule
        Molecule name.
    seed
        Seed this came from.

    Returns
    -------
    list[ase.atoms.Atoms]
        Minima followed by transition states.
    """
    frames: list[Atoms] = []
    for index in list(network.G.nodes):
        atoms = Atoms(symbols, network.get_minimum_coords(index).reshape(-1, 3))
        atoms.info.update(
            molecule=molecule,
            seed=seed,
            kind="minimum",
            label=f"min{index:02d}",
            mlip_energy_ev=float(network.get_minimum_energy(index)),
        )
        frames.append(atoms)
    for plus, minus, edge in list(network.G.edges):
        atoms = Atoms(symbols, network.get_ts_coords(plus, minus, edge).reshape(-1, 3))
        atoms.info.update(
            molecule=molecule,
            seed=seed,
            kind="transition_state",
            label=f"ts{plus}_{minus}_{edge}",
            connects=f"{plus},{minus}",
            mlip_energy_ev=float(network.get_ts_energy(plus, minus, edge)),
        )
        frames.append(atoms)
    return frames


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_landscape_search(mlip: tuple[str, Any]) -> None:
    """
    Search each molecule's landscape from a single starting conformer.

    Marked slow deliberately. One seed of aspirin at these settings took 52
    minutes with GFN2-xTB, and this runs six molecules at five seeds each. The
    seeds are independent, so the wall time is a scheduling question rather
    than a floor.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    starts_dir = _starts_dir()

    for molecule in MOLECULES:
        stem = Path(molecule.xyz).stem
        # The starts are named by the same stem as the reference structure,
        # which is not always the molecule name: salicylic_acid is
        # salicylic.xyz, and so salicylic_starts.xyz.
        conformers = read(starts_dir / f"{stem}_starts.xyz", index=":")
        conformers = conformers[:STARTS]
        symbols = conformers[0].get_chemical_symbols()
        potential = _AseCalculatorPotential(symbols, calc)

        for index, (conformer, seed) in enumerate(
            tqdm(
                list(zip(conformers, SEED_OFFSETS, strict=True)),
                desc=f"{model_name} {molecule.key} search",
                unit="start",
                disable=not sys.stderr.isatty(),
            )
        ):
            # One file per start, not one per molecule. The searches are
            # independent and comparing them is part of reading the result, so
            # they are kept apart here and merged in the analysis. A search
            # that fails then leaves no file rather than silently fewer rows.
            destination = out_dir / f"{molecule.key}_start{index}_found.extxyz"
            if destination.exists():
                continue
            try:
                network = _search(
                    symbols, conformer.get_positions().flatten(), potential, seed
                )
            except Exception as exc:
                warn(
                    f"{model_name} {molecule.key} start {index} failed: {exc}",
                    stacklevel=2,
                )
                continue
            frames = _frames(network, symbols, molecule.key, seed)
            for atoms in frames:
                atoms.info["start"] = index
                atoms.info["start_frame"] = int(conformer.info.get("frame", -1))
            if frames:
                write(destination, frames)
