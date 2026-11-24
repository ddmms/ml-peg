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

A start that finishes writes ``<molecule>_start<i>_found.extxyz``, every
structure it found, and one that fails writes ``<molecule>_start<i>_failed.json``
saying why. A success removes the record of an earlier failure only after its
own file is in place, so a run killed in between can leave both, and the
analysis then takes the found file. A second step judges each found structure
under the same model, without moving it, in ``<molecule>_start<i>_checks.npz``.

Notes
-----
``NEIGHBOURS = 6`` is derived rather than tuned. Pair selection runs once, so
each minimum is paired with its six nearest and that is the whole extent of the
search. Six is the worst-case neighbour rank over the DFT reference -- the
furthest a genuine reference saddle ever sits down the neighbour list of the
minima it joins, reached in aspirin -- against the published setting of two.

That rank counts distinct structures, and the search does not quite give it
six. It keeps symmetry copies of a minimum apart, so that a rearrangement
between copies can be found as an ordinary edge, while topsearch ranks
neighbours by a distance under which a copy is zero away. Copies of a minimum
and of its neighbours therefore take some of the six places: in mace-omol's
searches, a minimum's six nearest hold 2.4 distinct structures on average in
aspirin and 1.3 in paracetamol. The search is the same for every model, so the
comparison is even, but saddle coverage stays below what six distinct
neighbours would reach.

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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory, mkstemp
from typing import Any
from warnings import warn

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
import numpy as np
import pytest
from tqdm.auto import tqdm

from ml_peg.calcs.landscapes.stationary_points.calc_stationary_points import (
    _model_hessian,
)
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
# Each start of each molecule is one unit, ``<molecule>_start<i>``, and leaves
# one of two files: the structures it found, or a record of why it failed. The
# check step then adds the model's verdict on each found structure.
FOUND_SUFFIX = "_found.extxyz"
FAILED_SUFFIX = "_failed.json"
CHECKS_SUFFIX = "_checks.npz"
# The file creation mask, which outputs are written under. Read once, here:
# reading it means setting it, for every thread of the process, and back.
_UMASK = os.umask(0)
os.umask(_UMASK)
DATA_URI = (
    "https://github.com/VladCarare/mlp-landscapes/releases/download/landscape17-data-v2"
)
DATA_FILENAME = "Landscape17_v2.zip"
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
# Symmetry copies -- one minimum with equivalent atoms relabelled -- are kept
# apart rather than merged into one node, so that a saddle between two copies
# can be proposed: pair selection never pairs a node with itself. Of the
# reference's 15 saddles recorded as joining a minimum to itself, 2 end in a
# relabelled copy and need this (azobenzene ts1_1_0, malonaldehyde ts1_1_1);
# the other 13 join a conformer to its mirror image, which the search keeps
# apart anyway, as its comparison does not allow inversion. This departs from
# the published runs, which merged copies.
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
        "Landscape17_v2"
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

    def __init__(self, symbols: list[str], calc: Calculator) -> None:
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


def _search(
    symbols: list[str], start: np.ndarray, potential: Potential, seed: int
) -> KineticTransitionNetwork:
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
    # topsearch writes the network it has so far (min.data, ts.data, their
    # coordinates and the pair list) to the working directory as it goes,
    # without being asked, so the search is run in a scratch directory to
    # keep those out of the user's. Nothing the search reads is addressed
    # by a relative path.
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


def _frames(
    network: KineticTransitionNetwork, symbols: list[str], molecule: str, seed: int
) -> list[Atoms]:
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


def _unit_path(out_dir: Path, molecule: str, start: int, suffix: str) -> Path:
    """
    Get the path of one of a start's output files.

    Parameters
    ----------
    out_dir
        The model's output directory.
    molecule
        Molecule key.
    start
        Start index.
    suffix
        Which file: ``FOUND_SUFFIX``, ``FAILED_SUFFIX`` or ``CHECKS_SUFFIX``.

    Returns
    -------
    pathlib.Path
        Path of that file.
    """
    return out_dir / f"{molecule}_start{start}{suffix}"


@contextmanager
def _atomic(destination: Path) -> Iterator[Path]:
    """
    Write a file so that it appears whole or not at all.

    The block writes to a new sibling file, which is flushed to disk and then
    replaces ``destination`` once the block finishes. A crash part-way leaves
    nothing at ``destination``, so a half-written file can neither be read as a
    result nor stop the start being rerun. The sibling's name is unique, so two
    runs writing the same file do not share it, and ends in ``.partial``, which
    no output glob matches.

    Parameters
    ----------
    destination
        Final path of the file.

    Yields
    ------
    pathlib.Path
        Path to write the file to.
    """
    descriptor, name = mkstemp(
        dir=destination.parent, prefix=f"{destination.name}.", suffix=".partial"
    )
    os.close(descriptor)
    partial = Path(name)
    try:
        # mkstemp makes the file private to its owner; the output gets the
        # permissions any other file written here would.
        partial.chmod(0o666 & ~_UMASK)
        yield partial
        with partial.open("rb") as written:
            os.fsync(written.fileno())
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _needs_search(out_dir: Path, molecule: str, start: int) -> bool:
    """
    Check whether a start is still to be searched, clearing an empty found file.

    A start is done once its found file exists and has something in it. An
    empty one is what a write cut short left behind before writes were atomic,
    and it is removed here: the analysis lets a found file outrank a failure
    record, so left beside the record of a rerun that failed, it would void the
    model for good.

    Parameters
    ----------
    out_dir
        The model's output directory.
    molecule
        Molecule key.
    start
        Start index.

    Returns
    -------
    bool
        Whether the start has no found file, or had only an empty one.
    """
    found = _unit_path(out_dir, molecule, start, FOUND_SUFFIX)
    if not found.exists():
        return True
    if found.stat().st_size > 0:
        return False
    found.unlink()
    return True


def _describe(exc: Exception) -> str:
    """
    Describe an exception by its type and message.

    Parameters
    ----------
    exc
        The exception.

    Returns
    -------
    str
        ``"<ExceptionType>: <message>"``.
    """
    return f"{type(exc).__name__}: {exc}"


def _record_failure(
    out_dir: Path, molecule: str, start: int, seed: int, error: str
) -> None:
    """
    Warn that a start failed, and record it in ``<molecule>_start<i>_failed.json``.

    The record is what tells a start that ran and failed apart from one that
    never ran: the analysis counts the first as a search that found nothing,
    while a start with no file at all voids the model's metrics.

    Parameters
    ----------
    out_dir
        The model's output directory, which is named after the model.
    molecule
        Molecule key.
    start
        Start index.
    seed
        Seed the start was given.
    error
        What went wrong.
    """
    warn(f"{out_dir.name} {molecule} start {start} failed: {error}", stacklevel=3)
    record = {"molecule": molecule, "start": start, "seed": seed, "error": error}
    with _atomic(_unit_path(out_dir, molecule, start, FAILED_SUFFIX)) as partial:
        partial.write_text(json.dumps(record), encoding="utf8")


def _starts(
    molecule: _Molecule, starts_dir: Path, calc: Calculator
) -> tuple[list[str], _AseCalculatorPotential, list[tuple[Atoms, int]]]:
    """
    Read a molecule's starting conformers and build the potential they share.

    Parameters
    ----------
    molecule
        Molecule to search.
    starts_dir
        Directory holding the starting conformers.
    calc
        ASE calculator to search with.

    Returns
    -------
    tuple[list[str], _AseCalculatorPotential, list[tuple[ase.atoms.Atoms, int]]]
        Chemical symbols, the potential, and each start's conformer and seed.
    """
    # The starts are named by the same stem as the reference structure,
    # which is not always the molecule name: salicylic_acid is
    # salicylic.xyz, and so salicylic_starts.xyz.
    stem = Path(molecule.xyz).stem
    conformers = read(starts_dir / f"{stem}_starts.xyz", index=":")[:STARTS]
    symbols = conformers[0].get_chemical_symbols()
    potential = _AseCalculatorPotential(symbols, calc)
    return symbols, potential, list(zip(conformers, SEED_OFFSETS, strict=True))


def _search_start(
    out_dir: Path,
    molecule: str,
    start: int,
    seed: int,
    conformer: Atoms,
    symbols: list[str],
    potential: Potential,
) -> None:
    """
    Search from one start, and write what it found or why it failed.

    Anything that raises -- the search, converting its network, or writing the
    result -- fails the start, and so does a search that finds nothing at all.
    A success removes any failure record left by an earlier run.

    Parameters
    ----------
    out_dir
        The model's output directory.
    molecule
        Molecule key.
    start
        Start index.
    seed
        Seed for this start.
    conformer
        The conformer the search starts from.
    symbols
        Chemical symbols.
    potential
        Potential to search on.
    """
    try:
        network = _search(symbols, conformer.get_positions().flatten(), potential, seed)
        frames = _frames(network, symbols, molecule, seed)
        for atoms in frames:
            atoms.info["start"] = start
            atoms.info["start_frame"] = int(conformer.info.get("frame", -1))
        if frames:
            found = _unit_path(out_dir, molecule, start, FOUND_SUFFIX)
            with _atomic(found) as partial:
                write(partial, frames, format="extxyz")
    except Exception as exc:
        _record_failure(out_dir, molecule, start, seed, _describe(exc))
        return
    if not frames:
        error = "the search returned no structures"
        _record_failure(out_dir, molecule, start, seed, error)
        return
    _unit_path(out_dir, molecule, start, FAILED_SUFFIX).unlink(missing_ok=True)


@pytest.mark.multi_day
@pytest.mark.parametrize("mlip", MODELS.items())
def test_landscape_search(mlip: tuple[str, Any]) -> None:
    """
    Search each molecule's landscape from a single starting conformer.

    Multi-day: on an NVIDIA H200, one search takes ten minutes to six and a
    half hours with the MACE models measured, and the thirty searches of one
    model add up to 46 to 59 hours run four to a GPU. The searches are
    independent, so the wall time is a scheduling question rather than a
    floor.

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
        # One file per start, not one per molecule. The searches are
        # independent and comparing them is part of reading the result, so
        # they are kept apart here and merged in the analysis. A start that
        # fails leaves a record saying so, rather than silently fewer rows.
        # A start whose structures are already written is done; a failed one,
        # or one that left an empty file, is tried again.
        try:
            symbols, potential, starts = _starts(molecule, starts_dir, calc)
        except Exception as exc:
            for start, seed in enumerate(SEED_OFFSETS):
                if _needs_search(out_dir, molecule.key, start):
                    _record_failure(out_dir, molecule.key, start, seed, _describe(exc))
            continue

        for start, (conformer, seed) in enumerate(
            tqdm(
                starts,
                desc=f"{model_name} {molecule.key} search",
                unit="start",
                disable=not sys.stderr.isatty(),
            )
        ):
            if not _needs_search(out_dir, molecule.key, start):
                continue
            _search_start(
                out_dir, molecule.key, start, seed, conformer, symbols, potential
            )


def _check(calc: Calculator, atoms: Atoms) -> tuple[float, np.ndarray]:
    """
    Evaluate one found structure under the model, where it is.

    Parameters
    ----------
    calc
        ASE calculator to evaluate with.
    atoms
        The structure.

    Returns
    -------
    tuple[float, numpy.ndarray]
        The largest force on any atom, in eV/A, and the model's Cartesian
        Hessian in eV/A^2.
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    gradient = _AseCalculatorPotential(symbols, calc).gradient(positions.flatten())
    max_force = float(np.linalg.norm(gradient.reshape(-1, 3), axis=1).max())
    return max_force, _model_hessian(calc, symbols, positions)


def _checks(
    calc: Calculator,
    frames: list[Atoms],
    labels: list[str],
    where: str,
    stored: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """
    Evaluate every structure of one start, each on its own.

    A structure the model fails on is flagged, with a NaN force and an all-NaN
    Hessian; the others are unaffected. A structure that earlier checks of the
    same found file evaluated without failing keeps that result, so a model
    that always fails on one structure costs that structure on each run, not
    the whole file.

    Parameters
    ----------
    calc
        ASE calculator to evaluate with.
    frames
        The structures, as the found file holds them.
    labels
        Each structure's label.
    where
        Model and file, for warnings.
    stored
        Earlier checks of this same found file, or None.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``max_force`` and ``failed``, one entry per structure in file order,
        and ``hessian_<label>`` for every structure.
    """
    max_force = np.full(len(frames), np.nan)
    failed = np.zeros(len(frames), dtype=bool)
    hessians: dict[str, np.ndarray] = {}
    for index, (atoms, label) in enumerate(zip(frames, labels, strict=True)):
        if stored is not None and not stored["failed"][index]:
            max_force[index] = stored["max_force"][index]
            hessians[f"hessian_{label}"] = stored[f"hessian_{label}"]
            continue
        try:
            max_force[index], hessian = _check(calc, atoms)
        except Exception as exc:
            warn(f"{where}: check failed on {label}: {_describe(exc)}", stacklevel=2)
            failed[index] = True
            hessian = np.full((3 * len(atoms),) * 2, np.nan)
        hessians[f"hessian_{label}"] = hessian
    return {"max_force": max_force, "failed": failed, **hessians}


def _stored_checks(checks: Path, digest: str) -> dict[str, np.ndarray] | None:
    """
    Read a start's earlier checks, if they were made of this found file.

    Parameters
    ----------
    checks
        The start's checks file.
    digest
        SHA-256 of its found file as it is now.

    Returns
    -------
    dict[str, numpy.ndarray] | None
        Every array the checks hold; None if there are none, they cannot be
        read, or they were made of a different found file or before checks
        carried a fingerprint.
    """
    try:
        with np.load(checks) as data:
            stored = {key: data[key] for key in data.files}
        same = str(stored["found_sha256"]) == digest
    except Exception:
        return None
    return stored if same else None


@pytest.mark.multi_day
@pytest.mark.parametrize("mlip", MODELS.items())
def test_check_found_structures(mlip: tuple[str, Any]) -> None:
    """
    Evaluate every structure the search found, under the model that found it.

    The search keeps what its own convergence tests accept, which is looser
    than what the analysis counts as a minimum or a saddle. For each structure,
    in found-file order, this records the largest force on any atom and the
    model's Cartesian Hessian, computed as the stationary points benchmark
    computes it, for the analysis to judge the structure by. Structures are
    evaluated where they are and never moved.

    The results go to ``<molecule>_start<i>_checks.npz`` beside each found
    file, with the SHA-256 of the found file they were made from. A start is
    redone when that no longer matches; when it matches, only the structures
    that failed are tried again.

    Marked multi-day with the search it judges, since the analysis voids a
    model without its checks, so that one flag runs the benchmark end to end.
    On its own it takes roughly one to five hours per model: each structure
    costs a finite-difference Hessian of 6N force calls, the 3 Nov searches
    found 690 to 966 structures per model, and the stationary points
    benchmark's Hessians took 5.5 to 18 s each on an H200, MACE to UMA.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    out_dir = OUT_PATH / model_name

    for found in tqdm(
        sorted(out_dir.glob(f"*{FOUND_SUFFIX}")),
        desc=f"{model_name} checks",
        unit="start",
        disable=not sys.stderr.isatty(),
    ):
        checks = found.with_name(found.name.removesuffix(FOUND_SUFFIX) + CHECKS_SUFFIX)
        where = f"{model_name} {found.name}"
        # Hashed and parsed from the same bytes, so the fingerprint is of
        # exactly what was checked.
        content = found.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        # A failure may be transient -- a GPU out of memory, say -- so checks
        # with one are not final: the structures that failed are tried again.
        stored = _stored_checks(checks, digest)
        if stored is not None and not stored["failed"].any():
            continue
        try:
            frames = read(StringIO(content.decode()), index=":", format="extxyz")
            if not frames:
                raise ValueError("it holds no structures")
            labels = [atoms.info["label"] for atoms in frames]
            kinds = [atoms.info["kind"] for atoms in frames]
        except Exception as exc:
            warn(f"{where}: unreadable, not checked: {_describe(exc)}", stacklevel=2)
            continue
        arrays = _checks(calc, frames, labels, where, stored)
        with _atomic(checks) as partial, partial.open("wb") as handle:
            np.savez(
                handle,
                labels=np.array(labels, dtype=str),
                kinds=np.array(kinds, dtype=str),
                found_sha256=np.array(digest),
                **arrays,
            )
