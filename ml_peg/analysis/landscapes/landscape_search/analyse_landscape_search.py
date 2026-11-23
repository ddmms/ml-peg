"""
Analyse the Landscape17 landscape search results and emit metrics.

Reads the structures ``calc_landscape_search.py`` found and asks how much of
the DFT landscape they account for. Nothing in the calculation looks at the
reference; the comparison happens here, which keeps the search honest and
means a metric can be redefined without repeating it.

Notes
-----
A structure counts only if the model holds it as what the search says it is:
a minimum with no force left and no downhill direction, or a saddle with one
dominant downhill direction. The calculation's check step records both, since
topsearch accepts the minima its saddle searches descend into without checking
either, and ends its saddle searches on the force alone.

The five searches find most structures more than once, so copies are merged
first, so that each structure counts once. Two found structures are copies when
they are closer than half the gap between the two closest distinct reference
structures of their kind (``MINIMA_GROUP_RADIUS``, ``SADDLE_GROUP_RADIUS``); a
merged structure is compared with the reference through its closest copy.

A found structure is a reference structure when it is within ``MATCH_RADIUS``
of it, and the pairing is one to one: each reference structure is matched to at
most one found structure and each found structure to at most one reference.
Recovery counts the reference structures that have a match, so two found minima
near the same DFT minimum recover it once. The pairing makes as many matches as
it can, then picks the closest. Matching each reference to its nearest in turn
would go wrong where two reference structures sit close together -- paracetamol's
minima 2 and 3 are 0.34 Angstrom apart -- since the first could take a found
structure the second needed.

Coverage and accuracy are reported separately throughout. A model that finds
two of twenty eight minima and places both perfectly is not a good model, and
one number that mixes the two would say it was.

A model whose search outputs are incomplete -- a search missing, one the
protocol does not run, a file that cannot be read, or found structures the
check step has not judged -- scores NaN on every metric, rather than being
scored on the searches that happen to exist. A search that raised counts as
one that found nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from warnings import warn

from ase import neighborlist
from ase.atoms import Atoms
from ase.io import read, write
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from ml_peg.analysis.landscapes.stationary_points.hessian_metrics import (
    internal_basis,
)
from ml_peg.analysis.utils.decorators import build_table, plot_violin
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.landscapes.landscape_search.calc_landscape_search import (
    CHECKS_SUFFIX,
    FAILED_SUFFIX,
    FOUND_SUFFIX,
    MOLECULES,
    STARTS,
    _data_dir,
    _Molecule,
    _unit_path,
)
from ml_peg.calcs.landscapes.stationary_points.calc_stationary_points import (
    SETTLED_FORCE,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

pytest.importorskip("topsearch", reason="Please install the `topsearch` extra")

from topsearch.data.coordinates import MolecularCoordinates  # noqa: E402
from topsearch.data.kinetic_transition_network import (  # noqa: E402
    KineticTransitionNetwork,
)
from topsearch.similarity.molecular_similarity import (  # noqa: E402
    MolecularSimilarity,
)

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map()

CALC_PATH = CALCS_ROOT / "landscapes" / "landscape_search" / "outputs"
OUT_PATH = APP_ROOT / "data" / "landscapes" / "landscape_search"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# The calculation writes one set of files per search, named for its molecule
# and starting conformer by its ``_unit_path``: what it found, or a record that
# it raised, and the model's checks of what it found. They are pooled here
# rather than there, so that what each search found on its own stays on disk
# and can be compared.
# Below this RMSD a found structure is the reference structure it is compared
# with: for matching, for extra structures (further than this from every
# reference), for near misses, and for a saddle's endpoints. It compares a found
# structure with a reference one; the group radii below compare two found
# structures with each other. The same value the search uses to decide whether
# a newly found minimum is new.
MATCH_RADIUS = 0.3
# Found structures all within this RMSD of each other are copies of one. Just
# under half the closest separation of two distinct reference structures of
# the same kind, so that a model's counterparts of two references are never
# taken for one structure and left to match only one of them: paracetamol's
# minima 2 and 3 are 0.340 A apart, aspirin's saddles 30 and 34 0.186 A.
# Grouping at MATCH_RADIUS would merge both pairs.
MINIMA_GROUP_RADIUS = 0.15
SADDLE_GROUP_RADIUS = 0.09
# Outer edge of a near miss, as a multiple of MATCH_RADIUS. A reference
# whose nearest neighbour falls between the two is one the search reached
# but not closely enough to count, which a pass rate alone cannot show.
NEAR_MISS_FACTOR = 1.5
# Cost of a pairing further apart than MATCH_RADIUS. Larger than any total of
# pairings within it, so the assignment makes as many of those as it can before
# it minimises their RMSD, and finite, which the solver requires.
UNMATCHABLE = 1.0e6
# What a reference with nothing found near it adds to the median nearest
# distance, in A. Infinite would be the truth, but the app reads an infinite
# cell as missing and voids the model's whole score; any distance past the
# scoring threshold scores the same, and this one is past any distance between
# two conformers of these molecules, so it reads as what it is.
UNFOUND_DISTANCE = 10.0
# A found saddle is first-order when its lowest curvature, translation and
# rotation projected out, is below -REACTION_MODE eV/A^2 -- the floor topsearch's
# own check_valid_ts puts on a reaction mode -- and any second negative
# curvature is at most SECOND_MODE_RATIO of it; a found minimum has no curvature
# below -REACTION_MODE at all. The rule Landscape17 was built
# with: two negative modes are one saddle when one dominates. The DFT reference
# itself has a second mode of -0.0012 against -0.459 at paracetamol ts0_1_0, a
# ratio of 0.003; the closest call among the stationary points benchmark's
# saddles is 0.03, and the first it rejects is 0.22.
REACTION_MODE = 1e-3
SECOND_MODE_RATIO = 0.1
# What a found structure is, best first; a group of copies is the best of its
# members. GOOD is the molecule, at a stationary point of the kind it is
# labelled; NOT_STATIONARY fails the model's check for that kind.
GOOD = "good"
DISTORTED = "distorted"
OFF_REFERENCE = "off_reference"
NOT_STATIONARY = "not_stationary"
STATUS_ORDER = (GOOD, DISTORTED, OFF_REFERENCE, NOT_STATIONARY)
# ASE's natural cutoffs scaled. At 1.0 the covalent radii are tight enough that
# every molecule containing hydrogen reads as fragmented; 1.1 to 1.2 is the
# usable range, and the neighbour list adds the skin to EACH radius, so it is
# left at zero here rather than being used to widen the bonds.
BOND_CUTOFF_SCALE = 1.2
# How far a bond may stray from its length in the reference minima before the
# minimum holding it is called distorted, as a fraction of that length. The
# bonding graph cannot see this at all: MACE-OFF-23-L converges paracetamol
# minima with C7-O8 crushed to 0.86 A against DFT's 1.36, and they come back
# with the reference's twenty bonds, none added and none lost, because a bond
# at two thirds of its length is still a bond.
#
# Calibrated against those structures rather than guessed. The model's genuine
# minima hold that bond within half a percent of DFT's, and no DFT minimum is
# more than 1.7% from the mean over the molecule's minima on any bond, while
# the crushed ones sit at 37%. A fifth is far from both populations, which is
# what makes it a threshold rather than a midpoint.
DISTORTED_BOND_TOLERANCE = 0.2
# Where the flagged structures go, under the model's directory. The fraction
# says a model converged impossible geometry; only the structures say what it
# did.
DISTORTED_SUBDIR = "distorted_minima"
# topsearch's alignment tries random starting rotations and keeps the best, so
# the distance it reports depends on which it drew. Unseeded, the same pair
# scores differently from one run to the next -- in the last digits for most
# pairs, and on either side of the matching radius for one near it -- and no
# published number can be reproduced.
ALIGNMENT_SEED = 0


@contextmanager
def _seeded() -> Iterator[None]:
    """
    Seed NumPy's global generator for the duration, then put it back.

    topsearch draws its starting rotations from the global generator, so that
    is the one to seed. Restoring it afterwards keeps scoring from disturbing
    anything else that draws from it.

    Yields
    ------
    None
        Control, with the generator seeded with ``ALIGNMENT_SEED``.
    """
    state = np.random.get_state()
    np.random.seed(ALIGNMENT_SEED)
    try:
        yield
    finally:
        np.random.set_state(state)


def _comparer(early_exit: float = 0.0) -> MolecularSimilarity:
    """
    Build the structure comparison used for every distance here.

    ``distance_criterion`` is an early-exit threshold, so any non-zero value
    returns a distance that is merely below it rather than the smallest one.
    Zero, the default, for a distance that gets reported; a yes-or-no test may
    pass its own radius, which only lets it stop sooner.

    Parameters
    ----------
    early_exit
        Distance below which the alignment may stop searching, in Å.

    Returns
    -------
    topsearch.similarity.molecular_similarity.MolecularSimilarity
        Permutation- and inversion-aware comparison.
    """
    return MolecularSimilarity(
        distance_criterion=early_exit,
        energy_criterion=1e8,
        weighted=False,
        allow_inversion=True,
    )


def _rmsd(
    comparer: MolecularSimilarity,
    symbols: list[str],
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Get the RMSD between two structures at their best alignment.

    Seeded per pair, so that a pair's distance does not depend on how many
    others were aligned before it.

    Parameters
    ----------
    comparer
        Comparison to use, from ``_comparer``.
    symbols
        Chemical symbols.
    first
        Flattened positions.
    second
        Flattened positions.

    Returns
    -------
    float
        RMSD in Å, over rotations, translations, inversion and permutations of
        equivalent atoms.
    """
    coords = MolecularCoordinates(symbols, np.asarray(first, dtype=float).copy())
    with _seeded():
        return float(
            comparer.closest_distance(coords, np.asarray(second, dtype=float).copy())
        )


def _bonds(atoms: Atoms) -> frozenset[tuple[int, int]]:
    """
    Get which atoms are bonded, as an unordered set of pairs.

    Parameters
    ----------
    atoms
        Structure to analyse.

    Returns
    -------
    frozenset[tuple[int, int]]
        Bonded index pairs, each ordered low to high.
    """
    cutoffs = neighborlist.natural_cutoffs(atoms, mult=BOND_CUTOFF_SCALE)
    neighbours = neighborlist.NeighborList(
        cutoffs, skin=0.0, self_interaction=False, bothways=True
    )
    neighbours.update(atoms)
    matrix = neighbours.get_connectivity_matrix(sparse=False)
    return frozenset(
        (i, j)
        for i in range(len(atoms))
        for j in range(i + 1, len(atoms))
        if matrix[i, j]
    )


def _labelled_graph(bonds: frozenset[tuple[int, int]], symbols: list[str]) -> nx.Graph:
    """
    Build a bonding graph whose nodes carry their element.

    Parameters
    ----------
    bonds
        Bonded index pairs, as ``_bonds`` returns them.
    symbols
        Chemical symbol of each atom.

    Returns
    -------
    networkx.Graph
        One node per atom, labelled ``element``, and one edge per bond.
    """
    graph = nx.Graph()
    # Every atom is a node, bonded or not: an atom that has come away entirely
    # leaves no edge, and would otherwise drop out of the comparison.
    graph.add_nodes_from(
        (index, {"element": symbol}) for index, symbol in enumerate(symbols)
    )
    graph.add_edges_from(bonds)
    return graph


def _in_reference_labelling(
    atoms: Atoms, reference_bonds: frozenset[tuple[int, int]]
) -> Atoms | None:
    """
    Put a structure's atoms in the reference's order, if it is the same molecule.

    Matching is permutation-invariant, so a structure can be the reference
    molecule with equivalent atoms labelled differently -- two hydrogens on
    different carbons exchanged -- and still match it at 0 A. Index by index it
    then looks rebonded, and every bond length involving those atoms looks
    wrong, although nothing about the molecule has changed. So bonding is
    compared as graphs with elements matched, and the isomorphism found is used
    to relabel the structure before anything is measured.

    Parameters
    ----------
    atoms
        Structure to relabel.
    reference_bonds
        Bonding of the reference molecule, as ``_bonds`` returns it.

    Returns
    -------
    ase.atoms.Atoms | None
        The structure with its atoms reordered to the reference's, or None if
        its bonds differ from the molecule's however equivalent atoms are
        relabelled.
    """
    bonds = _bonds(atoms)
    if bonds == reference_bonds:
        return atoms
    symbols = atoms.get_chemical_symbols()
    matcher = isomorphism.GraphMatcher(
        _labelled_graph(bonds, symbols),
        _labelled_graph(reference_bonds, symbols),
        node_match=isomorphism.categorical_node_match("element", None),
    )
    if not matcher.is_isomorphic():
        return None
    found_for = {reference: found for found, reference in matcher.mapping.items()}
    return atoms[[found_for[index] for index in range(len(atoms))]]


def _bond_deviations(
    atoms: Atoms, reference_lengths: dict[tuple[int, int], float]
) -> dict[tuple[int, int], float]:
    """
    Measure how far each of the molecule's bonds has moved from its reference length.

    The pairs come from the reference, never from ``atoms``. A structure that
    has rebonded is therefore measured only on the bonds the molecule is
    supposed to have, and whatever it has grown instead is the off-reference
    check's business rather than this one's.

    Index by index, so ``atoms`` must already be in the reference's atom
    order -- see ``_in_reference_labelling`` -- for pair ``(i, j)`` to mean the
    same two atoms in each.

    A deviation that is not finite -- a search that returned NaN coordinates,
    or a reference bond of zero length -- comes back as infinity rather than as
    NaN. NaN compares false against every threshold, so a structure nothing can
    be said about would otherwise pass as undistorted.

    Parameters
    ----------
    atoms
        Structure to measure.
    reference_lengths
        Reference bond lengths, as ``_reference_lengths`` returns them.

    Returns
    -------
    dict[tuple[int, int], float]
        Absolute deviation of each bond as a fraction of its reference length.
    """
    pairs = list(reference_lengths)
    found = np.array([atoms.get_distance(i, j) for i, j in pairs], dtype=float)
    expected = np.array([reference_lengths[pair] for pair in pairs], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviations = np.abs(found - expected) / expected
    return {
        pair: float(deviation) if np.isfinite(deviation) else float(np.inf)
        for pair, deviation in zip(pairs, deviations, strict=True)
    }


def _is_distorted(
    atoms: Atoms, reference_lengths: dict[tuple[int, int], float]
) -> bool:
    """
    Say whether a minimum holds one of the molecule's bonds at an impossible length.

    The bonding comparison cannot answer this. It asks whether a pair is
    bonded, and a crushed bond still is, so a minimum can be converged, carry
    the reference's bonding graph exactly, and still be a geometry no chemistry
    produces. This asks how long the bond became.

    Parameters
    ----------
    atoms
        Minimum to test.
    reference_lengths
        Reference bond lengths, as ``_reference_lengths`` returns them.

    Returns
    -------
    bool
        True if any of the molecule's bonds is further from its reference
        length than ``DISTORTED_BOND_TOLERANCE``, as a fraction of it.
    """
    deviations = _bond_deviations(atoms, reference_lengths)
    return any(
        deviation > DISTORTED_BOND_TOLERANCE for deviation in deviations.values()
    )


def _classify(
    atoms: Atoms,
    reference_bonds: frozenset[tuple[int, int]],
    reference_lengths: dict[tuple[int, int], float],
) -> str:
    """
    Say whether a structure is the molecule, and if so in a shape it can have.

    A structure is off-reference when its bonds differ from the molecule's,
    however equivalent atoms (a methyl's hydrogens, say) are relabelled. Its
    bond lengths are then not measured: the bond a migrated hydrogen left would
    read as stretched, and count the same structure twice. A structure with the
    molecule's bonds but one of them at an impossible length is distorted.

    Parameters
    ----------
    atoms
        Structure to classify.
    reference_bonds
        Bonding of the reference molecule, as ``_bonds`` returns it.
    reference_lengths
        Reference bond lengths, as ``_reference_lengths`` returns them.

    Returns
    -------
    str
        ``GOOD``, ``DISTORTED`` or ``OFF_REFERENCE``.
    """
    relabelled = _in_reference_labelling(atoms, reference_bonds)
    if relabelled is None:
        return OFF_REFERENCE
    if _is_distorted(relabelled, reference_lengths):
        return DISTORTED
    return GOOD


def _curvatures(hessian: np.ndarray, atoms: Atoms) -> np.ndarray | None:
    """
    Get a structure's curvatures with translation and rotation projected out.

    Parameters
    ----------
    hessian
        The model's Hessian at the structure, in eV/Å², as the check step
        saved it.
    atoms
        The structure.

    Returns
    -------
    numpy.ndarray | None
        Internal curvatures in eV/Å², ascending; None if the Hessian is not
        finite, as when the model could not evaluate the structure.
    """
    hessian = np.asarray(hessian, dtype=float)
    if not np.all(np.isfinite(hessian)):
        return None
    basis = internal_basis(atoms)
    return np.linalg.eigvalsh(basis.T @ (0.5 * (hessian + hessian.T)) @ basis)


def _first_order(hessian: np.ndarray, atoms: Atoms) -> bool:
    """
    Say whether the model's curvature makes a found saddle a first-order one.

    Parameters
    ----------
    hessian
        The model's Hessian at the saddle, in eV/Å².
    atoms
        The saddle.

    Returns
    -------
    bool
        True if the lowest internal curvature is below ``-REACTION_MODE`` and
        any second negative one is at most ``SECOND_MODE_RATIO`` of it.
    """
    curvatures = _curvatures(hessian, atoms)
    if curvatures is None:
        return False
    reaction = curvatures[0]
    return bool(
        reaction < -REACTION_MODE and curvatures[1] >= SECOND_MODE_RATIO * reaction
    )


def _held_minimum(max_force: float, hessian: np.ndarray, atoms: Atoms) -> bool:
    """
    Say whether the model holds a found minimum as a minimum.

    topsearch's own check is not used: it keeps translation and rotation in
    and ignores the six lowest eigenvalues, so away from an exact stationary
    point a real downhill direction can hide among them, and it passes a
    second-order saddle with forces below the tolerance.

    Parameters
    ----------
    max_force
        Largest force left on any atom, in eV/Å.
    hessian
        The model's Hessian at the structure, in eV/Å².
    atoms
        The structure.

    Returns
    -------
    bool
        True if no atom has more than ``SETTLED_FORCE`` left and no internal
        curvature is below ``-REACTION_MODE``.
    """
    curvatures = _curvatures(hessian, atoms)
    return bool(
        max_force <= SETTLED_FORCE
        and curvatures is not None
        and curvatures[0] >= -REACTION_MODE
    )


def _distances(
    reference: list[np.ndarray], found: list[np.ndarray], symbols: list[str]
) -> np.ndarray:
    """
    Build the RMSD between every reference structure and every found one.

    Parameters
    ----------
    reference
        Reference positions, flattened.
    found
        Found positions, flattened.
    symbols
        Chemical symbols.

    Returns
    -------
    numpy.ndarray
        Distances, reference by found.
    """
    comparer = _comparer()
    matrix = np.zeros((len(reference), len(found)))
    for i, ref in enumerate(reference):
        for j, other in enumerate(found):
            matrix[i, j] = _rmsd(comparer, symbols, ref, other)
    return matrix


def _group(
    found: list[np.ndarray], symbols: list[str], radius: float
) -> list[list[int]]:
    """
    Group copies of one structure found by different searches.

    Each structure joins the first group all of whose members are within
    ``radius`` of it, and otherwise starts its own. All of them rather than the
    first: a group may then not stretch to twice the radius, as one anchored on
    its first member can.

    Parameters
    ----------
    found
        Found positions, flattened, in the order the searches are read.
    symbols
        Chemical symbols.
    radius
        Distance below which two structures are copies, in Å.

    Returns
    -------
    list[list[int]]
        Indices into ``found``, one list per distinct structure.
    """
    comparer = _comparer(radius)
    groups: list[list[int]] = []
    for index, position in enumerate(found):
        for members in groups:
            if all(
                _rmsd(comparer, symbols, found[member], position) < radius
                for member in members
            ):
                members.append(index)
                break
        else:
            groups.append([index])
    return groups


def _group_copies(
    found: list[_Found], statuses: list[str], symbols: list[str], radius: float
) -> list[list[int]]:
    """
    Group copies, the structures the model holds apart from the rest.

    A structure the model does not hold, or that is not the molecule, never
    joins a group of ones that are: a group is as close to a reference as its
    closest copy, and one anchored on a structure between two references could
    gather both references' counterparts and match only one of them.

    Parameters
    ----------
    found
        Found copies of one kind.
    statuses
        Status of each, as ``_status`` returns them.
    symbols
        Chemical symbols.
    radius
        Distance below which two structures are copies, in Å.

    Returns
    -------
    list[list[int]]
        Indices into ``found``, one list per distinct structure, ``GOOD`` ones
        first.
    """
    groups = []
    for good in (True, False):
        indices = [i for i, status in enumerate(statuses) if (status == GOOD) == good]
        positions = [found[i].atoms.get_positions().flatten() for i in indices]
        groups += [
            [indices[i] for i in group] for group in _group(positions, symbols, radius)
        ]
    return groups


def _assign(matrix: np.ndarray) -> dict[int, tuple[int, float]]:
    """
    Pair reference structures with found ones, one to one.

    As many pairs within ``MATCH_RADIUS`` as possible, then the least total
    RMSD among those. Minimising total RMSD first and dropping the distant
    pairs afterwards can match fewer: given 0.28 and 0.05 against one
    reference and 0.32 and 0.295 against another, it takes 0.05 and 0.32 and
    keeps one match where two were available.

    Parameters
    ----------
    matrix
        Distances, reference by found; infinite for a structure that may not
        be matched.

    Returns
    -------
    dict[int, tuple[int, float]]
        Reference index to the found index it matched and their distance.
    """
    if matrix.size == 0:
        return {}
    costs = np.where(matrix < MATCH_RADIUS, matrix, UNMATCHABLE)
    rows, columns = linear_sum_assignment(costs)
    return {
        int(r): (int(c), float(matrix[r, c]))
        for r, c in zip(rows, columns, strict=True)
        if matrix[r, c] < MATCH_RADIUS
    }


def _nearest(matrix: np.ndarray) -> list[float]:
    """
    Distance from each reference structure to the closest one found.

    Reported alongside the match counts because those are a pass rate over a
    hard radius: a reference missed by 0.31 Å and one missed by 3 Å both
    score zero, which reads as total failure rather than as the systematic
    offset the first one is. Unconstrained, unlike ``_assign`` -- the question
    is whether the search reached this reference at all, and the one-to-one
    constraint would answer a different one when two references share a
    nearest neighbour.

    Parameters
    ----------
    matrix
        Distances, reference by found.

    Returns
    -------
    list[float]
        One distance per reference, infinite where nothing was found.
    """
    if matrix.size == 0:
        return [float("inf")] * matrix.shape[0]
    return [float(d) for d in matrix.min(axis=1)]


def _near_missed(nearest: list[float]) -> int:
    """
    Count references the search reached but not closely enough to match.

    Parameters
    ----------
    nearest
        Distance from each reference to the closest structure found.

    Returns
    -------
    int
        How many fall between the matching radius and ``NEAR_MISS_FACTOR``
        times it.
    """
    return sum(
        1 for d in nearest if MATCH_RADIUS <= d < NEAR_MISS_FACTOR * MATCH_RADIUS
    )


def _endpoints(atoms: Atoms) -> tuple[int, ...]:
    """
    Read which two minima a found transition state joins.

    The calculation writes this as the string "0,1", but extxyz does not read
    it back as one: a comma-separated pair of integers comes back as an
    integer array. Both forms are accepted here rather than assuming either,
    since which one appears depends on the reader rather than on us.

    Parameters
    ----------
    atoms
        A found transition state.

    Returns
    -------
    tuple[int, ...]
        The two minimum indices, or empty if the structure carries none.
    """
    connects = atoms.info.get("connects")
    if connects is None:
        return ()
    if isinstance(connects, str):
        try:
            return tuple(int(x) for x in connects.split(","))
        except ValueError:
            return ()
    return tuple(int(x) for x in np.atleast_1d(connects))


@dataclass(frozen=True)
class _Reference:
    """
    One molecule's DFT landscape, and what its structures are compared on.

    Attributes
    ----------
    symbols
        Chemical symbols, in the reference's atom order.
    minima
        Minimum positions, flattened.
    saddles
        Saddle positions, flattened.
    joins
        The two minima each saddle joins, as indices into ``minima``.
    bonds
        Bonding of the molecule, as ``_bonds`` returns it.
    lengths
        Length of each bond, as ``_reference_lengths`` returns them.
    """

    symbols: list[str]
    minima: list[np.ndarray]
    saddles: list[np.ndarray]
    joins: list[tuple[int, int]]
    bonds: frozenset[tuple[int, int]]
    lengths: dict[tuple[int, int], float]


def _reference_lengths(
    base: Atoms, bonds: frozenset[tuple[int, int]], minima: list[np.ndarray]
) -> dict[tuple[int, int], float]:
    """
    Get the length of each of the molecule's bonds, averaged over its DFT minima.

    The structure the bonding is read from is not a stationary point:
    paracetamol's C7-O8 is 1.279 A there against 1.360-1.365 in every DFT
    minimum, which would put a model that reproduces DFT 7% off before it had
    made any error. The minima store their atoms in more than one labelling, so
    each is put in the reference's order first.

    Parameters
    ----------
    base
        Structure fixing the atom order and the bonding.
    bonds
        Its bonds, as ``_bonds`` returns them.
    minima
        DFT minimum positions.

    Returns
    -------
    dict[tuple[int, int], float]
        Mean length in Å of each bond.

    Raises
    ------
    ValueError
        If a minimum is not the molecule, which means the reference is broken.
    """
    pairs = sorted(bonds)
    symbols = base.get_chemical_symbols()
    table = []
    for positions in minima:
        relabelled = _in_reference_labelling(
            Atoms(symbols, positions=np.reshape(positions, (-1, 3))), bonds
        )
        if relabelled is None:
            raise ValueError("A DFT minimum is not the reference molecule")
        table.append([relabelled.get_distance(*pair) for pair in pairs])
    means = np.mean(table, axis=0)
    return {pair: float(length) for pair, length in zip(pairs, means, strict=True)}


def _reference(molecule: _Molecule) -> _Reference:
    """
    Read one molecule's DFT minima and transition states.

    Parameters
    ----------
    molecule
        Molecule to read, from the calculation module's list.

    Returns
    -------
    _Reference
        The molecule's reference landscape.
    """
    path = _data_dir() / molecule.directory
    base = read(path / molecule.xyz)
    network = KineticTransitionNetwork()
    network.read_network(text_path=f"{path}/", text_string="")
    nodes = list(network.G.nodes)
    minima = [network.get_minimum_coords(i) for i in nodes]
    saddles, joins = [], []
    for plus, minus, edge in list(network.G.edges):
        saddles.append(network.get_ts_coords(plus, minus, edge))
        joins.append((nodes.index(plus), nodes.index(minus)))
    bonds = _bonds(base)
    return _Reference(
        symbols=base.get_chemical_symbols(),
        minima=minima,
        saddles=saddles,
        joins=joins,
        bonds=bonds,
        lengths=_reference_lengths(base, bonds, minima),
    )


@dataclass(frozen=True)
class _Found:
    """
    One structure one search found.

    Attributes
    ----------
    atoms
        The structure, with the search's ``kind`` and ``label``.
    start
        Which search found it.
    stationary
        Whether the model's checks confirm it as the kind of stationary point
        it is labelled: a settled minimum, or a first-order saddle.
    """

    atoms: Atoms
    start: int
    stationary: bool


@dataclass(frozen=True)
class _Search:
    """
    What one search found.

    Attributes
    ----------
    minima
        Its minima, in file order.
    saddles
        Its transition states, in file order.
    error
        Why it failed, for a search that raised; None otherwise.
    unevaluated
        How many of its structures the check step could not evaluate.
    """

    minima: tuple[_Found, ...]
    saddles: tuple[_Found, ...]
    error: str | None = None
    unevaluated: int = 0


def _read_search(found_path: Path, checks_path: Path, start: int) -> _Search:
    """
    Read one search's found structures with the model's checks of them.

    Parameters
    ----------
    found_path
        The search's found-structures file.
    checks_path
        The check step's file for it.
    start
        Which search it is.

    Returns
    -------
    _Search
        Its minima and saddles, each marked with whether the checks confirm it.

    Raises
    ------
    ValueError
        If the file holds no structures, the checks are of another version of
        it, or a structure is not labelled as the calculation labels them.
    """
    digest = hashlib.sha256(found_path.read_bytes()).hexdigest()
    frames = read(found_path, index=":", format="extxyz")
    if not frames:
        raise ValueError(f"{found_path.name} holds no structures")
    labels = [str(atoms.info["label"]) for atoms in frames]
    minima, saddles = [], []
    with np.load(checks_path) as checks:
        if str(checks["found_sha256"]) != digest:
            raise ValueError(f"{checks_path.name} checks another version of it")
        if [str(label) for label in checks["labels"]] != labels:
            raise ValueError(f"{checks_path.name} checks other structures")
        for index, atoms in enumerate(frames):
            kind = atoms.info["kind"]
            hessian = checks[f"hessian_{labels[index]}"]
            if kind == "minimum":
                if not re.fullmatch(r"min\d+", labels[index]):
                    raise ValueError(f"minimum labelled {labels[index]!r}")
                held = _held_minimum(checks["max_force"][index], hessian, atoms)
                minima.append(_Found(atoms, start, held))
            elif kind == "transition_state":
                if len(_endpoints(atoms)) != 2:
                    raise ValueError(f"{labels[index]} names no pair of minima")
                saddles.append(_Found(atoms, start, _first_order(hessian, atoms)))
            else:
                raise ValueError(
                    f"{found_path.name} holds a structure of kind {kind!r}"
                )
        unevaluated = int(np.count_nonzero(checks["failed"]))
    return _Search(tuple(minima), tuple(saddles), unevaluated=unevaluated)


def _read_searches(model_name: str) -> dict[tuple[str, int], _Search] | None:
    """
    Read every search one model ran, if they are exactly the protocol's.

    Searches that raised, and structures the check step could not evaluate,
    are warned about: both are scored -- as a search that found nothing, and
    as a structure that is not a stationary point -- and an error of the
    machine rather than the model is reason to run the calculation again.

    Parameters
    ----------
    model_name
        Model whose outputs to read.

    Returns
    -------
    dict[tuple[str, int], _Search] | None
        Each search by molecule and start; None, with a warning, if a search is
        missing, one is not in the protocol, or a file cannot be read.
    """
    directory = CALC_PATH / model_name
    units = [(molecule.key, start) for molecule in MOLECULES for start in range(STARTS)]
    found = {unit: _unit_path(directory, *unit, FOUND_SUFFIX) for unit in units}
    failed = {unit: _unit_path(directory, *unit, FAILED_SUFFIX) for unit in units}
    present = set(directory.glob(f"*{FOUND_SUFFIX}")) | set(
        directory.glob(f"*{FAILED_SUFFIX}")
    )
    problems = [
        f"{found[unit].name} missing"
        for unit in units
        if not found[unit].exists() and not failed[unit].exists()
    ] + [
        f"{path.name} not in the protocol"
        for path in sorted(present - set(found.values()) - set(failed.values()))
    ]
    if problems:
        warn(
            f"{model_name}: {len(problems)} problems with its searches, such as "
            f"{problems[0]}; every metric is NaN",
            stacklevel=2,
        )
        return None
    searches = {}
    for unit in units:
        # The found file wins over a failure record: it is written whole before
        # the record of an earlier failure is removed, so a crash between the
        # two leaves both, and the rerun never revisits a finished search.
        path = found[unit] if found[unit].exists() else failed[unit]
        try:
            if path == failed[unit]:
                error = str(json.loads(path.read_text())["error"])
                searches[unit] = _Search((), (), error=error)
            else:
                checks = _unit_path(directory, *unit, CHECKS_SUFFIX)
                searches[unit] = _read_search(path, checks, unit[1])
        except Exception as exc:
            warn(
                f"Could not read {path} with its checks: {exc}; every metric of "
                f"{model_name} is NaN",
                stacklevel=2,
            )
            return None
    raised = [(unit, s.error) for unit, s in searches.items() if s.error is not None]
    if raised:
        (key, start), error = raised[0]
        warn(
            f"{model_name}: {len(raised)} of {len(units)} searches raised and are "
            f"scored as finding nothing, such as {key} start {start}: {error}",
            stacklevel=2,
        )
    unevaluated = sum(s.unevaluated for s in searches.values())
    if unevaluated:
        warn(
            f"{model_name}: the check step could not evaluate {unevaluated} found "
            "structures, which are scored as not stationary",
            stacklevel=2,
        )
    return searches


def _write_distorted(
    model_name: str,
    molecule_key: str,
    distorted: list[Atoms],
    reference_lengths: dict[tuple[int, int], float],
) -> None:
    """
    Save the flagged minima beside the count they produced.

    A fraction says a model converged impossible geometry but not what the
    geometry was, and the difference between a bond crushed to two thirds and
    one stretched to four thirds is the whole diagnosis. Each structure is
    annotated with the bond that flagged it and both of its lengths, so the
    file answers that without the comparison having to be run again.

    Parameters
    ----------
    model_name
        Model whose minima these are.
    molecule_key
        Molecule they belong to.
    distorted
        Minima ``_is_distorted`` flagged. Nothing is written when there are
        none, so a model with no distorted minima leaves no file rather than
        an empty one.
    reference_lengths
        Reference bond lengths, as ``_reference_lengths`` returns them.
    """
    if not distorted:
        return
    out_dir = OUT_PATH / model_name / DISTORTED_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = []
    for atoms in distorted:
        deviations = _bond_deviations(atoms, reference_lengths)
        # Never empty: a structure is in this list because one of these
        # exceeded the tolerance.
        worst_pair, worst = max(deviations.items(), key=lambda item: item[1])
        # A copy, because the same structures are counted and reported
        # elsewhere and should not acquire these keys there.
        labelled = atoms.copy()
        labelled.info["distorted_bonds"] = sum(
            1 for d in deviations.values() if d > DISTORTED_BOND_TOLERANCE
        )
        labelled.info["worst_bond"] = f"{worst_pair[0]}-{worst_pair[1]}"
        labelled.info["worst_deviation"] = worst
        labelled.info["worst_bond_length"] = float(atoms.get_distance(*worst_pair))
        labelled.info["reference_bond_length"] = reference_lengths[worst_pair]
        annotated.append(labelled)
    write(out_dir / f"{molecule_key}_distorted.extxyz", annotated, format="extxyz")


def _node(atoms: Atoms) -> int:
    """
    Get a found minimum's index in the network of the search that found it.

    Parameters
    ----------
    atoms
        A found minimum, labelled ``minNN`` -- which ``_read_search`` checks.

    Returns
    -------
    int
        The index.
    """
    return int(str(atoms.info["label"]).removeprefix("min"))


def _copy_distances(
    reference: list[np.ndarray],
    found: list[_Found],
    statuses: list[str],
    symbols: list[str],
) -> np.ndarray:
    """
    Build the RMSD from each reference structure to each found copy.

    Only ``GOOD`` copies -- held by the model as the stationary point they are
    labelled, with the molecule's bonds, none of them distorted -- get a
    distance. The others are set infinitely far, so they can never be matched
    or count as the nearest; they stay in the matrix rather than being removed,
    so that column ``i`` is still copy ``i``.

    Parameters
    ----------
    reference
        Reference positions of one kind, flattened.
    found
        Found copies of the same kind.
    statuses
        Status of each copy.
    symbols
        Chemical symbols.

    Returns
    -------
    numpy.ndarray
        Distances, reference by copy.
    """
    matrix = np.full((len(reference), len(found)), np.inf)
    good = [index for index, status in enumerate(statuses) if status == GOOD]
    if good:
        matrix[:, good] = _distances(
            reference,
            [found[index].atoms.get_positions().flatten() for index in good],
            symbols,
        )
    return matrix


def _by_group(matrix: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """
    Reduce distances to copies to distances to the structures they are copies of.

    Parameters
    ----------
    matrix
        Distances, reference by copy.
    groups
        Copies of each structure, as ``_group`` returns them.

    Returns
    -------
    numpy.ndarray
        Distances, reference by structure: each its closest copy's.
    """
    if not groups:
        return np.zeros((matrix.shape[0], 0))
    return np.column_stack([matrix[:, members].min(axis=1) for members in groups])


def _status(found: _Found, reference: _Reference) -> str:
    """
    Say what one found structure is, as ``STATUS_ORDER`` names it.

    Parameters
    ----------
    found
        The found structure.
    reference
        The molecule's reference landscape.

    Returns
    -------
    str
        Its status.
    """
    if not found.stationary:
        return NOT_STATIONARY
    return _classify(found.atoms, reference.bonds, reference.lengths)


def _group_status(statuses: list[str], group: list[int]) -> str:
    """
    Get the status of a structure found more than once: the best of its copies'.

    Parameters
    ----------
    statuses
        Status of each copy, as ``_status`` returns them.
    group
        Indices of the structure's copies.

    Returns
    -------
    str
        The best status, in ``STATUS_ORDER``.
    """
    return min((statuses[index] for index in group), key=STATUS_ORDER.index)


def _extra(kinds: list[str], distances: np.ndarray) -> int:
    """
    Count the structures that are the molecule but no reference structure.

    Further than the matching radius from every reference, not merely left
    unmatched: a second copy of a matched minimum, grouped apart from the first
    because it sat a little further away, is not a minimum DFT lacks.

    Parameters
    ----------
    kinds
        Status of each structure.
    distances
        Distances, reference by structure.

    Returns
    -------
    int
        How many ``GOOD`` structures are further than ``MATCH_RADIUS`` from
        every reference.
    """
    return sum(
        1
        for index, kind in enumerate(kinds)
        if kind == GOOD and not np.any(distances[:, index] < MATCH_RADIUS)
    )


def _join(
    saddle: _Found,
    pair: tuple[int, int],
    endpoint: dict[tuple[int, int], int],
    near: np.ndarray,
) -> bool:
    """
    Say whether one found saddle joins a given pair of reference minima.

    Its endpoints are named by node index within its own search. They must be
    two different nodes -- a saddle whose descents meet is no rearrangement,
    and would pass a reference saddle joining a minimum to a symmetry copy of
    itself -- and each a confirmed minimum within the matching radius of one of
    the pair.

    Parameters
    ----------
    saddle
        The found saddle.
    pair
        The reference minima it should join.
    endpoint
        Index into the found minima of each (search, node).
    near
        Whether each found minimum is a confirmed minimum within the matching
        radius of each reference minimum, reference by found.

    Returns
    -------
    bool
        True if it joins them, either way round.
    """
    ends = _endpoints(saddle.atoms)
    if len(ends) != 2 or ends[0] == ends[1]:
        return False
    plus, minus = (endpoint.get((saddle.start, end)) for end in ends)
    if plus is None or minus is None:
        return False
    first, second = pair
    return bool(
        (near[first, plus] and near[second, minus])
        or (near[first, minus] and near[second, plus])
    )


def _compare_molecule(
    reference: _Reference, minima: list[_Found], saddles: list[_Found]
) -> tuple[dict, list[Atoms]]:
    """
    Match everything the searches of one molecule found against its reference.

    Parameters
    ----------
    reference
        The molecule's reference landscape.
    minima
        Every minimum its searches found, copies included.
    saddles
        Every transition state its searches found, copies included.

    Returns
    -------
    tuple[dict, list[ase.atoms.Atoms]]
        Recovery, distances, connectivity and counts; and one structure per
        distorted minimum, in the reference's atom order, for writing out.
    """
    symbols = reference.symbols
    minima_status = [_status(found, reference) for found in minima]
    saddle_status = [_status(found, reference) for found in saddles]
    minima_groups = _group_copies(minima, minima_status, symbols, MINIMA_GROUP_RADIUS)
    saddle_groups = _group_copies(saddles, saddle_status, symbols, SADDLE_GROUP_RADIUS)
    minima_copies = _copy_distances(reference.minima, minima, minima_status, symbols)
    saddle_copies = _copy_distances(reference.saddles, saddles, saddle_status, symbols)
    minima_distances = _by_group(minima_copies, minima_groups)
    saddle_distances = _by_group(saddle_copies, saddle_groups)
    minima_match = _assign(minima_distances)
    saddle_match = _assign(saddle_distances)
    minima_kinds = [_group_status(minima_status, group) for group in minima_groups]
    saddle_kinds = [_group_status(saddle_status, group) for group in saddle_groups]

    # A recovered reference saddle is credited through any copy within the
    # matching radius of it, whichever group the copy fell in: two copies of one
    # saddle can sit further apart than the saddles' grouping radius. The copies
    # that are not GOOD sit at infinity in the distances, so they never qualify.
    # Each copy's endpoints are read in the network of the search that found it.
    endpoint = {(f.start, _node(f.atoms)): index for index, f in enumerate(minima)}
    near = minima_copies < MATCH_RADIUS
    preserved = sum(
        1
        for ref in saddle_match
        if any(
            saddle_copies[ref, copy] < MATCH_RADIUS
            and _join(saddles[copy], reference.joins[ref], endpoint, near)
            for copy in range(len(saddles))
        )
    )

    minima_nearest = _nearest(minima_distances)
    saddle_nearest = _nearest(saddle_distances)
    distorted = [
        _in_reference_labelling(
            minima[next(i for i in group if minima_status[i] == DISTORTED)].atoms,
            reference.bonds,
        )
        for group, kind in zip(minima_groups, minima_kinds, strict=True)
        if kind == DISTORTED
    ]
    results = {
        "n_reference_minima": len(reference.minima),
        "n_reference_saddles": len(reference.saddles),
        "n_found_minima": len(minima_groups),
        "n_found_saddles": len(saddle_groups),
        "n_settled_minima": len(minima_groups) - minima_kinds.count(NOT_STATIONARY),
        "minima_matched": len(minima_match),
        "saddles_matched": len(saddle_match),
        "minima_rmsd": [d for _, d in minima_match.values()],
        "saddle_rmsd": [d for _, d in saddle_match.values()],
        "minima_nearest": minima_nearest,
        "saddle_nearest": saddle_nearest,
        "minima_near_missed": _near_missed(minima_nearest),
        "connectivity_preserved": preserved,
        "off_reference_minima": minima_kinds.count(OFF_REFERENCE),
        "distorted_minima": minima_kinds.count(DISTORTED),
        "unsettled_minima": minima_kinds.count(NOT_STATIONARY),
        "extra_minima": _extra(minima_kinds, minima_distances),
        "not_first_order_saddles": saddle_kinds.count(NOT_STATIONARY),
        "extra_saddles": _extra(saddle_kinds, saddle_distances),
    }
    return results, distorted


def _compare(model_name: str) -> dict | None:
    """
    Match everything one model found against the reference, molecule by molecule.

    Parameters
    ----------
    model_name
        Model whose search results to match.

    Returns
    -------
    dict | None
        Per-molecule results, as ``_compare_molecule`` returns them; None if
        the model's searches are incomplete.
    """
    searches = _read_searches(model_name)
    if searches is None:
        return None
    results = {}
    for molecule in MOLECULES:
        reference = _reference(molecule)
        runs = [searches[molecule.key, start] for start in range(STARTS)]
        results[molecule.key], distorted = _compare_molecule(
            reference,
            [found for run in runs for found in run.minima],
            [found for run in runs for found in run.saddles],
        )
        _write_distorted(model_name, molecule.key, distorted, reference.lengths)
    return results


def _pooled(comparison: dict | None, numerator: str, denominator: str) -> float:
    """
    Pool a count over molecules.

    Pooled rather than averaged over molecules, so each reference structure
    counts once. Aspirin holds 11 of the 28 minima and 37 of the 67 saddles,
    and averaging per molecule would give ethanol's two minima the same weight
    as all of aspirin's.

    Parameters
    ----------
    comparison
        Per-molecule results, or None if the model's outputs were rejected.
    numerator
        Key to sum on top.
    denominator
        Key to sum underneath.

    Returns
    -------
    float
        The pooled fraction; zero when there is nothing to take a fraction of
        -- a model that found no minima has none distorted, and NaN would void
        its whole score -- and NaN if the outputs were rejected.
    """
    if comparison is None:
        return np.nan
    top = sum(v[numerator] for v in comparison.values())
    bottom = sum(v[denominator] for v in comparison.values())
    return float(top / bottom) if bottom else 0.0


def _median_nearest(comparison: dict | None, key: str) -> float:
    """
    Take the median nearest distance over every reference structure.

    A reference with nothing usable found near it stays in, at
    ``UNFOUND_DISTANCE``, so that a model is not scored on the references it
    happened to reach.

    Parameters
    ----------
    comparison
        Per-molecule results, or None if the model's outputs were rejected.
    key
        ``minima_nearest`` or ``saddle_nearest``.

    Returns
    -------
    float
        The median in Å; ``UNFOUND_DISTANCE`` if more than half the references
        had nothing near them, NaN if the outputs were rejected.
    """
    if comparison is None:
        return np.nan
    distances = [min(d, UNFOUND_DISTANCE) for v in comparison.values() for d in v[key]]
    return float(np.median(distances)) if distances else np.nan


@pytest.fixture
def comparison() -> dict[str, dict | None]:
    """
    Match every model's landscape against the reference.

    Returns
    -------
    dict[str, dict | None]
        Per-model, per-molecule comparison; None for a model whose searches
        are incomplete.
    """
    return {model_name: _compare(model_name) for model_name in MODELS}


def _per_structure(comparison: dict[str, dict | None], key: str) -> dict[str, list]:
    """
    Collect one per-structure value over every molecule, for each model.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.
    key
        Which value.

    Returns
    -------
    dict[str, list]
        The finite values for each model, empty for a model whose searches are
        incomplete. Infinite distances, references with nothing found near
        them, have no place on a plot; the medians keep them.
    """
    return {
        model_name: [
            d for v in (results or {}).values() for d in v[key] if np.isfinite(d)
        ]
        for model_name, results in comparison.items()
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_minima_rmsd.json",
    title="How closely found minima match the DFT ones",
    y_label="RMSD to the matched DFT minimum / Å",
)
def minima_rmsd(comparison: dict[str, dict | None]) -> dict[str, list]:
    """
    Get the distance from each matched DFT minimum to the model's.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.

    Returns
    -------
    dict[str, list]
        Per-structure RMSD in Å.
    """
    return _per_structure(comparison, "minima_rmsd")


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_saddle_rmsd.json",
    title="How closely found transition states match the DFT ones",
    y_label="RMSD to the matched DFT saddle / Å",
)
def saddle_rmsd(comparison: dict[str, dict | None]) -> dict[str, list]:
    """
    Get the distance from each matched DFT transition state to the model's.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.

    Returns
    -------
    dict[str, list]
        Per-structure RMSD in Å.
    """
    return _per_structure(comparison, "saddle_rmsd")


def _summarise(
    values: dict[str, list], statistic: Callable[[np.ndarray], float]
) -> dict[str, float]:
    """
    Reduce each model's per-structure values to one number.

    Parameters
    ----------
    values
        Per-structure values.
    statistic
        Callable such as ``numpy.median``.

    Returns
    -------
    dict[str, float]
        One number per model, NaN where nothing matched.
    """
    results = {}
    for model_name, entries in values.items():
        entries = np.asarray(entries, dtype=float)
        results[model_name] = float(statistic(entries)) if entries.size else np.nan
    return results


def _pooled_per_model(
    comparison: dict[str, dict | None], numerator: str, denominator: str
) -> dict[str, float]:
    """
    Pool one count over molecules for every model.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.
    numerator
        Key to sum on top.
    denominator
        Key to sum underneath.

    Returns
    -------
    dict[str, float]
        The pooled fraction for each model, as ``_pooled`` gives it.
    """
    return {m: _pooled(c, numerator, denominator) for m, c in comparison.items()}


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_minima_nearest.json",
    title="How close the search got to every DFT minimum",
    y_label="Distance to the nearest found minimum / Å",
)
def minima_nearest(comparison: dict[str, dict | None]) -> dict[str, list]:
    """
    Get the distance to the closest found minimum for every DFT minimum.

    Over all reference minima rather than the matched ones, so a model whose
    structures sit a little outside the matching radius is distinguishable
    from one that found nothing like them.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.

    Returns
    -------
    dict[str, list]
        Per-reference distance in Å.
    """
    return _per_structure(comparison, "minima_nearest")


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_saddle_nearest.json",
    title="How close the search got to every DFT transition state",
    y_label="Distance to the nearest found saddle / Å",
)
def saddle_nearest(comparison: dict[str, dict | None]) -> dict[str, list]:
    """
    Get the distance to the closest found saddle for every DFT saddle.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.

    Returns
    -------
    dict[str, list]
        Per-reference distance in Å.
    """
    return _per_structure(comparison, "saddle_nearest")


def _metric_values(comparison: dict[str, dict | None]) -> dict[str, dict]:
    """
    Compute every table row from the comparison.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    minima_rmsd = _per_structure(comparison, "minima_rmsd")
    saddle_rmsd = _per_structure(comparison, "saddle_rmsd")
    return {
        "Minima recovered": _pooled_per_model(
            comparison, "minima_matched", "n_reference_minima"
        ),
        "Matched minima RMSD (median)": _summarise(minima_rmsd, np.median),
        "Matched minima RMSD (90th pct)": _summarise(
            minima_rmsd, lambda v: np.percentile(v, 90)
        ),
        "Saddles recovered": _pooled_per_model(
            comparison, "saddles_matched", "n_reference_saddles"
        ),
        "Matched saddle RMSD (median)": _summarise(saddle_rmsd, np.median),
        "Matched saddle RMSD (90th pct)": _summarise(
            saddle_rmsd, lambda v: np.percentile(v, 90)
        ),
        "Connectivity preserved": _pooled_per_model(
            comparison, "connectivity_preserved", "n_reference_saddles"
        ),
        "Extra minima": _pooled_per_model(
            comparison, "extra_minima", "n_reference_minima"
        ),
        "Extra saddles": _pooled_per_model(
            comparison, "extra_saddles", "n_reference_saddles"
        ),
        "Off-reference minima": _pooled_per_model(
            comparison, "off_reference_minima", "n_settled_minima"
        ),
        "Distorted minima": _pooled_per_model(
            comparison, "distorted_minima", "n_settled_minima"
        ),
        "Unsettled minima": _pooled_per_model(
            comparison, "unsettled_minima", "n_found_minima"
        ),
        "Saddles not first-order": _pooled_per_model(
            comparison, "not_first_order_saddles", "n_found_saddles"
        ),
        "Minima near-missed": _pooled_per_model(
            comparison, "minima_near_missed", "n_reference_minima"
        ),
        "Nearest minima RMSD (median)": {
            m: _median_nearest(c, "minima_nearest") for m, c in comparison.items()
        },
        "Nearest saddle RMSD (median)": {
            m: _median_nearest(c, "saddle_nearest") for m, c in comparison.items()
        },
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "landscape_search_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    comparison: dict[str, dict | None],
    minima_rmsd: dict[str, list],
    saddle_rmsd: dict[str, list],
    minima_nearest: dict[str, list],
    saddle_nearest: dict[str, list],
) -> dict[str, dict]:
    """
    Collect all metrics.

    Parameters
    ----------
    comparison
        Per-model, per-molecule results, as ``comparison`` returns them.
    minima_rmsd
        Requested so that its figure is written; the rows are computed from
        ``comparison``.
    saddle_rmsd
        The same.
    minima_nearest
        The same. The median nearest distance in particular cannot come from
        the figure's data, which leaves out the references nothing was found
        near.
    saddle_nearest
        The same.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return _metric_values(comparison)


def test_landscape_search(metrics: dict[str, dict]) -> None:
    """
    Run the Landscape17 landscape search analysis.

    Parameters
    ----------
    metrics
        All benchmark metric names and dictionary of values for each model.
    """
    # The element info the app filters by comes from the reference molecules,
    # not from get_struct_info as in the stationary points analysis: that reads
    # the mock model's outputs, and the mock cannot produce any here -- basin
    # hopping on a potential that is zero everywhere has no gradient to follow.
    write_struct_info([_data_dir() / m.directory / m.xyz for m in MOLECULES], OUT_PATH)
