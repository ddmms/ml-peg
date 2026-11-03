"""Analyse the Landscape17 landscape search results and emit metrics.

Reads the structures ``calc_landscape_search.py`` found and asks how much of
the DFT landscape they account for. Nothing in the calculation looks at the
reference; the comparison happens here, which keeps the search honest and
means a metric can be redefined without repeating it.

Matching
--------
Every reference structure is paired with at most one found structure and vice
versa, by solving the assignment that minimises total RMSD rather than taking
each reference's nearest in turn. Greedy nearest-first gets this wrong whenever
two reference structures sit close together: the first takes the shared
neighbour and the second is left unmatched although a perfectly good partner
was available. On this reference that is not hypothetical -- paracetamol's
minima 2 and 3 are 0.34 Angstrom apart, barely outside the radius at which two
structures are called the same.

Pairs further apart than that radius are then dropped, so a poor assignment
counts as no match rather than a bad one.

Coverage and accuracy are reported separately throughout. A model that finds
two of twenty eight minima and places both perfectly is not a good model, and
one number that mixes the two would say it was.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ase import neighborlist
from ase.atoms import Atoms
from ase.io import read, write
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from ml_peg.analysis.utils.decorators import build_table, plot_violin
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.landscapes.landscape_search.calc_landscape_search import (
    MOLECULES,
    _data_dir,
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

FOUND_GLOB = "*_found.extxyz"
# The calculation writes one file per starting conformer. They are merged here
# rather than there, so that what each search found on its own stays on disk
# and can be compared.
# No get_struct_info here, unlike the stationary points analysis. It reads
# the mock model's outputs, and the mock cannot produce any: basin hopping
# on a potential that returns zero everywhere has no gradient to follow and
# no curvature to climb. Nothing in this module needs the labels it supplies,
# and when an app page wants structures to show it will have to stage them
# from a model that can actually run.
# Below this RMSD two structures are the same one. The same radius the search
# itself uses to decide whether a newly found minimum is new.
SAME_STRUCTURE = 0.3
# Outer edge of a near miss, as a multiple of SAME_STRUCTURE. A reference
# whose nearest neighbour falls between the two is one the search reached
# but not closely enough to count, which a pass rate alone cannot show.
NEAR_MISS_FACTOR = 1.5
# Stand-in cost for a pairing that must never be made. The assignment
# solver rejects a non-finite cost, so excluded structures are priced out
# of it rather than removed: dropping them from the list would renumber
# every minimum and break the merge indices that connectivity follows.
UNMATCHABLE = 1.0e6
# ASE's natural cutoffs scaled. At 1.0 the covalent radii are tight enough that
# every molecule containing hydrogen reads as fragmented; 1.1 to 1.2 is the
# usable range, and the neighbour list adds the skin to EACH radius, so it is
# left at zero here rather than being used to widen the bonds.
BOND_CUTOFF_SCALE = 1.2
# How far a bond may stray from its length in the reference molecule before the
# minimum holding it is called distorted, as a fraction of that length. The
# bonding graph cannot see this at all: MACE-OFF-23-L converges paracetamol
# minima with C7-O8 crushed to 0.86 A against the reference's 1.28, and they
# come back with the reference's twenty bonds, none added and none lost,
# because a bond at two thirds of its length is still a bond.
#
# Calibrated against those structures rather than guessed. Over the molecule's
# genuine minima the largest deviation of any bond is 7% -- C7-O8 relaxing from
# the reference's 1.279 to 1.363-1.369 -- while the crushed ones sit at 33%.
# A fifth separates the two with roughly threefold headroom below and
# half again above, which is what makes it a threshold rather than a midpoint:
# it is not near either population.
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


def _comparer() -> MolecularSimilarity:
    """
    Build the structure comparison used for every distance here.

    ``distance_criterion`` is zero deliberately: it is an early-exit threshold,
    so any non-zero value returns a distance that is merely below it rather
    than the smallest one. Harmless for a yes-or-no test and wrong for a
    distance that gets reported.

    Returns
    -------
    topsearch.similarity.molecular_similarity.MolecularSimilarity
        Permutation- and inversion-aware comparison.
    """
    return MolecularSimilarity(
        distance_criterion=0.0,
        energy_criterion=1e8,
        weighted=False,
        allow_inversion=True,
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
        no relabelling makes its bonding the reference molecule's.
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


def _bond_lengths(
    atoms: Atoms, bonds: frozenset[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """
    Measure a given set of bonds in one structure.

    Parameters
    ----------
    atoms
        Structure to measure.
    bonds
        Index pairs to measure, as ``_bonds`` returns them.

    Returns
    -------
    dict[tuple[int, int], float]
        Length in Angstrom for each pair.
    """
    return {pair: float(atoms.get_distance(*pair)) for pair in bonds}


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
        Reference bond lengths, as ``_bond_lengths`` returns them.

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
        Reference bond lengths, as ``_bond_lengths`` returns them.

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
    with _seeded():
        for i, ref in enumerate(reference):
            coords = MolecularCoordinates(symbols, np.asarray(ref).copy())
            for j, other in enumerate(found):
                matrix[i, j] = comparer.closest_distance(
                    coords, np.asarray(other).copy()
                )
                coords.position = np.asarray(ref).copy()
    return matrix


def _assign(matrix: np.ndarray) -> dict[int, tuple[int, float]]:
    """
    Pair reference structures with found ones, one to one, at minimum total RMSD.

    Parameters
    ----------
    matrix
        Distances, reference by found.

    Returns
    -------
    dict[int, tuple[int, float]]
        Reference index to the found index it matched and their distance,
        omitting pairs further apart than ``SAME_STRUCTURE``.
    """
    if matrix.size == 0:
        return {}
    rows, columns = linear_sum_assignment(matrix)
    return {
        int(r): (int(c), float(matrix[r, c]))
        for r, c in zip(rows, columns, strict=True)
        if matrix[r, c] < SAME_STRUCTURE
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


def _unphysical(
    structures: list[Atoms],
    reference_bonds: frozenset[tuple[int, int]],
    reference_lengths: dict[tuple[int, int], float],
) -> list[int]:
    """
    Find the structures the molecule cannot physically adopt.

    Either no relabelling of equivalent atoms makes the bonding the reference
    molecule's, or one of the molecule's own bonds is held at a length it
    cannot have. Such a structure is not a
    node of the landscape, so it is kept out of the network -- counted and
    written out first, because how many a model produces is itself a result.

    Parameters
    ----------
    structures
        Minima or transition states found by the search.
    reference_bonds
        Bonding of the reference molecule, as ``_bonds`` returns it.
    reference_lengths
        Reference bond lengths, as ``_bond_lengths`` returns them.

    Returns
    -------
    list[int]
        Indices into ``structures``, ascending.
    """
    flagged = []
    for index, atoms in enumerate(structures):
        relabelled = _in_reference_labelling(atoms, reference_bonds)
        if relabelled is None or _is_distorted(relabelled, reference_lengths):
            flagged.append(index)
    return flagged


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
        1 for d in nearest if SAME_STRUCTURE <= d < NEAR_MISS_FACTOR * SAME_STRUCTURE
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


def _reference(molecule) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list]:
    """
    Read one molecule's DFT minima and transition states.

    Parameters
    ----------
    molecule
        Molecule to read, from the calculation module's list.

    Returns
    -------
    tuple
        Symbols, minima positions, saddle positions, and the pair of minima
        each saddle joins.
    """
    path = _data_dir() / molecule.directory
    symbols = read(path / molecule.xyz).get_chemical_symbols()
    network = KineticTransitionNetwork()
    network.read_network(text_path=f"{path}/", text_string="")
    nodes = list(network.G.nodes)
    minima = [network.get_minimum_coords(i) for i in nodes]
    saddles, joins = [], []
    for plus, minus, edge in list(network.G.edges):
        saddles.append(network.get_ts_coords(plus, minus, edge))
        joins.append((nodes.index(plus), nodes.index(minus)))
    return symbols, minima, saddles, joins


def _merge(
    frames: list[Atoms], symbols: list[str]
) -> tuple[list[Atoms], dict[tuple[int, int], int]]:
    """
    Collapse structures found more than once into one each.

    The searches explore the same landscape independently, so most of what
    they find they find together. Counting those repeats would make a model
    look productive for having been run five times.

    The mapping it returns is what makes connectivity work. A found saddle
    records the minima it joins by their index *within its own search*, and
    those indices mean nothing once the searches are pooled. Returning where
    each original structure ended up lets a saddle's endpoints be followed
    through the merge; without it every lookup misses and connectivity reads
    zero however good the model is.

    Parameters
    ----------
    frames
        All structures of one kind from every search.
    symbols
        Chemical symbols.

    Returns
    -------
    tuple[list[ase.atoms.Atoms], dict[tuple[int, int], int]]
        One structure per distinct geometry, and a map from each original
        (search, node index) to its position in that list.
    """
    comparer = _comparer()
    kept: list[Atoms] = []
    where: dict[tuple[int, int], int] = {}
    with _seeded():
        for atoms in frames:
            position = atoms.get_positions().flatten()
            index = None
            for candidate, other in enumerate(kept):
                coords = MolecularCoordinates(symbols, other.get_positions().flatten())
                if comparer.closest_distance(coords, position.copy()) < SAME_STRUCTURE:
                    index = candidate
                    break
            if index is None:
                index = len(kept)
                kept.append(atoms)
            origin = _origin(atoms)
            if origin is not None:
                where[origin] = index
    return kept, where


def _origin(atoms: Atoms) -> tuple[int, int] | None:
    """
    Get which search a structure came from and its node index there.

    Parameters
    ----------
    atoms
        A found structure.

    Returns
    -------
    tuple[int, int] | None
        The search index and the node index, or None if either is missing.
    """
    start = atoms.info.get("start")
    label = str(atoms.info.get("label", ""))
    if start is None or not label.startswith("min"):
        return None
    try:
        return int(start), int(label[3:])
    except ValueError:
        return None


def _found(model_name: str, molecule_key: str, kind: str) -> list[Atoms]:
    """
    Read everything one model found for one molecule, across all its searches.

    One file per starting conformer, so a search that failed is an absent file
    rather than missing rows. Sorted so that the order does not depend on how
    the filesystem happens to return names, which decides which copy of a
    duplicate structure is kept.

    Parameters
    ----------
    model_name
        Model whose outputs to read.
    molecule_key
        Molecule name.
    kind
        ``minimum`` or ``transition_state``.

    Returns
    -------
    list[ase.atoms.Atoms]
        Structures of that kind, still including duplicates between searches.
    """
    frames: list[Atoms] = []
    for path in sorted(
        (CALC_PATH / model_name).glob(f"{molecule_key}_start*_found.extxyz")
    ):
        found = read(path, index=":")
        frames += found if isinstance(found, list) else [found]
    return [a for a in frames if a.info.get("kind") == kind]


def _searches(model_name: str, molecule_key: str) -> int:
    """
    Count how many searches of this molecule produced a result.

    Parameters
    ----------
    model_name
        Model whose outputs to read.
    molecule_key
        Molecule name.

    Returns
    -------
    int
        Number of starting conformers that finished.
    """
    return len(
        list((CALC_PATH / model_name).glob(f"{molecule_key}_start*_found.extxyz"))
    )


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
        Reference bond lengths, as ``_bond_lengths`` returns them.
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


def _compare(model_name: str) -> dict:
    """
    Match everything one model found against the reference, molecule by molecule.

    Returns
    -------
    dict
        Per-molecule recovery, distances, connectivity and off-reference counts.
    """
    results = {}
    for molecule in MOLECULES:
        symbols, ref_minima, ref_saddles, joins = _reference(molecule)
        found_minima, minima_origin = _merge(
            _found(model_name, molecule.key, "minimum"), symbols
        )
        found_saddles, _ = _merge(
            _found(model_name, molecule.key, "transition_state"), symbols
        )
        if not found_minima:
            continue

        minima_distances = _distances(
            ref_minima, [a.get_positions().flatten() for a in found_minima], symbols
        )
        saddle_distances = _distances(
            ref_saddles,
            [a.get_positions().flatten() for a in found_saddles],
            symbols,
        )
        # A structure the model cannot physically hold is not a node of the
        # landscape, so it is kept out of the network and out of every measure
        # taken on it -- but counted first, and written out, because how much
        # of this a model produces is itself a result. Excluded by column
        # rather than removed from the list: the merge indices that
        # connectivity follows would otherwise all shift.
        reference_atoms = read(_data_dir() / molecule.directory / molecule.xyz)
        reference_bonds = _bonds(reference_atoms)
        reference_lengths = _bond_lengths(reference_atoms, reference_bonds)
        relabelled = [_in_reference_labelling(a, reference_bonds) for a in found_minima]
        off_reference = [r for r in relabelled if r is None]
        # A structure that is not the molecule has no reference order to be put
        # in, so its bonds are measured as they stand.
        distorted = [
            a if r is None else r
            for a, r in zip(found_minima, relabelled, strict=True)
            if _is_distorted(a if r is None else r, reference_lengths)
        ]
        _write_distorted(model_name, molecule.key, distorted, reference_lengths)
        excluded = _unphysical(found_minima, reference_bonds, reference_lengths)
        if excluded:
            minima_distances[:, excluded] = np.inf
        minima_match = _assign(
            np.where(np.isinf(minima_distances), UNMATCHABLE, minima_distances)
        )

        # The same test on transition states: a saddle holding a bond at an
        # impossible length is no more a feature of the molecule's landscape
        # than a minimum is.
        excluded_saddles = _unphysical(
            found_saddles, reference_bonds, reference_lengths
        )
        if excluded_saddles:
            saddle_distances[:, excluded_saddles] = np.inf
        saddle_match = _assign(
            np.where(np.isinf(saddle_distances), UNMATCHABLE, saddle_distances)
        )

        # Which reference minimum each found minimum corresponds to, for the
        # connectivity test. A found minimum matching nothing is -1.
        found_to_ref = {found: ref for ref, (found, _) in minima_match.items()}
        preserved = 0
        for ref_index, (found_index, _) in saddle_match.items():
            saddle = found_saddles[found_index]
            ends = _endpoints(saddle)
            if len(ends) != 2:
                continue
            # The saddle names its endpoints by index within the search that
            # found it, so each is followed through the merge before being
            # compared with the reference.
            # extxyz reads this back as a numpy integer, which the merge
            # keyed by a plain one.
            search = int(saddle.info["start"])
            mapped = sorted(
                found_to_ref.get(minima_origin.get((search, e), -1), -1) for e in ends
            )
            if mapped == sorted(joins[ref_index]):
                preserved += 1

        results[molecule.key] = {
            "n_searches": _searches(model_name, molecule.key),
            "n_reference_minima": len(ref_minima),
            "n_reference_saddles": len(ref_saddles),
            "n_found_minima": len(found_minima),
            "n_found_saddles": len(found_saddles),
            "minima_matched": len(minima_match),
            "saddles_matched": len(saddle_match),
            "minima_rmsd": [d for _, d in minima_match.values()],
            "saddle_rmsd": [d for _, d in saddle_match.values()],
            "minima_nearest": _nearest(minima_distances),
            "saddle_nearest": _nearest(saddle_distances),
            "minima_near_missed": _near_missed(_nearest(minima_distances)),
            "saddle_near_missed": _near_missed(_nearest(saddle_distances)),
            "connectivity_preserved": preserved,
            "off_reference_minima": len(off_reference),
            "distorted_minima": len(distorted),
            "excluded_minima": len(excluded),
            "excluded_saddles": len(excluded_saddles),
        }
    return results


def _pooled(comparison: dict, numerator: str, denominator: str) -> float:
    """
    Pool a count over molecules.

    Pooled rather than averaged over molecules, so each reference structure
    counts once. Aspirin holds 11 of the 28 minima and 37 of the 67 saddles,
    and averaging per molecule would give ethanol's two minima the same weight
    as all of aspirin's.

    Parameters
    ----------
    comparison
        Per-molecule results.
    numerator
        Key to sum on top.
    denominator
        Key to sum underneath.

    Returns
    -------
    float
        The pooled fraction, NaN if nothing was searched.
    """
    top = sum(v[numerator] for v in comparison.values())
    bottom = sum(v[denominator] for v in comparison.values())
    return float(top / bottom) if bottom else np.nan


@pytest.fixture
def comparison() -> dict[str, dict]:
    """
    Match every model's landscape against the reference.

    Returns
    -------
    dict[str, dict]
        Per-model, per-molecule comparison.
    """
    return {model_name: _compare(model_name) for model_name in MODELS}


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_minima_rmsd.json",
    title="How closely found minima match the DFT ones",
    y_label="RMSD to the matched DFT minimum / Å",
)
def minima_rmsd(comparison: dict[str, dict]) -> dict[str, list]:
    """
    Get the distance from each matched DFT minimum to the model's.

    Returns
    -------
    dict[str, list]
        Per-structure RMSD in Å.
    """
    return {
        model_name: [
            d for v in comparison[model_name].values() for d in v["minima_rmsd"]
        ]
        for model_name in MODELS
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_saddle_rmsd.json",
    title="How closely found transition states match the DFT ones",
    y_label="RMSD to the matched DFT saddle / Å",
)
def saddle_rmsd(comparison: dict[str, dict]) -> dict[str, list]:
    """
    Get the distance from each matched DFT transition state to the model's.

    Returns
    -------
    dict[str, list]
        Per-structure RMSD in Å.
    """
    return {
        model_name: [
            d for v in comparison[model_name].values() for d in v["saddle_rmsd"]
        ]
        for model_name in MODELS
    }


def _summarise(values: dict[str, list], statistic) -> dict[str, float]:
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
    for model_name in MODELS:
        entries = np.asarray(values[model_name], dtype=float)
        entries = entries[np.isfinite(entries)]
        results[model_name] = float(statistic(entries)) if entries.size else np.nan
    return results


@pytest.fixture
def get_minima_recovered(comparison: dict[str, dict]) -> dict[str, float]:
    """
    Get the fraction of DFT minima the search found.

    Returns
    -------
    dict[str, float]
        Fraction for each model.
    """
    return {
        m: _pooled(comparison[m], "minima_matched", "n_reference_minima")
        for m in MODELS
    }


@pytest.fixture
def get_saddles_recovered(comparison: dict[str, dict]) -> dict[str, float]:
    """
    Get the fraction of DFT transition states the search found.

    Returns
    -------
    dict[str, float]
        Fraction for each model.
    """
    return {
        m: _pooled(comparison[m], "saddles_matched", "n_reference_saddles")
        for m in MODELS
    }


@pytest.fixture
def get_connectivity(comparison: dict[str, dict]) -> dict[str, float]:
    """
    Get the fraction of DFT saddles both found and joining the same minima.

    Over every reference saddle, not only the matched ones, so a model that
    finds few cannot score well on the few it finds.

    Returns
    -------
    dict[str, float]
        Fraction for each model.
    """
    return {
        m: _pooled(comparison[m], "connectivity_preserved", "n_reference_saddles")
        for m in MODELS
    }


@pytest.fixture
def get_off_reference(comparison: dict[str, dict]) -> dict[str, float]:
    """
    Get the fraction of found minima whose bonding differs from the molecule's.

    Not necessarily wrong: a model may find a genuine isomer and only place it
    at the wrong energy. It is reported because a search that invents structures
    is doing something a recovery fraction cannot show.

    Returns
    -------
    dict[str, float]
        Fraction for each model.
    """
    return {
        m: _pooled(comparison[m], "off_reference_minima", "n_found_minima")
        for m in MODELS
    }


@pytest.fixture
def get_distorted_minima(comparison: dict[str, dict]) -> dict[str, float]:
    """
    Get the fraction of found minima holding a bond at an impossible length.

    Distinct from the off-reference fraction, and not a subset of it. That one
    compares bonding graphs, and a bond crushed to two thirds of its length is
    still a bond, so the graph comes back identical and the minimum reads as
    the right molecule. A model scoring here is wrong about the shape of a
    molecule it has the connectivity of, which the search cannot tell it: the
    structures are converged, and nothing short of the reference geometry says
    they are impossible.

    Returns
    -------
    dict[str, float]
        Fraction for each model.
    """
    return {
        m: _pooled(comparison[m], "distorted_minima", "n_found_minima") for m in MODELS
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_minima_nearest.json",
    title="How close the search got to every DFT minimum",
    y_label="Distance to the nearest found minimum / Å",
)
def minima_nearest(comparison: dict[str, dict]) -> dict[str, list]:
    """
    Get the distance to the closest found minimum for every DFT minimum.

    Over all reference minima rather than the matched ones, so a model whose
    structures sit a little outside the matching radius is distinguishable
    from one that found nothing like them. References with nothing found at
    all are left out, having no finite distance to show; the recovery counts
    already carry that case.

    Returns
    -------
    dict[str, list]
        Per-reference distance in Å.
    """
    return {
        model_name: [
            d
            for v in comparison[model_name].values()
            for d in v["minima_nearest"]
            if np.isfinite(d)
        ]
        for model_name in MODELS
    }


@pytest.fixture
@plot_violin(
    filename=OUT_PATH / "figure_saddle_nearest.json",
    title="How close the search got to every DFT transition state",
    y_label="Distance to the nearest found saddle / Å",
)
def saddle_nearest(comparison: dict[str, dict]) -> dict[str, list]:
    """
    Get the distance to the closest found saddle for every DFT saddle.

    Returns
    -------
    dict[str, list]
        Per-reference distance in Å.
    """
    return {
        model_name: [
            d
            for v in comparison[model_name].values()
            for d in v["saddle_nearest"]
            if np.isfinite(d)
        ]
        for model_name in MODELS
    }


@pytest.fixture
def get_near_missed(comparison: dict[str, dict]) -> dict[str, float]:
    """
    Get the fraction of reference minima the search reached but did not match.

    Returns
    -------
    dict[str, float]
        One fraction per model.
    """
    return {
        m: _pooled(comparison[m], "minima_near_missed", "n_reference_minima")
        for m in MODELS
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
    get_minima_recovered: dict[str, float],
    minima_rmsd: dict[str, list],
    get_saddles_recovered: dict[str, float],
    saddle_rmsd: dict[str, list],
    get_connectivity: dict[str, float],
    get_off_reference: dict[str, float],
    get_distorted_minima: dict[str, float],
    minima_nearest: dict[str, list],
    saddle_nearest: dict[str, list],
    get_near_missed: dict[str, float],
) -> dict[str, dict]:
    """
    Collect all metrics.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Minima recovered": get_minima_recovered,
        "Matched minima RMSD (median)": _summarise(minima_rmsd, np.median),
        "Matched minima RMSD (90th pct)": _summarise(
            minima_rmsd, lambda v: np.percentile(v, 90)
        ),
        "Saddles recovered": get_saddles_recovered,
        "Matched saddle RMSD (median)": _summarise(saddle_rmsd, np.median),
        "Matched saddle RMSD (90th pct)": _summarise(
            saddle_rmsd, lambda v: np.percentile(v, 90)
        ),
        "Connectivity preserved": get_connectivity,
        "Off-reference minima": get_off_reference,
        "Distorted minima": get_distorted_minima,
        "Minima near-missed": get_near_missed,
        "Nearest minima RMSD (median)": _summarise(minima_nearest, np.median),
        "Nearest saddle RMSD (median)": _summarise(saddle_nearest, np.median),
    }


def test_landscape_search(metrics: dict[str, dict]) -> None:
    """
    Run the Landscape17 landscape search analysis.

    Parameters
    ----------
    metrics
        All benchmark metric names and dictionary of values for each model.
    """
    return
