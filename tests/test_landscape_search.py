"""Unit tests for the Landscape17 landscape search analysis."""

from __future__ import annotations

from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import all_changes
from ase.calculators.lj import LennardJones
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.landscapes.landscape_search import analyse_landscape_search
from ml_peg.analysis.landscapes.landscape_search.analyse_landscape_search import (
    ALIGNMENT_SEED,
    DEFAULT_THRESHOLDS,
    DEFAULT_TOOLTIPS,
    DEFAULT_WEIGHTS,
    DISTORTED,
    DISTORTED_SUBDIR,
    GOOD,
    MATCH_RADIUS,
    MINIMA_GROUP_RADIUS,
    NEAR_MISS_FACTOR,
    OFF_REFERENCE,
    REACTION_MODE,
    SECOND_MODE_RATIO,
    UNFOUND_DISTANCE,
    _assign,
    _bond_deviations,
    _bonds,
    _classify,
    _compare_molecule,
    _distances,
    _first_order,
    _Found,
    _group,
    _held_minimum,
    _in_reference_labelling,
    _is_distorted,
    _median_nearest,
    _metric_values,
    _near_missed,
    _nearest,
    _pooled,
    _read_searches,
    _Reference,
    _reference_lengths,
    _seeded,
    _write_distorted,
)
from ml_peg.analysis.landscapes.stationary_points.hessian_metrics import (
    internal_basis,
)
from ml_peg.calcs.landscapes.landscape_search import calc_landscape_search

# Paracetamol's phenol fragment, the three atoms MACE-OFF-23-L gets wrong.
# Real numbers throughout: the DFT minima have C7-O8 at 1.3601-1.3647 Å (mean
# 1.3623) and O8-H17 at about 0.94, the model's good minima put C7-O8 at
# 1.3635-1.3690, and the minima it converges with the bond crushed put it at
# 0.8572-0.8605.
REFERENCE_CO = 1.3623
GOOD_CO = (1.3635, 1.3689, 1.3690)
CRUSHED_CO = (0.8572, 0.8573, 0.8574, 0.8605)
# Displacement of H from O, 0.937 Å long and far enough off the C-O axis that
# C and H are never within bonding range of each other.
OH_OFFSET = (0.4685, 0.8114, 0.0)


def _fragment(c_o: float) -> Atoms:
    """
    Build the C-O-H fragment with its C-O bond at a chosen length.

    Parameters
    ----------
    c_o
        Length of the C-O bond, in Å.

    Returns
    -------
    ase.atoms.Atoms
        The three-atom fragment.
    """
    oxygen = np.array([c_o, 0.0, 0.0])
    return Atoms(
        "COH", positions=[[0.0, 0.0, 0.0], oxygen, oxygen + np.array(OH_OFFSET)]
    )


@pytest.fixture
def reference_lengths() -> dict[tuple[int, int], float]:
    """
    Get the reference fragment's bond lengths, as the comparison builds them.

    Returns
    -------
    dict[tuple[int, int], float]
        Length of each reference bond.
    """
    reference = _fragment(REFERENCE_CO)
    return _reference_lengths(reference, _bonds(reference), [reference.get_positions()])


def test_crushed_bond_is_flagged_although_the_graph_is_unchanged(
    reference_lengths,
) -> None:
    """A crushed bond is still a bond, so only the length can see it."""
    crushed = _fragment(CRUSHED_CO[0])
    assert _bonds(crushed) == _bonds(_fragment(REFERENCE_CO))
    assert _is_distorted(crushed, reference_lengths)


@pytest.mark.parametrize("c_o", GOOD_CO)
def test_genuine_minima_are_not_flagged(c_o, reference_lengths) -> None:
    """
    The model's good minima sit within half a percent of DFT's C-O, which must pass.

    Parameters
    ----------
    c_o
        Length of the C-O bond in one of those minima, in Å.
    """
    assert not _is_distorted(_fragment(c_o), reference_lengths)


@pytest.mark.parametrize("c_o", CRUSHED_CO)
def test_crushed_minima_are_flagged(c_o, reference_lengths) -> None:
    """
    The model's crushed minima compress C-O by 37%, which must fail.

    Parameters
    ----------
    c_o
        Length of the C-O bond in one of those minima, in Å.
    """
    assert _is_distorted(_fragment(c_o), reference_lengths)


def test_the_threshold_sits_between_the_two_populations(reference_lengths) -> None:
    """The calibration itself: neither population is near the threshold."""
    tolerance = analyse_landscape_search.DISTORTED_BOND_TOLERANCE
    worst_good = max(
        max(_bond_deviations(_fragment(c_o), reference_lengths).values())
        for c_o in GOOD_CO
    )
    mildest_crushed = min(
        max(_bond_deviations(_fragment(c_o), reference_lengths).values())
        for c_o in CRUSHED_CO
    )
    assert worst_good < 0.08
    assert mildest_crushed > 0.32
    assert worst_good * 2 < tolerance < mildest_crushed


def test_deviation_is_a_fraction_of_the_reference_length(reference_lengths) -> None:
    """Fractional, not absolute, so one threshold serves every bond."""
    deviations = _bond_deviations(_fragment(CRUSHED_CO[0]), reference_lengths)
    assert deviations[0, 1] == pytest.approx(
        abs(CRUSHED_CO[0] - REFERENCE_CO) / REFERENCE_CO
    )
    assert deviations[1, 2] == pytest.approx(0.0, abs=1e-12)


def test_only_the_reference_bonds_are_measured(reference_lengths) -> None:
    """
    A structure that has rebonded belongs to the off-reference check.

    The hydrogen here has swung round until it bonds to the carbon as well,
    but both of the molecule's own bonds are at their reference lengths, so
    this metric must say nothing about it.
    """
    rebonded = Atoms(
        "COH",
        positions=[
            [0.0, 0.0, 0.0],
            [REFERENCE_CO, 0.0, 0.0],
            [REFERENCE_CO - 0.937, 0.0, 0.0],
        ],
    )
    assert _bonds(rebonded) != _bonds(_fragment(REFERENCE_CO))
    assert not _is_distorted(rebonded, reference_lengths)


def test_nan_coordinates_are_flagged_rather_than_passed(reference_lengths) -> None:
    """A length that cannot be compared is a failure, not a pass."""
    broken = _fragment(REFERENCE_CO)
    positions = broken.get_positions()
    positions[1] = np.nan
    broken.set_positions(positions)
    deviations = _bond_deviations(broken, reference_lengths)
    assert not np.isnan(list(deviations.values())).any()
    assert _is_distorted(broken, reference_lengths)


def test_superposed_atoms_do_not_raise(reference_lengths) -> None:
    """Two atoms on top of each other give a zero length, not an exception."""
    superposed = _fragment(REFERENCE_CO)
    positions = superposed.get_positions()
    positions[2] = positions[1]
    superposed.set_positions(positions)
    assert _bond_deviations(superposed, reference_lengths)[1, 2] == pytest.approx(1.0)
    assert _is_distorted(superposed, reference_lengths)


def test_a_zero_length_reference_bond_does_not_raise() -> None:
    """Dividing by the reference length must not be able to throw."""
    assert _is_distorted(_fragment(REFERENCE_CO), {(0, 1): 0.0})


def test_flagged_structures_are_written_out(monkeypatch, tmp_path) -> None:
    """
    A count is not inspectable; the structures behind it are.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    tmp_path
        Pytest temporary directory.
    """
    monkeypatch.setattr(analyse_landscape_search, "OUT_PATH", tmp_path)
    reference = _fragment(REFERENCE_CO)
    lengths = _reference_lengths(
        reference, _bonds(reference), [reference.get_positions()]
    )
    crushed = _fragment(CRUSHED_CO[0])
    crushed.info["label"] = "min07"

    _write_distorted("a_model", "paracetamol", [crushed], lengths)

    written = tmp_path / "a_model" / DISTORTED_SUBDIR / "paracetamol_distorted.extxyz"
    assert written.exists()
    frames = read(written, index=":")
    assert len(frames) == 1
    info = frames[0].info
    assert info["label"] == "min07"
    assert info["worst_bond"] == "0-1"
    assert info["distorted_bonds"] == 1
    assert info["reference_bond_length"] == pytest.approx(REFERENCE_CO)
    assert info["worst_bond_length"] == pytest.approx(CRUSHED_CO[0])
    assert info["worst_deviation"] == pytest.approx(
        abs(CRUSHED_CO[0] - REFERENCE_CO) / REFERENCE_CO
    )


def test_nothing_is_written_when_nothing_is_flagged(monkeypatch, tmp_path) -> None:
    """
    A clean model leaves no file rather than an empty one.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    tmp_path
        Pytest temporary directory.
    """
    monkeypatch.setattr(analyse_landscape_search, "OUT_PATH", tmp_path)
    _write_distorted("a_model", "paracetamol", [], {(0, 1): REFERENCE_CO})
    assert not (tmp_path / "a_model").exists()


# --- nearest-distance reporting -------------------------------------------
# Recovery is a pass rate over a hard 0.3 A radius, so a reference missed by
# 0.31 A and one missed by 3 A score the same. mace-mp-0a on paracetamol put
# two of four reference minima at 0.308 and 0.352 A and scored zero, which
# reads as total failure rather than as the systematic offset it is. These
# cover reporting the distance alongside the count.


def test_nearest_reports_a_distance_for_every_reference() -> None:
    """Every reference gets a distance, matched or not."""
    matrix = np.array([[0.1, 0.9], [0.8, 0.7], [2.0, 2.5]])
    assert _nearest(matrix) == pytest.approx([0.1, 0.7, 2.0])


def test_nearest_reports_misses_not_only_matches() -> None:
    """A reference just outside the cut still reports how far it was.

    This is the whole point of the measure: ``_assign`` drops the pair, so
    without this the near miss is indistinguishable from nothing found.
    """
    just_outside = MATCH_RADIUS + 0.008
    matrix = np.array([[just_outside]])
    assert _assign(matrix) == {}
    assert _nearest(matrix) == pytest.approx([just_outside])


def test_nearest_is_not_constrained_by_the_assignment() -> None:
    """Two references may name the same nearest structure.

    ``_assign`` is one-to-one and must give that structure to one of them,
    and the other is dropped. The
    question here is "did the search get anywhere near this reference", which
    the one-to-one constraint would distort.
    """
    matrix = np.array([[0.05, 1.0], [0.06, 1.0]])
    assert _nearest(matrix) == pytest.approx([0.05, 0.06])
    # One structure, two references that both want it: the assignment can
    # only hand it to one, and the other drops out with nothing recorded.
    assert len(_assign(matrix)) == 1


def test_nearest_is_infinite_when_nothing_was_found() -> None:
    """No found structures means no distance, reported as infinite."""
    assert _nearest(np.zeros((3, 0))) == [float("inf")] * 3


def test_nearest_of_an_empty_reference_is_empty() -> None:
    """No references means nothing to report."""
    assert _nearest(np.zeros((0, 4))) == []


def test_near_miss_counts_only_the_band_outside_the_radius() -> None:
    """Inside the radius is a match, far outside is a failure; neither counts."""
    inside = MATCH_RADIUS - 0.01
    band = MATCH_RADIUS + 0.01
    far = NEAR_MISS_FACTOR * MATCH_RADIUS + 0.01
    assert _near_missed([inside, band, far]) == 1


def test_near_miss_boundaries_are_half_open() -> None:
    """Exactly at the radius is a near miss; exactly at the outer edge is not."""
    assert _near_missed([MATCH_RADIUS]) == 1
    assert _near_missed([NEAR_MISS_FACTOR * MATCH_RADIUS]) == 0


def test_near_miss_ignores_references_with_nothing_found() -> None:
    """An infinite distance is not a near miss."""
    assert _near_missed([float("inf"), float("inf")]) == 0


def test_the_paracetamol_near_misses_are_counted() -> None:
    """The case this measure exists for.

    mace-mp-0a put two of paracetamol's four reference minima at 0.308 and
    0.352 Å and matched none of the four, which reads as total failure.
    """
    assert _near_missed([0.308, 0.352, 0.586, 0.571]) == 2


# --- classifying a found structure ------------------------------------------
# Rebonded, the molecule with a bond at an impossible length, or the molecule.


def test_classify_accepts_the_molecule(reference_lengths) -> None:
    """The reference geometry itself is the molecule."""
    good = _fragment(REFERENCE_CO)
    assert _classify(good, _bonds(good), reference_lengths) == GOOD


def test_classify_flags_a_rebonded_structure(reference_lengths) -> None:
    """
    Bonding that differs from the reference is enough on its own.

    The hydrogen bonds to the carbon as well while both of the molecule's own
    bonds keep their reference lengths, so only the bonding comparison can
    flag it.
    """
    rebonded = Atoms(
        "COH",
        positions=[
            [0.0, 0.0, 0.0],
            [REFERENCE_CO, 0.0, 0.0],
            [REFERENCE_CO - 0.937, 0.0, 0.0],
        ],
    )
    assert not _is_distorted(rebonded, reference_lengths)
    status = _classify(rebonded, _bonds(_fragment(REFERENCE_CO)), reference_lengths)
    assert status == OFF_REFERENCE


def test_classify_flags_a_distorted_structure(reference_lengths) -> None:
    """A reference bond held at an impossible length is distorted."""
    crushed = _fragment(CRUSHED_CO[0])
    status = _classify(crushed, _bonds(_fragment(REFERENCE_CO)), reference_lengths)
    assert status == DISTORTED


def test_a_migration_is_off_reference_and_not_distorted(reference_lengths) -> None:
    """
    A hydrogen that has moved counts once, as off-reference.

    Measured index by index, the bond it left reads as stretched far past the
    tolerance, which would count the same structure again as distorted.
    """
    migrated = Atoms(
        "COH",
        positions=[[0.0, 0.0, 0.0], [REFERENCE_CO, 0.0, 0.0], [-1.09, 0.0, 0.0]],
    )
    status = _classify(migrated, _bonds(_fragment(REFERENCE_CO)), reference_lengths)
    assert status == OFF_REFERENCE


# --- one-to-one matching ------------------------------------------------------


def test_assign_makes_as_many_matches_as_it_can() -> None:
    """
    The pairing maximises matches within the radius before minimising RMSD.

    The cheaper assignment, 0.05 + 0.32, drops the 0.32 pair and matches one;
    the dearer one, 0.28 + 0.295, matches both.
    """
    matrix = np.array([[0.28, 0.05], [0.32, 0.295]])
    assert _assign(matrix) == {
        0: (0, pytest.approx(0.28)),
        1: (1, pytest.approx(0.295)),
    }


def test_assign_never_pairs_beyond_the_radius() -> None:
    """A pair outside the radius is no match, however it is priced."""
    assert _assign(np.array([[MATCH_RADIUS + 0.01]])) == {}
    assert _assign(np.array([[np.inf, 0.05]])) == {0: (1, pytest.approx(0.05))}


# --- comparing a molecule's search results ----------------------------------
# Built on the C-O-H fragment turned about its C-O axis, which keeps every bond
# length, so each structure is the molecule and the distances between them can
# be set exactly. Distances are plain RMSD, no alignment, so that a test's
# numbers are the ones it builds.

# Distance of the hydrogen from the C-O axis, which sets how far a turn moves it.
_ARM = float(np.hypot(OH_OFFSET[1], OH_OFFSET[2]))


def _plain_rmsd(_comparer, _symbols, first, second) -> float:
    """
    Get the RMSD of two structures as they stand.

    Parameters
    ----------
    _comparer
        Unused; the alignment it would do is what this replaces.
    _symbols
        Unused.
    first
        Flattened positions.
    second
        Flattened positions.

    Returns
    -------
    float
        Root mean square displacement per atom, in Å.
    """
    difference = np.reshape(first, (-1, 3)) - np.reshape(second, (-1, 3))
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


@pytest.fixture
def plain_rmsd(monkeypatch) -> None:
    """
    Use plain RMSD for every distance the comparison takes.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(analyse_landscape_search, "_rmsd", _plain_rmsd)


def _turned(angle: float) -> Atoms:
    """
    Build the fragment with its hydrogen turned about the C-O axis.

    Parameters
    ----------
    angle
        Turn in radians.

    Returns
    -------
    ase.atoms.Atoms
        The fragment, every bond at its reference length.
    """
    atoms = _fragment(REFERENCE_CO)
    positions = atoms.get_positions()
    y, z = positions[2, 1], positions[2, 2]
    positions[2, 1] = y * np.cos(angle) - z * np.sin(angle)
    positions[2, 2] = y * np.sin(angle) + z * np.cos(angle)
    atoms.set_positions(positions)
    return atoms


def _turn(rmsd: float) -> float:
    """
    Get the turn that moves a structure a given RMSD from where it was.

    Parameters
    ----------
    rmsd
        Distance wanted, in Å.

    Returns
    -------
    float
        Turn in radians.
    """
    return float(2 * np.arcsin(rmsd * np.sqrt(3) / (2 * _ARM)))


def _reference(
    minima: list[float],
    saddles: list[float] = (),
    joins: list[tuple[int, int]] = (),
) -> _Reference:
    """
    Build a reference landscape of turned fragments.

    Parameters
    ----------
    minima
        Turn of each reference minimum.
    saddles
        Turn of each reference saddle.
    joins
        The two minima each saddle joins.

    Returns
    -------
    _Reference
        The reference.
    """
    base = _fragment(REFERENCE_CO)
    bonds = _bonds(base)
    return _Reference(
        symbols=base.get_chemical_symbols(),
        minima=[_turned(a).get_positions().flatten() for a in minima],
        saddles=[_turned(a).get_positions().flatten() for a in saddles],
        joins=list(joins),
        bonds=bonds,
        lengths=_reference_lengths(base, bonds, [base.get_positions()]),
    )


def _minimum(angle: float, start: int, node: int, stationary: bool = True) -> _Found:
    """
    Build a minimum one search found.

    Parameters
    ----------
    angle
        Turn of the structure.
    start
        Which search found it.
    node
        Its index in that search's network.
    stationary
        Whether the model's checks confirmed it.

    Returns
    -------
    _Found
        The found minimum.
    """
    atoms = _turned(angle)
    atoms.info.update(kind="minimum", label=f"min{node:02d}")
    return _Found(atoms, start, stationary)


def _saddle(
    angle: float, start: int, ends: tuple[int, int], stationary: bool = True
) -> _Found:
    """
    Build a transition state one search found.

    Parameters
    ----------
    angle
        Turn of the structure.
    start
        Which search found it.
    ends
        The nodes of that search's network it joins.
    stationary
        Whether the model's checks confirmed it.

    Returns
    -------
    _Found
        The found saddle.
    """
    atoms = _turned(angle)
    atoms.info.update(
        kind="transition_state",
        label=f"ts{ends[0]}_{ends[1]}_0",
        connects=f"{ends[0]},{ends[1]}",
    )
    return _Found(atoms, start, stationary)


def _scaled_oh(found: _Found, factor: float) -> _Found:
    """
    Scale a found structure's O-H bond, moving only the hydrogen.

    Scaled by a quarter, the bond is past the distortion tolerance while the
    structure moves 0.135 Å, inside the matching radius.

    Parameters
    ----------
    found
        Structure to change.
    factor
        Scale applied to the bond.

    Returns
    -------
    _Found
        The same structure, its bond scaled.
    """
    positions = found.atoms.get_positions()
    positions[2] = positions[1] + factor * (positions[2] - positions[1])
    found.atoms.set_positions(positions)
    return found


def test_a_full_match(plain_rmsd) -> None:
    """Both minima and the saddle between them found and joined."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, distorted = _compare_molecule(
        reference,
        [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1)],
        [_saddle(np.pi / 2, 0, (0, 1))],
    )
    assert results["minima_matched"] == 2
    assert results["saddles_matched"] == 1
    assert results["connectivity_preserved"] == 1
    assert results["extra_minima"] == 0
    assert results["extra_saddles"] == 0
    assert distorted == []


def test_copies_from_different_searches_count_once(plain_rmsd) -> None:
    """Five searches finding the same minimum have found one minimum."""
    reference = _reference([0.0])
    found = [_minimum(_turn(0.01 * start), start, 0) for start in range(5)]
    results, _ = _compare_molecule(reference, found, [])
    assert results["n_found_minima"] == 1
    assert results["minima_matched"] == 1
    assert results["extra_minima"] == 0


def test_one_structure_cannot_claim_two_references(plain_rmsd) -> None:
    """
    Copies of one structure are one structure, however many searches found it.

    Each of two references 0.2 Å apart is within the radius of a structure
    found twice, midway between them. Matched copy by copy, it would claim
    both.
    """
    half = _turn(0.1)
    reference = _reference([-half, half])
    results, _ = _compare_molecule(
        reference, [_minimum(0.0, 0, 0), _minimum(0.0, 1, 0)], []
    )
    assert results["minima_matched"] == 1


def test_neighbouring_structures_each_keep_their_reference(plain_rmsd) -> None:
    """
    Two found minima 0.24 Å apart are two minima.

    Paracetamol's minima 2 and 3 are 0.34 Å apart. A model that finds both
    puts its two structures closer together than the matching radius, and
    grouping copies at that radius would merge them and lose one reference.
    """
    second = _turn(0.34)
    reference = _reference([0.0, second])
    near = _turn(0.05)
    found = [_minimum(near, 0, 0), _minimum(second - near, 1, 0)]
    assert (
        MATCH_RADIUS
        > _plain_rmsd(None, None, found[0].atoms.positions, found[1].atoms.positions)
        > MINIMA_GROUP_RADIUS
    )
    results, _ = _compare_molecule(reference, found, [])
    assert results["minima_matched"] == 2


def test_a_group_is_as_close_as_its_closest_copy(plain_rmsd) -> None:
    """The copy found first does not stand in for a closer one found later."""
    reference = _reference([0.0])
    found = [_minimum(_turn(0.12), 0, 0), _minimum(_turn(0.02), 1, 0)]
    results, _ = _compare_molecule(reference, found, [])
    assert results["n_found_minima"] == 1
    assert results["minima_rmsd"] == [pytest.approx(0.02)]


def test_a_close_copy_is_not_lost_behind_a_near_miss(plain_rmsd) -> None:
    """
    A minimum 0.04 Å from the reference counts although a 0.31 Å one came first.

    Merged at the matching radius, the near miss found first would absorb the
    close copy and the reference would read as unfound.
    """
    reference = _reference([0.0])
    found = [_minimum(_turn(0.31), 0, 0), _minimum(_turn(0.04), 1, 0)]
    results, _ = _compare_molecule(reference, found, [])
    assert results["minima_matched"] == 1
    assert results["minima_rmsd"] == [pytest.approx(0.04)]
    assert results["extra_minima"] == 1


def test_an_unmatched_minimum_is_extra(plain_rmsd) -> None:
    """A minimum of the molecule that is in no reference is counted."""
    reference = _reference([0.0])
    results, _ = _compare_molecule(
        reference, [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1)], []
    )
    assert results["minima_matched"] == 1
    assert results["extra_minima"] == 1


def test_an_unmatched_saddle_is_extra(plain_rmsd) -> None:
    """The same for transition states."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, _ = _compare_molecule(
        reference,
        [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1)],
        [_saddle(np.pi / 2, 0, (0, 1)), _saddle(-np.pi / 2, 0, (0, 1))],
    )
    assert results["saddles_matched"] == 1
    assert results["extra_saddles"] == 1


def test_a_molecule_where_nothing_was_found_scores_zero(plain_rmsd) -> None:
    """Nothing found is no match against every reference, not a missing molecule."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, _ = _compare_molecule(reference, [], [])
    assert results["n_reference_minima"] == 2
    assert results["n_reference_saddles"] == 1
    assert results["minima_matched"] == 0
    assert results["saddles_matched"] == 0
    assert results["connectivity_preserved"] == 0
    assert results["minima_nearest"] == [np.inf, np.inf]
    assert results["saddle_nearest"] == [np.inf]


# --- the model's checks -------------------------------------------------------


def test_an_unsettled_minimum_is_counted_and_not_matched(plain_rmsd) -> None:
    """A structure the model does not hold as a minimum is not one."""
    reference = _reference([0.0])
    results, _ = _compare_molecule(
        reference, [_minimum(0.0, 0, 0, stationary=False)], []
    )
    assert results["minima_matched"] == 0
    assert results["unsettled_minima"] == 1
    assert results["n_settled_minima"] == 0
    assert results["extra_minima"] == 0
    assert results["minima_nearest"] == [np.inf]


def test_a_saddle_of_the_wrong_order_is_counted_and_not_matched(plain_rmsd) -> None:
    """Nor is a saddle the model's curvature says is not first-order."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, _ = _compare_molecule(
        reference,
        [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1)],
        [_saddle(np.pi / 2, 0, (0, 1), stationary=False)],
    )
    assert results["saddles_matched"] == 0
    assert results["not_first_order_saddles"] == 1
    assert results["connectivity_preserved"] == 0


def test_a_saddle_into_an_unsettled_minimum_joins_nothing(plain_rmsd) -> None:
    """An endpoint that is not a minimum cannot make the connection."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, _ = _compare_molecule(
        reference,
        [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1, stationary=False)],
        [_saddle(np.pi / 2, 0, (0, 1))],
    )
    assert results["saddles_matched"] == 1
    assert results["connectivity_preserved"] == 0


def test_an_unphysical_saddle_is_counted_and_not_matched(plain_rmsd) -> None:
    """A saddle holding a bond at an impossible length is not matched."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    stretched = _scaled_oh(_saddle(np.pi / 2, 0, (0, 1)), 1.25)
    results, _ = _compare_molecule(
        reference, [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1)], [stretched]
    )
    assert results["not_first_order_saddles"] == 0
    assert results["saddles_matched"] == 0


def test_a_distorted_minimum_is_counted_returned_and_not_matched(plain_rmsd) -> None:
    """Distorted minima are counted, handed back for writing, and kept out."""
    reference = _reference([0.0])
    crushed = _scaled_oh(_minimum(0.0, 0, 0), 0.75)
    results, distorted = _compare_molecule(reference, [crushed], [])
    assert results["distorted_minima"] == 1
    assert results["minima_matched"] == 0
    assert len(distorted) == 1


# --- connectivity ---------------------------------------------------------------


def test_any_copy_of_a_saddle_can_preserve_connectivity(plain_rmsd) -> None:
    """
    A later search that wired the saddle correctly is not ignored.

    The first search's copy descends into a structure far from the second
    reference minimum; the second search's copy into the right one.
    """
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    minima = [
        _minimum(0.0, 0, 0),
        _minimum(np.pi - 1.2, 0, 1),
        _minimum(0.0, 1, 0),
        _minimum(np.pi, 1, 1),
    ]
    saddles = [_saddle(np.pi / 2, 0, (0, 1)), _saddle(np.pi / 2, 1, (0, 1))]
    results, _ = _compare_molecule(reference, minima, saddles)
    assert results["connectivity_preserved"] == 1


def test_endpoints_are_read_within_the_saddles_own_search(plain_rmsd) -> None:
    """Node 1 of one search is not node 1 of another."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    minima = [
        _minimum(0.0, 0, 0),
        _minimum(np.pi - 1.2, 0, 1),
        _minimum(np.pi, 1, 1),
    ]
    results, _ = _compare_molecule(reference, minima, [_saddle(np.pi / 2, 0, (0, 1))])
    assert results["connectivity_preserved"] == 0


def test_a_saddle_whose_descents_meet_joins_nothing(plain_rmsd) -> None:
    """
    Both descents into the same node is not a rearrangement.

    Credited, it would pass a reference saddle that joins a minimum to a
    symmetry copy of itself.
    """
    reference = _reference([0.0], [np.pi / 2], [(0, 0)])
    results, _ = _compare_molecule(
        reference, [_minimum(0.0, 0, 0)], [_saddle(np.pi / 2, 0, (0, 0))]
    )
    assert results["saddles_matched"] == 1
    assert results["connectivity_preserved"] == 0


def test_a_saddle_between_two_copies_preserves_a_self_edge(plain_rmsd) -> None:
    """Two distinct nodes that both are the reference minimum make a self-edge."""
    reference = _reference([0.0], [np.pi / 2], [(0, 0)])
    minima = [_minimum(0.0, 0, 0), _minimum(_turn(0.05), 0, 1)]
    results, _ = _compare_molecule(reference, minima, [_saddle(np.pi / 2, 0, (0, 1))])
    assert results["connectivity_preserved"] == 1


def test_connectivity_needs_the_saddle_found(plain_rmsd) -> None:
    """The right endpoints through the wrong saddle are no connection."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, _ = _compare_molecule(
        reference,
        [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1)],
        [_saddle(-np.pi / 2, 0, (0, 1))],
    )
    assert results["saddles_matched"] == 0
    assert results["connectivity_preserved"] == 0


# --- grouping and distances -------------------------------------------------


def test_group_puts_copies_together_in_order(plain_rmsd) -> None:
    """Copies join the first group they are close to; the rest start their own."""
    frames = [_turned(0.0), _turned(np.pi), _turned(_turn(0.01))]
    symbols = frames[0].get_chemical_symbols()
    positions = [a.get_positions().flatten() for a in frames]
    assert _group(positions, symbols, MINIMA_GROUP_RADIUS) == [
        [0, 2],
        [1],
    ]


def test_a_distance_does_not_depend_on_what_else_is_scored() -> None:
    """
    Each pair is aligned from the same seeded start, whatever came before it.

    Seeded once for a whole matrix, a pair's distance would depend on how many
    pairs were aligned before it.
    """
    reference, found, symbols = _scored_pair()
    other = molecule("CH3OH")
    other.rattle(stdev=0.2, seed=7)
    alone = _distances(reference, found, symbols)
    after = _distances(reference, [other.get_positions().flatten(), *found], symbols)
    assert alone[0, 0] == after[0, 1]


# --- curvature at a found saddle -------------------------------------------


def _hessian(
    internal_eigenvalues: list[float], atoms: Atoms | None = None, rigid: float = 0.0
) -> np.ndarray:
    """
    Build a fragment Hessian with chosen internal curvatures.

    Parameters
    ----------
    internal_eigenvalues
        Curvature of each of the three internal modes, in eV/Å².
    atoms
        The fragment it is the Hessian of; unturned by default.
    rigid
        Curvature given to translation and rotation, which must not count.

    Returns
    -------
    numpy.ndarray
        The 9 x 9 Hessian.
    """
    basis = internal_basis(_turned(0.0) if atoms is None else atoms)
    rigid_basis = np.linalg.svd(basis, full_matrices=True)[0][:, basis.shape[1] :]
    return basis @ np.diag(internal_eigenvalues) @ basis.T + rigid * (
        rigid_basis @ rigid_basis.T
    )


@pytest.mark.parametrize(
    "eigenvalues",
    [
        [-0.4588, -0.0012, 0.0583],  # the DFT reference's paracetamol ts0_1_0
        [-0.5059, -0.0023, 0.1],
        [-0.2, 0.05, 0.1],
    ],
)
def test_one_dominant_negative_mode_is_first_order(eigenvalues) -> None:
    """
    A second negative mode much smaller than the reaction mode is allowed.

    Parameters
    ----------
    eigenvalues
        Internal curvatures, in eV/Å².
    """
    assert _first_order(_hessian(eigenvalues), _turned(0.0))


@pytest.mark.parametrize(
    "eigenvalues",
    [
        [-0.0807, -0.0174, 0.1],  # mace-mp-0a aspirin ts0_7_1, ratio 0.22
        [0.0041, 0.0468, 0.1],  # mace-mp-0a ethanol ts1_1_0, no negative mode
        [-0.0005, 0.05, 0.1],  # a reaction mode too shallow to be one
    ],
)
def test_other_curvatures_are_not_first_order(eigenvalues) -> None:
    """
    Two comparable negative modes, or none, is not a first-order saddle.

    Parameters
    ----------
    eigenvalues
        Internal curvatures, in eV/Å².
    """
    assert not _first_order(_hessian(eigenvalues), _turned(0.0))


def test_the_limits_of_the_rule() -> None:
    """Just inside the ratio and the reaction-mode floor passes; just past fails."""
    reaction = -0.2
    inside = [reaction, 0.99 * SECOND_MODE_RATIO * reaction, 0.1]
    past = [reaction, 1.01 * SECOND_MODE_RATIO * reaction, 0.1]
    assert _first_order(_hessian(inside), _turned(0.0))
    assert not _first_order(_hessian(past), _turned(0.0))
    assert _first_order(_hessian([-1.01 * REACTION_MODE, 0.05, 0.1]), _turned(0.0))
    assert not _first_order(_hessian([-0.99 * REACTION_MODE, 0.05, 0.1]), _turned(0.0))


def test_rigid_body_motion_is_projected_out() -> None:
    """Curvature along translation and rotation is not a mode of the molecule."""
    assert _first_order(_hessian([-0.2, 0.05, 0.1], rigid=-5.0), _turned(0.0))


def test_a_non_finite_hessian_is_not_first_order() -> None:
    """A Hessian the model could not compute confirms nothing."""
    hessian = _hessian([-0.2, 0.05, 0.1])
    hessian[0, 0] = np.nan
    assert not _first_order(hessian, _turned(0.0))


# --- reading the searches -----------------------------------------------------


@pytest.fixture
def two_searches(monkeypatch, tmp_path) -> SimpleNamespace:
    """
    Point the reader at an empty model directory expecting two searches.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    tmp_path
        Pytest temporary directory.

    Returns
    -------
    SimpleNamespace
        The model's directory, as ``directory``.
    """
    monkeypatch.setattr(analyse_landscape_search, "CALC_PATH", tmp_path)
    monkeypatch.setattr(
        analyse_landscape_search, "MOLECULES", (SimpleNamespace(key="frag"),)
    )
    monkeypatch.setattr(analyse_landscape_search, "STARTS", 2)
    directory = tmp_path / "a_model"
    directory.mkdir()
    return SimpleNamespace(directory=directory)


def _frag_path(directory: Path, start: int, suffix: str) -> Path:
    """
    Get one of a test search's files, named as the calculation names them.

    Parameters
    ----------
    directory
        The model's output directory.
    start
        Which search.
    suffix
        Which file, as the calculation's suffix constants name it.

    Returns
    -------
    pathlib.Path
        The path.
    """
    return calc_landscape_search._unit_path(directory, "frag", start, suffix)


def _write_found(
    directory: Path,
    start: int,
    max_force: tuple[float, float] = (1e-3, 1e-3),
    lowest: tuple[float, float] = (0.05, 0.05),
    failed: tuple[bool, bool, bool] = (False, False, False),
) -> None:
    """
    Write one search's found structures and the model's checks of them.

    Two minima and the saddle between them.

    Parameters
    ----------
    directory
        The model's output directory.
    start
        Which search.
    max_force
        Largest force left on each minimum, eV/Å.
    lowest
        Lowest internal curvature of each minimum, eV/Å².
    failed
        Whether the check step could not evaluate each structure.
    """
    frames = [
        _minimum(0.0, start, 0).atoms,
        _minimum(np.pi, start, 1).atoms,
        _saddle(np.pi / 2, start, (0, 1)).atoms,
    ]
    found = _frag_path(directory, start, calc_landscape_search.FOUND_SUFFIX)
    write(found, frames, format="extxyz")
    hessians = [
        _hessian([lowest[0], 0.1, 0.2], frames[0]),
        _hessian([lowest[1], 0.1, 0.2], frames[1]),
        _hessian([-0.2, 0.05, 0.1], frames[2]),
    ]
    np.savez(
        _frag_path(directory, start, calc_landscape_search.CHECKS_SUFFIX),
        labels=np.array([a.info["label"] for a in frames]),
        kinds=np.array([a.info["kind"] for a in frames]),
        max_force=np.array([*max_force, 1e-3]),
        failed=np.array(failed),
        found_sha256=np.array(hashlib.sha256(found.read_bytes()).hexdigest()),
        **{
            f"hessian_{a.info['label']}": hessian
            for a, hessian in zip(frames, hessians, strict=True)
        },
    )


def _write_failed(directory: Path, start: int) -> None:
    """
    Write the record of a search that raised.

    Parameters
    ----------
    directory
        The model's output directory.
    start
        Which search.
    """
    _frag_path(directory, start, calc_landscape_search.FAILED_SUFFIX).write_text(
        json.dumps({"molecule": "frag", "start": start, "error": "RuntimeError: x"})
    )


def test_complete_searches_are_read(two_searches) -> None:
    """
    One search found structures, the other raised and found nothing.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _write_failed(two_searches.directory, 1)
    with pytest.warns(UserWarning, match="RuntimeError: x"):
        searches = _read_searches("a_model")
    assert set(searches) == {("frag", 0), ("frag", 1)}
    assert [f.stationary for f in searches["frag", 0].minima] == [True, True]
    assert [f.stationary for f in searches["frag", 0].saddles] == [True]
    assert {f.start for f in searches["frag", 0].minima} == {0}
    assert searches["frag", 1].minima == ()
    assert searches["frag", 1].saddles == ()


@pytest.mark.parametrize(
    ("max_force", "lowest", "expected"),
    [
        ((2e-2, 1e-3), (0.05, 0.05), [False, True]),
        # The second-order saddle topsearch's own minimum check passes.
        ((1e-3, 5.9e-3), (0.05, -0.214), [True, False]),
        ((np.nan, 1e-3), (0.05, 0.05), [False, True]),
        ((1e-3, 1e-3), (np.nan, 0.05), [False, True]),
    ],
)
def test_a_minimum_must_be_settled_with_no_downhill_direction(
    two_searches, max_force, lowest, expected
) -> None:
    """
    Force left above the tolerance, or a downhill direction, is no minimum.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    max_force
        Largest force left on each minimum, eV/Å.
    lowest
        Lowest internal curvature of each minimum, eV/Å².
    expected
        Whether each should be held as a minimum.
    """
    _write_found(two_searches.directory, 0, max_force, lowest)
    _write_found(two_searches.directory, 1)
    searches = _read_searches("a_model")
    assert [f.stationary for f in searches["frag", 0].minima] == expected


def test_structures_the_check_step_could_not_evaluate_are_warned_about(
    two_searches,
) -> None:
    """
    An error of the machine rather than the model is reason to rerun.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0, failed=(True, False, False))
    _write_found(two_searches.directory, 1)
    with pytest.warns(UserWarning, match="could not evaluate 1 found"):
        _read_searches("a_model")


def _void(directory: Path) -> None:
    """
    Check that the model's searches are rejected, with a warning.

    Parameters
    ----------
    directory
        The model's output directory.
    """
    with pytest.warns(UserWarning, match="NaN"):
        assert _read_searches(directory.name) is None


def test_a_missing_search_voids_the_model(two_searches) -> None:
    """
    A search with no outputs at all is not scored as one that found nothing.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _void(two_searches.directory)


def test_an_unexpected_search_voids_the_model(two_searches) -> None:
    """
    A search the protocol does not run, left over from another, is not pooled in.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    for start in range(3):
        _write_found(two_searches.directory, start)
    _void(two_searches.directory)


def test_a_found_file_outranks_a_stale_failure_record(two_searches) -> None:
    """
    A search that failed once and then succeeded is read as a success.

    The found file is written whole before the old failure record is removed,
    so a crash between the two leaves both; the rerun skips a search whose
    found file exists, and would never remove the record.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _write_found(two_searches.directory, 1)
    _write_failed(two_searches.directory, 1)
    searches = _read_searches("a_model")
    assert len(searches["frag", 1].minima) == 2


def test_a_truncated_found_file_voids_the_model(two_searches) -> None:
    """
    A file cut off mid-write does not abort the analysis or score a fragment.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _write_found(two_searches.directory, 1)
    path = _frag_path(two_searches.directory, 1, calc_landscape_search.FOUND_SUFFIX)
    text = path.read_text()
    path.write_text(text[: len(text) // 2])
    _void(two_searches.directory)


def test_an_empty_found_file_voids_the_model(two_searches) -> None:
    """
    A found file with no structures is not a search that found nothing.

    The calculation never writes one -- an empty search is recorded as a
    failure -- so an empty file is one a crash left before its data reached
    the disk.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _write_found(two_searches.directory, 1)
    path = _frag_path(two_searches.directory, 1, calc_landscape_search.FOUND_SUFFIX)
    path.write_text("")
    checks = _frag_path(two_searches.directory, 1, calc_landscape_search.CHECKS_SUFFIX)
    np.savez(
        checks,
        labels=np.array([], dtype=str),
        kinds=np.array([], dtype=str),
        max_force=np.array([]),
        failed=np.array([], dtype=bool),
        found_sha256=np.array(hashlib.sha256(b"").hexdigest()),
    )
    _void(two_searches.directory)


def test_found_structures_without_checks_void_the_model(two_searches) -> None:
    """
    Structures the model's checks never ran on are not judged as passing.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _write_found(two_searches.directory, 1)
    _frag_path(two_searches.directory, 1, calc_landscape_search.CHECKS_SUFFIX).unlink()
    _void(two_searches.directory)


def _rewrite_checks(directory: Path, start: int, **changes: np.ndarray) -> None:
    """
    Change some arrays of a search's checks file.

    Parameters
    ----------
    directory
        The model's output directory.
    start
        Which search.
    **changes
        Arrays to replace, by key.
    """
    path = _frag_path(directory, start, calc_landscape_search.CHECKS_SUFFIX)
    with np.load(path) as checks:
        arrays = dict(checks)
    arrays.update(changes)
    np.savez(path, **arrays)


def test_checks_of_other_structures_void_the_model(two_searches) -> None:
    """
    Checks whose structures are not the file's are not applied.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _write_found(two_searches.directory, 1)
    _rewrite_checks(
        two_searches.directory, 0, labels=np.array(["min00", "min02", "ts0_1_0"])
    )
    _void(two_searches.directory)


def test_checks_of_another_version_of_the_file_void_the_model(two_searches) -> None:
    """
    A found file replaced after its checks ran is not judged by them.

    A search rerun finds the same number of minima and saddles under the same
    labels, so the labels alone cannot tell the two versions apart.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    _write_found(two_searches.directory, 1)
    _rewrite_checks(two_searches.directory, 0, found_sha256=np.array("0" * 64))
    _void(two_searches.directory)


@pytest.mark.parametrize(
    ("index", "info"),
    [(0, {"label": "minimum0"}), (2, {"connects": "0-1"}), (2, {"kind": "saddle"})],
)
def test_structures_labelled_otherwise_void_the_model(
    two_searches, index, info
) -> None:
    """
    A change in how the calculation labels structures fails loudly.

    Read as they stand, a minimum without its node index and a saddle without
    the pair it joins would quietly score no connectivity at all.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    index
        Which structure to relabel.
    info
        What to change in it.
    """
    _write_found(two_searches.directory, 0)
    _write_found(two_searches.directory, 1)
    path = _frag_path(two_searches.directory, 0, calc_landscape_search.FOUND_SUFFIX)
    frames = read(path, index=":")
    frames[index].info.update(info)
    write(path, frames, format="extxyz")
    labels = np.array([str(a.info["label"]) for a in frames])
    with np.load(
        _frag_path(two_searches.directory, 0, calc_landscape_search.CHECKS_SUFFIX)
    ) as checks:
        hessians = {
            f"hessian_{new}": checks[f"hessian_{old}"]
            for old, new in zip(checks["labels"], labels, strict=True)
        }
    _rewrite_checks(
        two_searches.directory,
        0,
        labels=labels,
        found_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),
        **hessians,
    )
    _void(two_searches.directory)


def test_a_missing_model_directory_voids_the_model(two_searches) -> None:
    """
    A model that never ran is NaN, not an error.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    with pytest.warns(UserWarning, match="NaN"):
        assert _read_searches("never_ran") is None


def test_the_analysis_reads_what_the_check_step_writes(
    monkeypatch: pytest.MonkeyPatch, out_dir: Path
) -> None:
    """
    The checks file the calculation writes is the one the analysis reads.

    The real check step on a toy model's found file, read back by the real
    reader: the true minimum is held, and the linear chain is neither a minimum
    nor a first-order saddle -- it bends downhill two ways.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    out_dir
        The toy model's output directory, where the calculation writes.
    """
    chain = _frame(_chain(), "transition_state", "ts0_1_0")
    chain.info["connects"] = "0,1"
    frames = [
        _frame(_triangle(), "minimum", "min00"),
        _frame(_chain(), "minimum", "min01"),
        chain,
    ]
    _write_found_frames(out_dir, 0, frames)
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    monkeypatch.setattr(analyse_landscape_search, "CALC_PATH", out_dir.parent)
    monkeypatch.setattr(
        analyse_landscape_search, "MOLECULES", (SimpleNamespace(key=TOY),)
    )
    monkeypatch.setattr(analyse_landscape_search, "STARTS", 1)
    search = _read_searches(out_dir.name)[TOY, 0]
    assert [f.stationary for f in search.minima] == [True, False]
    assert [f.stationary for f in search.saddles] == [False]


def test_a_second_copy_within_the_radius_is_not_extra(plain_rmsd) -> None:
    """
    A minimum DFT has is not an extra minimum because it was found twice.

    Copies 0.02 and 0.20 Å from the reference are too far apart to group, and
    the matching takes one; the other is the same minimum, not a new one.
    """
    reference = _reference([0.0])
    found = [_minimum(_turn(0.02), 0, 0), _minimum(_turn(0.20), 1, 0)]
    results, _ = _compare_molecule(reference, found, [])
    assert results["minima_matched"] == 1
    assert results["extra_minima"] == 0


def test_a_structure_the_model_does_not_hold_anchors_no_group(plain_rmsd) -> None:
    """
    An unsettled structure between two references does not join their minima.

    Grouped with it, both good minima would fall in one group and match only
    one of the two references.
    """
    second = _turn(0.34)
    reference = _reference([0.0, second])
    near = _turn(0.05)
    found = [
        _minimum(second / 2, 0, 0, stationary=False),
        _minimum(near, 1, 0),
        _minimum(second - near, 2, 0),
    ]
    results, _ = _compare_molecule(reference, found, [])
    assert results["minima_matched"] == 2
    assert results["unsettled_minima"] == 1


def test_a_group_never_stretches_past_its_radius(plain_rmsd) -> None:
    """
    Copies either side of a first one, too far from each other, stay apart.

    Anchored on its first member alone, a group could span twice the radius.
    """
    step = _turn(0.14)
    frames = [_turned(0.0), _turned(step), _turned(-step)]
    symbols = frames[0].get_chemical_symbols()
    positions = [a.get_positions().flatten() for a in frames]
    assert _group(positions, symbols, MINIMA_GROUP_RADIUS) == [[0, 1], [2]]


def test_neighbouring_saddles_each_keep_their_reference(plain_rmsd) -> None:
    """
    Aspirin's saddles 30 and 34 are 0.186 Å apart; a model can find both.

    Its two saddles, each 0.02 Å from one reference, are 0.146 Å apart:
    grouped at the minima's radius they would be one.
    """
    second = _turn(0.186)
    near = _turn(0.02)
    reference = _reference([0.0, np.pi], [0.0, second], [(0, 1), (0, 1)])
    minima = [_minimum(np.pi / 2, 0, 0), _minimum(-np.pi / 2, 0, 1)]
    saddles = [_saddle(near, 0, (0, 1)), _saddle(second - near, 1, (0, 1))]
    results, _ = _compare_molecule(reference, minima, saddles)
    assert results["saddles_matched"] == 2


def test_a_saddle_joins_its_minima_either_way_round(plain_rmsd) -> None:
    """A saddle recorded as joining 1 and 0 joins 0 and 1."""
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, _ = _compare_molecule(
        reference,
        [_minimum(0.0, 0, 0), _minimum(np.pi, 0, 1)],
        [_saddle(np.pi / 2, 0, (1, 0))],
    )
    assert results["connectivity_preserved"] == 1


def test_a_copy_grouped_apart_can_still_preserve_connectivity(plain_rmsd) -> None:
    """
    Two copies of one saddle 0.14 Å apart fall in different groups.

    Both are within the matching radius of the reference saddle; the closer is
    wired to the wrong minima and the further to the right ones, which still
    counts.
    """
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    minima = [
        _minimum(0.0, 0, 0),
        _minimum(np.pi - 1.2, 0, 1),
        _minimum(0.0, 1, 0),
        _minimum(np.pi, 1, 1),
    ]
    saddles = [
        _saddle(np.pi / 2 + _turn(0.05), 0, (0, 1)),
        _saddle(np.pi / 2 + _turn(0.19), 1, (0, 1)),
    ]
    results, _ = _compare_molecule(reference, minima, saddles)
    assert results["n_found_saddles"] == 2
    assert results["connectivity_preserved"] == 1


def test_connectivity_needs_the_crediting_copy_within_the_radius(plain_rmsd) -> None:
    """
    Only a copy that is itself the reference saddle can credit its connection.

    The group matches through its closer copy, 0.25 Å away, which joins the
    wrong minima; the copy joining the right ones is 0.33 Å away.
    """
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    minima = [
        _minimum(0.0, 0, 0),
        _minimum(np.pi - 1.2, 0, 1),
        _minimum(0.0, 1, 0),
        _minimum(np.pi, 1, 1),
    ]
    saddles = [
        _saddle(np.pi / 2 + _turn(0.25), 0, (0, 1)),
        _saddle(np.pi / 2 + _turn(0.33), 1, (0, 1)),
    ]
    results, _ = _compare_molecule(reference, minima, saddles)
    assert results["saddles_matched"] == 1
    assert results["connectivity_preserved"] == 0


def test_searches_read_from_disk_score_end_to_end(two_searches, plain_rmsd) -> None:
    """
    Structures written as the calculation writes them score as built.

    Read back, ``connects`` is an integer array rather than the string that
    was written, and the saddle here names its minima in reverse.

    Parameters
    ----------
    two_searches
        Reader pointed at an empty model directory.
    """
    _write_found(two_searches.directory, 0)
    path = _frag_path(two_searches.directory, 0, calc_landscape_search.FOUND_SUFFIX)
    frames = read(path, index=":")
    frames[2].info["connects"] = "1,0"
    write(path, frames, format="extxyz")
    _rewrite_checks(
        two_searches.directory,
        0,
        found_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    _write_failed(two_searches.directory, 1)
    with pytest.warns(UserWarning, match="raised"):
        searches = _read_searches("a_model")
    run = searches["frag", 0]
    reference = _reference([0.0, np.pi], [np.pi / 2], [(0, 1)])
    results, _ = _compare_molecule(reference, list(run.minima), list(run.saddles))
    assert results["minima_matched"] == 2
    assert results["connectivity_preserved"] == 1


def _molecule_results(**counts: float) -> dict:
    """
    Build one molecule's comparison results, every count distinct by default.

    Parameters
    ----------
    **counts
        Values to set instead of the defaults.

    Returns
    -------
    dict
        Results as ``_compare_molecule`` returns them.
    """
    results = {
        "n_reference_minima": 4,
        "n_reference_saddles": 10,
        "n_found_minima": 9,
        "n_found_saddles": 12,
        "n_settled_minima": 8,
        "minima_matched": 3,
        "saddles_matched": 6,
        "minima_rmsd": [0.01, 0.03, 0.05],
        "saddle_rmsd": [0.02, 0.04],
        "minima_nearest": [0.01, 0.03, 0.05, np.inf],
        "saddle_nearest": [0.02, np.inf, np.inf],
        "minima_near_missed": 1,
        "connectivity_preserved": 5,
        "off_reference_minima": 2,
        "distorted_minima": 1,
        "unsettled_minima": 7,
        "extra_minima": 11,
        "not_first_order_saddles": 13,
        "extra_saddles": 14,
    }
    results.update(counts)
    return results


def test_every_row_has_its_own_numerator_and_denominator() -> None:
    """
    Each row pools its own count over its own total, over every molecule.

    Two molecules of different sizes, so that pooling and averaging differ,
    and a model whose outputs were rejected.
    """
    comparison = {
        "a": {
            "small": _molecule_results(),
            "large": _molecule_results(
                n_reference_minima=12,
                minima_matched=12,
                n_reference_saddles=40,
                connectivity_preserved=10,
                saddles_matched=20,
                n_settled_minima=2,
                distorted_minima=0,
            ),
        },
        "b": None,
    }
    values = _metric_values(comparison)
    expected = {
        "Minima recovered": 15 / 16,
        "Saddles recovered": 26 / 50,
        "Connectivity preserved": 15 / 50,
        "Extra minima": 22 / 16,
        "Extra saddles": 28 / 50,
        "Off-reference minima": 4 / 10,
        "Distorted minima": 1 / 10,
        "Unsettled minima": 14 / 18,
        "Saddles not first-order": 26 / 24,
        "Minima near-missed": 2 / 16,
        "Matched minima RMSD (median)": 0.03,
        "Matched saddle RMSD (median)": 0.03,
        "Nearest minima RMSD (median)": 0.04,
        "Nearest saddle RMSD (median)": UNFOUND_DISTANCE,
    }
    for name, value in expected.items():
        assert values[name]["a"] == pytest.approx(value), name
        assert np.isnan(values[name]["b"]), name
    assert set(values) == set(DEFAULT_THRESHOLDS)


def test_a_fraction_of_nothing_is_zero() -> None:
    """A model that found no minima has none distorted, rather than a NaN score."""
    results = {"m": _molecule_results(n_settled_minima=0, distorted_minima=0)}
    assert _pooled(results, "distorted_minima", "n_settled_minima") == 0.0


# --- holding a found minimum as one -------------------------------------------


def test_a_settled_minimum_is_held() -> None:
    """No force left and every internal curvature upward is a minimum."""
    atoms = _turned(0.0)
    assert _held_minimum(1e-3, _hessian([0.05, 0.1, 0.2], atoms), atoms)


@pytest.mark.parametrize(
    ("max_force", "curvatures"),
    [
        (2e-2, [0.05, 0.1, 0.2]),  # force left above the tolerance
        (5.9e-3, [-0.214, 0.1, 0.2]),  # a downhill direction, force below it
        (1e-3, [-1.01 * REACTION_MODE, 0.1, 0.2]),  # just past the floor
    ],
)
def test_a_minimum_with_force_or_a_downhill_direction_is_not_held(
    max_force, curvatures
) -> None:
    """
    Either is enough to say the structure is not a minimum.

    Parameters
    ----------
    max_force
        Largest force left on any atom, eV/Å.
    curvatures
        Internal curvatures, eV/Å².
    """
    atoms = _turned(0.0)
    assert not _held_minimum(max_force, _hessian(curvatures, atoms), atoms)


def test_rigid_body_motion_does_not_unsettle_a_minimum() -> None:
    """Curvature along translation and rotation is not a downhill direction."""
    atoms = _turned(0.0)
    hessian = _hessian([0.05, 0.1, 0.2], atoms, rigid=-5.0)
    assert _held_minimum(1e-3, hessian, atoms)


# --- pooling over molecules ------------------------------------------------------


def test_a_model_with_no_results_pools_to_nan() -> None:
    """Rejected outputs give NaN, never a fraction of what happened to be read."""
    assert np.isnan(_pooled(None, "minima_matched", "n_reference_minima"))


def test_the_nearest_median_counts_references_with_nothing_found() -> None:
    """
    A reference the search never came near stays in the median.

    Dropped, a model that found two molecules' structures well would report a
    small median over those alone.
    """
    one = {"a": {"minima_nearest": [0.05, np.inf]}, "b": {"minima_nearest": [np.inf]}}
    two = {"a": {"minima_nearest": [0.05, 0.1]}, "b": {"minima_nearest": [np.inf]}}
    assert _median_nearest(one, "minima_nearest") == UNFOUND_DISTANCE
    assert _median_nearest(two, "minima_nearest") == pytest.approx(0.1)
    assert np.isnan(_median_nearest(None, "minima_nearest"))


# --- the reference ----------------------------------------------------------


def test_reference_lengths_are_the_mean_over_the_dft_minima() -> None:
    """The xyz the bonding is read from is not where the bonds are measured."""
    base = _fragment(1.2795)
    minima = [_fragment(1.36).get_positions(), _fragment(1.37).get_positions()]
    lengths = _reference_lengths(base, _bonds(base), minima)
    assert lengths[0, 1] == pytest.approx(1.365)
    assert lengths[1, 2] == pytest.approx(np.linalg.norm(OH_OFFSET))


def test_reference_lengths_relabel_each_minimum() -> None:
    """
    A DFT minimum stored with equivalent atoms exchanged is measured correctly.

    Index by index, the exchanged hydrogen's C-H bond would be measured as an
    O-H bond. Relabelled, every bond is its own length again, give or take
    equivalent bonds -- a methyl's three C-H, 0.7% apart here -- trading
    places, which the isomorphism is free to do.
    """
    reference, relabelled = _relabelled_methanol()
    bonds = _bonds(reference)
    lengths = _reference_lengths(reference, bonds, [relabelled.get_positions()])
    assert lengths == pytest.approx(
        _reference_lengths(reference, bonds, [reference.get_positions()]), rel=1e-2
    )


def test_a_dft_minimum_that_is_not_the_molecule_raises() -> None:
    """A reference that contradicts itself stops the analysis."""
    reference, migrated = _migrated_methanol()
    with pytest.raises(ValueError, match="not the reference molecule"):
        _reference_lengths(reference, _bonds(reference), [migrated.get_positions()])


def test_every_metric_is_registered() -> None:
    """Every table row has a threshold, a tooltip and a weight."""
    assert set(DEFAULT_THRESHOLDS) == set(DEFAULT_TOOLTIPS) == set(DEFAULT_WEIGHTS)
    assert all(DEFAULT_TOOLTIPS.values())
    assert {
        "Extra minima",
        "Extra saddles",
        "Unsettled minima",
        "Saddles not first-order",
        "Distorted minima",
    } <= set(DEFAULT_THRESHOLDS)
    assert DEFAULT_WEIGHTS["Extra minima"] == 0.25
    assert DEFAULT_WEIGHTS["Distorted minima"] == 0.5
    assert DEFAULT_WEIGHTS["Extra saddles"] == 0


def _hydrogen_on(atoms: Atoms, heavy: int) -> int:
    """
    Find a hydrogen bonded to a given atom.

    Parameters
    ----------
    atoms
        Structure to search.
    heavy
        Index of the atom the hydrogen is bonded to.

    Returns
    -------
    int
        Index of the first such hydrogen.
    """
    symbols = atoms.get_chemical_symbols()
    return next(
        other
        for pair in sorted(_bonds(atoms))
        if heavy in pair
        for other in pair
        if other != heavy and symbols[other] == "H"
    )


def _relabelled_methanol() -> tuple[Atoms, Atoms]:
    """
    Build methanol and a copy with a methyl and the hydroxyl hydrogen exchanged.

    The copy is the same molecule in the same geometry, so a permutation-
    invariant match puts it on the reference at zero distance. Only the labels
    differ: index by index, the hydrogen that was on carbon is now on oxygen.

    Returns
    -------
    tuple[ase.atoms.Atoms, ase.atoms.Atoms]
        The reference and the relabelled copy.
    """
    reference = molecule("CH3OH")
    symbols = reference.get_chemical_symbols()
    on_carbon = _hydrogen_on(reference, symbols.index("C"))
    on_oxygen = _hydrogen_on(reference, symbols.index("O"))
    positions = reference.get_positions()
    positions[[on_carbon, on_oxygen]] = positions[[on_oxygen, on_carbon]]
    relabelled = reference.copy()
    relabelled.set_positions(positions)
    return reference, relabelled


def _migrated_methanol() -> tuple[Atoms, Atoms]:
    """
    Build methanol and a copy whose hydroxyl hydrogen has moved onto the carbon.

    Returns
    -------
    tuple[ase.atoms.Atoms, ase.atoms.Atoms]
        The reference and the migrated structure.
    """
    reference = molecule("CH3OH")
    symbols = reference.get_chemical_symbols()
    carbon, oxygen = symbols.index("C"), symbols.index("O")
    positions = reference.get_positions()
    # Straight away from the oxygen, which keeps it clear of the methyl hydrogens.
    away = positions[carbon] - positions[oxygen]
    positions[_hydrogen_on(reference, oxygen)] = positions[carbon] + 1.09 * (
        away / np.linalg.norm(away)
    )
    migrated = reference.copy()
    migrated.set_positions(positions)
    return reference, migrated


def test_a_relabelled_copy_looks_rebonded_index_by_index() -> None:
    """The case below is real: compared index by index, the copy is rebonded."""
    reference, relabelled = _relabelled_methanol()
    assert _bonds(relabelled) != _bonds(reference)


def test_a_relabelled_copy_is_the_molecule() -> None:
    """
    Exchanging equivalent atoms changes nothing about the molecule.

    The match is permutation-invariant, so the copy sits on the reference at
    zero distance, and excluding it discards a correct match. Three salicylic
    acid minima were lost that way, at 0.0065, 0.030 and 0.054 A.
    """
    reference, relabelled = _relabelled_methanol()
    bonds = _bonds(reference)
    status = _classify(
        relabelled,
        bonds,
        _reference_lengths(reference, bonds, [reference.get_positions()]),
    )
    assert status == GOOD


def test_a_migration_is_still_off_reference() -> None:
    """A migration is still flagged: the fix must not weaken the test."""
    reference, migrated = _migrated_methanol()
    bonds = _bonds(reference)
    status = _classify(
        migrated,
        bonds,
        _reference_lengths(reference, bonds, [reference.get_positions()]),
    )
    assert status == OFF_REFERENCE


def test_compare_matches_a_relabelled_copy() -> None:
    """The copy is matched, and counted as neither off-reference nor distorted."""
    reference, relabelled = _relabelled_methanol()
    relabelled.info.update(kind="minimum", label="min00")
    bonds = _bonds(reference)
    landscape = _Reference(
        symbols=reference.get_chemical_symbols(),
        minima=[reference.get_positions().flatten()],
        saddles=[],
        joins=[],
        bonds=bonds,
        lengths=_reference_lengths(reference, bonds, [reference.get_positions()]),
    )
    results, _ = _compare_molecule(landscape, [_Found(relabelled, 0, True)], [])
    assert results["off_reference_minima"] == 0
    assert results["distorted_minima"] == 0
    assert results["minima_matched"] == 1


def test_relabelling_restores_the_reference_bonding() -> None:
    """Put back in the reference's order, the copy is the reference again."""
    reference, relabelled = _relabelled_methanol()
    bonds = _bonds(reference)
    restored = _in_reference_labelling(relabelled, bonds)
    assert restored is not None
    assert _bonds(restored) == bonds
    assert not _is_distorted(
        restored, _reference_lengths(reference, bonds, [reference.get_positions()])
    )


def test_a_migration_has_no_reference_labelling() -> None:
    """No relabelling turns a migrated structure into the reference molecule."""
    reference, migrated = _migrated_methanol()
    assert _in_reference_labelling(migrated, _bonds(reference)) is None


def test_an_unchanged_structure_is_returned_as_it_is() -> None:
    """Identical bonding needs no isomorphism search, and gets none."""
    reference = molecule("CH3OH")
    assert _in_reference_labelling(reference, _bonds(reference)) is reference


def _scored_pair() -> tuple[list, list, list[str]]:
    """
    Build a reference and a slightly different structure to score against it.

    Returns
    -------
    tuple[list, list, list[str]]
        Reference positions, found positions and symbols, as ``_distances``
        takes them.
    """
    reference = molecule("CH3OH")
    other = reference.copy()
    other.rattle(stdev=0.05, seed=1)
    return (
        [reference.get_positions().flatten()],
        [other.get_positions().flatten()],
        reference.get_chemical_symbols(),
    )


def test_scoring_the_same_structures_twice_gives_identical_distances() -> None:
    """
    Distances are reproducible to the last bit.

    The alignment behind every distance starts from random rotations. Unseeded,
    two runs draw different ones, so a pair near the matching radius can land
    either side of it, and a published number cannot be reproduced.
    """
    pair = _scored_pair()
    assert np.array_equal(_distances(*pair), _distances(*pair))


@pytest.mark.parametrize("score", ["distances", "group"])
def test_scoring_leaves_the_global_generator_alone(score) -> None:
    """Fixing the alignment's seed must not disturb anything else drawing from NumPy."""
    reference, found, symbols = _scored_pair()
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    if score == "distances":
        _distances(reference, found, symbols)
    else:
        _group(reference + found, symbols, MINIMA_GROUP_RADIUS)
    assert np.random.random() == expected


def test_seeded_fixes_the_generator_and_restores_it() -> None:
    """Inside, draws follow the alignment seed; afterwards, the caller state is back."""
    np.random.seed(ALIGNMENT_SEED)
    seeded_draw = np.random.random()
    np.random.seed(99)
    expected_after = np.random.random()
    np.random.seed(99)
    with _seeded():
        assert np.random.random() == seeded_draw
    assert np.random.random() == expected_after


# --- the calculation: failure records, atomic writes, model checks ---------
# These drive the calc module's own pytest functions with a toy Lennard-Jones
# trimer standing in for the model and stubs standing in for topsearch, so
# nothing is downloaded and no model is loaded.

# Pair distance at the Lennard-Jones minimum, sigma = 1 Å.
LJ_PAIR = 2 ** (1 / 6)
# Spacing at which the linear trimer is stationary, V'(d) + V'(2d) = 0. It is
# a saddle of the trimer: bending the chain lowers the energy.
LJ_CHAIN = (2 * (1 + 2**-13) / (1 + 2**-7)) ** (1 / 6)
LJ_EPSILON = 10.0
# A structure placed beyond this, in Å along x, makes the toy model raise.
FAR = 40.0
TOY = "toy"
# Every Landscape17 molecule is a neutral closed-shell singlet, and a model
# that reads charge and spin must be told so.
NEUTRAL_SINGLET = (0, 1)
FOUND = calc_landscape_search.FOUND_SUFFIX
FAILED = calc_landscape_search.FAILED_SUFFIX
CHECKS = calc_landscape_search.CHECKS_SUFFIX


def _path(out_dir: Path, start: int, suffix: str) -> Path:
    """
    Get one of a toy start's output files, as the calculation names it.

    Parameters
    ----------
    out_dir
        The model's output directory.
    start
        Start index.
    suffix
        Which file: ``FOUND``, ``FAILED`` or ``CHECKS``.

    Returns
    -------
    pathlib.Path
        Path of that file.
    """
    return calc_landscape_search._unit_path(out_dir, TOY, start, suffix)


def _triangle() -> np.ndarray:
    """
    Get the equilateral trimer, the Lennard-Jones minimum.

    Returns
    -------
    numpy.ndarray
        Positions, 3 by 3, in Å.
    """
    return np.array(
        [[0.0, 0.0, 0.0], [LJ_PAIR, 0.0, 0.0], [LJ_PAIR / 2, LJ_PAIR * 3**0.5 / 2, 0]]
    )


def _chain() -> np.ndarray:
    """
    Get the linear trimer, a stationary point that is not a minimum.

    Returns
    -------
    numpy.ndarray
        Positions, 3 by 3, in Å.
    """
    return np.array([[-LJ_CHAIN, 0.0, 0.0], [0.0, 0.0, 0.0], [LJ_CHAIN, 0.0, 0.0]])


def _frame(positions: np.ndarray, kind: str, label: str) -> Atoms:
    """
    Build one structure as the search writes it.

    Parameters
    ----------
    positions
        Positions, 3 by 3, in Å.
    kind
        ``"minimum"`` or ``"transition_state"``.
    label
        Structure label.

    Returns
    -------
    ase.atoms.Atoms
        The trimer, carrying its kind and label.
    """
    atoms = Atoms("Ar3", positions=positions)
    atoms.info.update(molecule=TOY, kind=kind, label=label)
    return atoms


class _ToyCalculator(LennardJones):
    """
    The toy model: Lennard-Jones, but strict about what it is handed.

    It raises when the structure does not carry the neutral singlet's charge and
    spin, as a model that reads them would go wrong, and on a structure placed
    beyond ``FAR``, to stand in for a model failing on one structure. It counts
    the evaluations it is asked for, failed ones included.
    """

    calls = 0
    fails_far = True

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ) -> None:
        """
        Evaluate, or raise as described above.

        Parameters
        ----------
        atoms
            Structure to evaluate.
        properties
            Properties requested.
        system_changes
            What changed since the last evaluation.
        """
        self.calls += 1
        if atoms is not None:
            state = (atoms.info.get("charge"), atoms.info.get("spin"))
            if state != NEUTRAL_SINGLET:
                raise ValueError(f"handed charge and spin {state}")
            if (atoms.positions[:, 0].min() > FAR) == self.fails_far:
                raise RuntimeError("the model failed here")
        super().calculate(atoms, properties, system_changes)


class _FailsNearCalculator(_ToyCalculator):
    """The toy model the other way round: it fails only near the origin."""

    fails_far = False


class _BrokenCalculator(LennardJones):
    """A model that fails on every structure, as an out-of-memory GPU would."""

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ) -> None:
        """
        Raise.

        Parameters
        ----------
        atoms
            Structure to evaluate.
        properties
            Properties requested.
        system_changes
            What changed since the last evaluation.
        """
        raise RuntimeError("out of memory")


class _ToyModel:
    """
    Stand-in for an ml-peg model, handing out a toy calculator.

    Parameters
    ----------
    calculator
        Calculator to hand out.
    """

    def __init__(self, calculator: LennardJones) -> None:
        """
        Keep the calculator to hand out.

        Parameters
        ----------
        calculator
            Calculator to hand out.
        """
        self.calculator = calculator
        self.precisions: list[str] = []

    def get_calculator(self, precision: str) -> LennardJones:
        """
        Hand out the calculator, recording the precision asked for.

        Parameters
        ----------
        precision
            Precision requested.

        Returns
        -------
        ase.calculators.lj.LennardJones
            The toy calculator.
        """
        self.precisions.append(precision)
        return self.calculator


def _toy_model() -> _ToyModel:
    """
    Get the toy model.

    Returns
    -------
    _ToyModel
        The model.
    """
    return _ToyModel(_ToyCalculator(epsilon=LJ_EPSILON))


def _broken_model() -> _ToyModel:
    """
    Get a model that fails on everything.

    Returns
    -------
    _ToyModel
        The model.
    """
    return _ToyModel(_BrokenCalculator(epsilon=LJ_EPSILON))


@pytest.fixture
def out_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """
    Point the calculation at a toy molecule with five starts, in a scratch tree.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    tmp_path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        Where the toy model's outputs are written.
    """
    starts_dir = tmp_path / "starts"
    starts_dir.mkdir()
    starts = []
    for frame in range(calc_landscape_search.STARTS):
        atoms = Atoms("Ar3", positions=_triangle())
        atoms.rattle(stdev=0.01, seed=frame)
        atoms.info["frame"] = 10 * frame
        starts.append(atoms)
    write(starts_dir / f"{TOY}_starts.xyz", starts, format="extxyz")

    monkeypatch.setattr(calc_landscape_search, "OUT_PATH", tmp_path / "outputs")
    monkeypatch.setattr(calc_landscape_search, "_starts_dir", lambda: starts_dir)
    molecule_entry = calc_landscape_search._Molecule(TOY, TOY, f"{TOY}.xyz")
    monkeypatch.setattr(calc_landscape_search, "MOLECULES", (molecule_entry,))
    return tmp_path / "outputs" / TOY


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    found: list[Atoms] | None = None,
    search_error: Exception | None = None,
    frames_error: Exception | None = None,
) -> list[int]:
    """
    Stub the search and the conversion of its network into structures.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    found
        Structures every search finds. Defaults to one minimum.
    search_error
        Raised by the search instead, if given.
    frames_error
        Raised by the conversion instead, if given.

    Returns
    -------
    list[int]
        Seeds the search is called with, filled in as it runs.
    """
    seeds: list[int] = []
    frames = [_frame(_triangle(), "minimum", "min00")] if found is None else found

    def search(symbols, start, potential, seed) -> str:
        """
        Record the seed, then find the network or raise.

        Parameters
        ----------
        symbols
            Chemical symbols.
        start
            Starting coordinates.
        potential
            Potential to search on.
        seed
            Seed for this run.

        Returns
        -------
        str
            A stand-in for the network.
        """
        seeds.append(seed)
        if search_error is not None:
            raise search_error
        return "network"

    def convert(network, symbols, molecule, seed) -> list[Atoms]:
        """
        Convert the stand-in network, or raise.

        Parameters
        ----------
        network
            The stand-in network.
        symbols
            Chemical symbols.
        molecule
            Molecule name.
        seed
            Seed it came from.

        Returns
        -------
        list[ase.atoms.Atoms]
            Copies of the structures to find.
        """
        if frames_error is not None:
            raise frames_error
        return [atoms.copy() for atoms in frames]

    monkeypatch.setattr(calc_landscape_search, "_search", search)
    monkeypatch.setattr(calc_landscape_search, "_frames", convert)
    return seeds


def _failure(out_dir: Path, start: int) -> dict:
    """
    Read a start's failure record.

    Parameters
    ----------
    out_dir
        The model's output directory.
    start
        Start index.

    Returns
    -------
    dict
        The record.
    """
    return json.loads(_path(out_dir, start, FAILED).read_text())


def test_charge_and_spin_match_the_stationary_points_benchmark() -> None:
    """The Hessian comes from that module and is taken at its charge and spin."""
    from ml_peg.calcs.landscapes.stationary_points import calc_stationary_points

    here = (calc_landscape_search.CHARGE, calc_landscape_search.SPIN)
    assert here == (calc_stationary_points.CHARGE, calc_stationary_points.SPIN)
    assert here == NEUTRAL_SINGLET


def test_a_raising_search_writes_a_failure_record_and_no_found_file(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A search that raises is recorded as having run and failed, for every start.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    _stub(monkeypatch, search_error=RuntimeError("the search broke"))
    with pytest.warns(UserWarning, match="the search broke"):
        calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    for start, seed in enumerate(calc_landscape_search.SEED_OFFSETS):
        assert _failure(out_dir, start) == {
            "molecule": TOY,
            "start": start,
            "seed": seed,
            "error": "RuntimeError: the search broke",
        }
    assert not list(out_dir.glob(f"*{FOUND}"))


def test_a_raising_conversion_is_recorded_like_a_raising_search(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The record covers turning the network into structures, not only the search.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    _stub(monkeypatch, frames_error=KeyError("energy"))
    with pytest.warns(UserWarning):
        calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert _failure(out_dir, 0)["error"] == "KeyError: 'energy'"
    assert not list(out_dir.glob(f"*{FOUND}"))


def test_an_empty_search_is_a_failure(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A search that finds nothing at all ran and failed; it is not a missing run.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    _stub(monkeypatch, found=[])
    with pytest.warns(UserWarning, match="no structures"):
        calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    record = _failure(out_dir, 3)
    assert record["error"] == "the search returned no structures"
    assert record["seed"] == calc_landscape_search.SEED_OFFSETS[3]
    assert not list(out_dir.glob(f"*{FOUND}"))


def test_a_successful_search_writes_its_structures(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Every start's structures are written, tagged with the start they came from.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    seeds = _stub(monkeypatch)
    calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert seeds == list(calc_landscape_search.SEED_OFFSETS)
    for start in range(calc_landscape_search.STARTS):
        (written,) = read(_path(out_dir, start, FOUND), index=":")
        assert written.info["label"] == "min00"
        assert written.info["start"] == start
        assert written.info["start_frame"] == 10 * start
    assert not list(out_dir.glob(f"*{FAILED}"))
    assert not list(out_dir.glob("*.partial"))


def test_the_real_conversion_labels_structures_by_their_place_in_the_network(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Minima are named by node and saddles by the minima they join, in the file.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    from topsearch.data.kinetic_transition_network import KineticTransitionNetwork

    network = KineticTransitionNetwork()
    for shift, energy in ((0.0, -30.0), (5.0, -29.0), (10.0, -28.0)):
        network.add_minimum((_triangle() + shift).flatten(), energy)
    network.add_ts(_chain().flatten(), -20.0, 1, 2)
    monkeypatch.setattr(calc_landscape_search, "_search", lambda *args: network)
    calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    frames = read(_path(out_dir, 0, FOUND), index=":")
    assert [atoms.info["label"] for atoms in frames] == [
        "min00",
        "min01",
        "min02",
        "ts1_2_0",
    ]
    assert [atoms.info["kind"] for atoms in frames] == ["minimum"] * 3 + [
        "transition_state"
    ]
    assert [atoms.info["mlip_energy_ev"] for atoms in frames] == [
        -30.0,
        -29.0,
        -28.0,
        -20.0,
    ]
    assert frames[3].info["connects"].tolist() == [1, 2]
    assert frames[1].get_positions() == pytest.approx(_triangle() + 5.0)
    assert frames[3].get_positions() == pytest.approx(_chain())
    assert frames[3].info["seed"] == calc_landscape_search.SEED_OFFSETS[0]


def test_success_removes_a_stale_failure_record(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A start that failed before and succeeds now is no longer a failure.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    out_dir.mkdir(parents=True)
    stale = _path(out_dir, 2, FAILED)
    stale.write_text('{"error": "from an earlier run"}')
    _stub(monkeypatch)
    calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert _path(out_dir, 2, FOUND).exists()
    assert not stale.exists()


def test_a_finished_start_is_skipped_and_a_failed_one_retried(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The restart guard is a found file with something in it.

    A failure record does not stop a rerun, and nor does an empty found file,
    which is what a write cut short before the writes were atomic left behind.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    out_dir.mkdir(parents=True)
    finished = _path(out_dir, 0, FOUND)
    finished.write_text("kept as it is")
    _path(out_dir, 1, FAILED).write_text('{"error": "earlier"}')
    _path(out_dir, 2, FOUND).write_text("")
    seeds = _stub(monkeypatch)
    calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert seeds == list(calc_landscape_search.SEED_OFFSETS[1:])
    assert finished.read_text() == "kept as it is"
    assert _path(out_dir, 1, FOUND).exists()
    assert not _path(out_dir, 1, FAILED).exists()
    assert len(read(_path(out_dir, 2, FOUND), index=":")) == 1


def test_a_setup_failure_fails_every_unfinished_start(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    When the starts cannot be read, each start is recorded with that error.

    A finished start is left alone: its result stands. An empty found file is
    not a finished start.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    tmp_path
        Pytest temporary directory.
    """
    (tmp_path / "starts" / f"{TOY}_starts.xyz").unlink()
    out_dir.mkdir(parents=True)
    _path(out_dir, 4, FOUND).write_text("finished earlier")
    _path(out_dir, 3, FOUND).write_text("")
    seeds = _stub(monkeypatch)
    with pytest.warns(UserWarning, match="FileNotFoundError"):
        calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert seeds == []
    for start in range(4):
        record = _failure(out_dir, start)
        assert record["seed"] == calc_landscape_search.SEED_OFFSETS[start]
        assert record["error"].startswith("FileNotFoundError: ")
    assert not _path(out_dir, 4, FAILED).exists()
    # Left beside its failure record, the empty file would win and void the model.
    assert not _path(out_dir, 3, FOUND).exists()


def test_an_empty_found_file_does_not_outlive_a_failed_rerun(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The analysis lets a found file win, so an empty one must not stay to win.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    out_dir.mkdir(parents=True)
    _path(out_dir, 2, FOUND).write_text("")
    _stub(monkeypatch, search_error=RuntimeError("the search broke"))
    with pytest.warns(UserWarning, match="the search broke"):
        calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert not _path(out_dir, 2, FOUND).exists()
    assert _failure(out_dir, 2)["error"] == "RuntimeError: the search broke"


def test_a_crash_while_writing_leaves_nothing_at_the_final_path(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Half a file is never where the analysis looks, and is recorded as a failure.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    targets: list[Path] = []

    def crash(path, frames, **kwargs) -> None:
        """
        Write half a file, then fail.

        Parameters
        ----------
        path
            Where the structures are written.
        frames
            Structures to write.
        **kwargs
            Passed by the caller and ignored.
        """
        targets.append(Path(path))
        Path(path).write_text("half a frame")
        raise OSError("disk full")

    _stub(monkeypatch)
    monkeypatch.setattr(calc_landscape_search, "write", crash)
    with pytest.warns(UserWarning, match="disk full"):
        calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert not list(out_dir.glob(f"*{FOUND}"))
    assert not list(out_dir.glob("*.partial"))
    assert _failure(out_dir, 0)["error"] == "OSError: disk full"
    # Written beside the final file, under a name the analysis glob cannot match.
    assert all(target.parent == out_dir for target in targets)
    assert not any(fnmatch(target.name, f"*{FOUND}") for target in targets)


def test_a_crash_while_recording_a_failure_leaves_nothing_at_the_final_path(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Half a failure record would read as a record, so it must never land either.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    write_text = Path.write_text

    def crash(path: Path, data: str, *args, **kwargs) -> None:
        """
        Write the first few characters, then fail.

        Parameters
        ----------
        path
            Where the text is written.
        data
            Text to write.
        *args
            Passed by the caller and ignored.
        **kwargs
            Passed by the caller and ignored.
        """
        write_text(path, data[:5])
        raise OSError("disk full")

    _stub(monkeypatch, search_error=RuntimeError("the search broke"))
    monkeypatch.setattr(Path, "write_text", crash)
    with pytest.raises(OSError, match="disk full"), pytest.warns(UserWarning):
        calc_landscape_search.test_landscape_search((TOY, _toy_model()))

    assert not list(out_dir.glob(f"*{FAILED}"))
    assert not list(out_dir.glob("*.partial"))


def test_atomic_publishes_only_a_finished_file(tmp_path: Path) -> None:
    """
    The destination appears only once the block finishes, and never in part.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    """
    destination = calc_landscape_search._unit_path(tmp_path, "a", 0, FAILED)
    with pytest.raises(OSError, match="interrupted"):
        with calc_landscape_search._atomic(destination) as partial:
            partial.write_text("{")
            raise OSError("interrupted")
    assert list(tmp_path.iterdir()) == []

    with calc_landscape_search._atomic(destination) as partial:
        partial.write_text("{}")
        assert not destination.exists()
    assert destination.read_text() == "{}"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("suffix", [FOUND, FAILED, CHECKS])
def test_two_writers_never_share_a_partial_file(tmp_path: Path, suffix: str) -> None:
    """
    Two runs of one model writing the same file each get their own partial.

    Neither partial can be mistaken for an output by the analysis's globs.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    suffix
        Which output is being written.
    """
    destination = calc_landscape_search._unit_path(tmp_path, "a", 0, suffix)
    with (
        calc_landscape_search._atomic(destination) as first,
        calc_landscape_search._atomic(destination) as second,
    ):
        assert first != second
        for partial in (first, second):
            assert partial.parent == tmp_path
            assert not fnmatch(partial.name, f"*{FOUND}")
            assert not fnmatch(partial.name, f"*{FAILED}")
            assert not fnmatch(partial.name, f"*{CHECKS}")


def test_atomic_flushes_to_disk_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A power cut after the rename must not leave an empty file in its place.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """
    destination = calc_landscape_search._unit_path(tmp_path, "a", 0, FAILED)
    content = '{"start": 0}'
    synced: list[tuple[int, bool]] = []
    fsync = os.fsync

    def record(descriptor: int) -> None:
        """
        Note the file's size and whether it is published yet, then sync.

        Parameters
        ----------
        descriptor
            File descriptor being synced.
        """
        synced.append((os.fstat(descriptor).st_size, destination.exists()))
        fsync(descriptor)

    monkeypatch.setattr(calc_landscape_search.os, "fsync", record)
    with calc_landscape_search._atomic(destination) as partial:
        partial.write_text(content)
    assert synced == [(len(content), False)]
    assert destination.read_text() == content


def test_atomic_files_get_the_usual_permissions(tmp_path: Path) -> None:
    """
    A temporary file is private to its owner; the published one must not be.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    """
    destination = calc_landscape_search._unit_path(tmp_path, "a", 0, FAILED)
    with calc_landscape_search._atomic(destination) as partial:
        partial.write_text("{}")
    plain = tmp_path / "plain"
    plain.write_text("{}")
    assert destination.stat().st_mode == plain.stat().st_mode


def test_atomic_leaves_the_process_umask_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Reading the umask means setting it, for every thread, so it is read once.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """

    def refuse(mask: int) -> int:
        """
        Fail the test: nothing may set the umask while a file is written.

        Parameters
        ----------
        mask
            The mask that would be set.

        Returns
        -------
        int
            Never returns.
        """
        raise AssertionError(f"umask set to {mask:o} during a write")

    monkeypatch.setattr(calc_landscape_search.os, "umask", refuse)
    destination = calc_landscape_search._unit_path(tmp_path, "a", 0, FAILED)
    with calc_landscape_search._atomic(destination) as partial:
        partial.write_text("{}")
    assert destination.read_text() == "{}"


# --- the model check step ---------------------------------------------------


def _write_found_frames(out_dir: Path, start: int, frames: list[Atoms]) -> Path:
    """
    Write a found file as the search would have.

    Parameters
    ----------
    out_dir
        The model's output directory.
    start
        Start index.
    frames
        Structures found.

    Returns
    -------
    pathlib.Path
        The found file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    found = _path(out_dir, start, FOUND)
    write(found, frames, format="extxyz")
    return found


def _mixed_frames() -> list[Atoms]:
    """
    Build a found file's worth of structures, in no particular order.

    Returns
    -------
    list[ase.atoms.Atoms]
        A saddle-like point labelled a minimum, a transition state, the true
        minimum, and a minimum that is pushed off its stationary point.
    """
    pushed = _triangle()
    pushed[1, 0] += 0.05
    return [
        _frame(_chain(), "minimum", "min01"),
        _frame(_chain(), "transition_state", "ts0_1_0"),
        _frame(_triangle(), "minimum", "min00"),
        _frame(pushed, "minimum", "min02"),
    ]


def _checks(out_dir: Path, start: int) -> dict[str, np.ndarray]:
    """
    Read a start's checks, as the analysis would (no pickles).

    Parameters
    ----------
    out_dir
        The model's output directory.
    start
        Start index.

    Returns
    -------
    dict[str, numpy.ndarray]
        Every array in the file.
    """
    with np.load(_path(out_dir, start, CHECKS)) as data:
        return {key: data[key] for key in data.files}


def _max_force(positions: np.ndarray) -> float:
    """
    Get the toy model's largest per-atom force.

    Parameters
    ----------
    positions
        Positions, 3 by 3, in Å.

    Returns
    -------
    float
        Largest per-atom force norm, in eV/Å.
    """
    atoms = Atoms("Ar3", positions=positions)
    atoms.calc = LennardJones(epsilon=LJ_EPSILON)
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def test_checks_hold_exactly_the_agreed_arrays(out_dir: Path) -> None:
    """
    The analysis reads these names, so nothing more and nothing less.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    found = _write_found_frames(out_dir, 0, _mixed_frames())
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    checks = _checks(out_dir, 0)
    assert set(checks) == {
        "labels",
        "kinds",
        "max_force",
        "failed",
        "found_sha256",
        "hessian_min01",
        "hessian_ts0_1_0",
        "hessian_min00",
        "hessian_min02",
    }
    digest = checks["found_sha256"]
    assert digest.shape == ()
    assert digest.dtype.kind == "U"
    assert str(digest) == hashlib.sha256(found.read_bytes()).hexdigest()


def test_checks_list_every_structure_in_file_order(out_dir: Path) -> None:
    """
    Labels and kinds are read back in the order the found file holds them.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    _write_found_frames(out_dir, 0, _mixed_frames())
    model = _toy_model()
    calc_landscape_search.test_check_found_structures((TOY, model))

    checks = _checks(out_dir, 0)
    assert checks["labels"].tolist() == ["min01", "ts0_1_0", "min00", "min02"]
    assert checks["kinds"].tolist() == [
        "minimum",
        "transition_state",
        "minimum",
        "minimum",
    ]
    assert checks["labels"].dtype.kind == "U"
    assert checks["kinds"].dtype.kind == "U"
    assert checks["failed"].tolist() == [False] * 4
    assert checks["failed"].dtype == np.bool_
    assert model.precisions == ["high"]


def test_checks_record_the_largest_force_on_any_atom(out_dir: Path) -> None:
    """
    The force is the model's own, at the structure as found.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    found = _write_found_frames(out_dir, 0, _mixed_frames())
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    max_force = _checks(out_dir, 0)["max_force"]
    assert max_force.dtype == np.float64
    expected = [_max_force(atoms.get_positions()) for atoms in read(found, ":")]
    np.testing.assert_allclose(max_force, expected, rtol=1e-12, atol=1e-12)
    # The pushed minimum is the one with a force to measure.
    assert max_force[3] > 1.0


def test_checks_save_the_model_hessian_of_every_structure(out_dir: Path) -> None:
    """
    Minima and saddles alike, in eV/Å^2, so the analysis can judge both.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    from ml_peg.calcs.landscapes.stationary_points.calc_stationary_points import (
        _model_hessian,
    )

    found = _write_found_frames(out_dir, 0, _mixed_frames())
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    checks = _checks(out_dir, 0)
    for atoms in read(found, index=":"):
        hessian = checks[f"hessian_{atoms.info['label']}"]
        assert hessian.shape == (9, 9)
        assert hessian.dtype == np.float64
        expected = _model_hessian(
            LennardJones(epsilon=LJ_EPSILON), ["Ar"] * 3, atoms.get_positions()
        )
        np.testing.assert_allclose(hessian, expected, rtol=1e-12, atol=1e-12)


def _far_and_near_frames() -> list[Atoms]:
    """
    Build a found file with two structures the toy model fails on.

    Returns
    -------
    list[ase.atoms.Atoms]
        A minimum and a saddle placed beyond ``FAR``, then the same near the
        origin.
    """
    shift = np.array([FAR + 10.0, 0.0, 0.0])
    return [
        _frame(_triangle() + shift, "minimum", "min00"),
        _frame(_chain() + shift, "transition_state", "ts0_1_0"),
        _frame(_triangle(), "minimum", "min01"),
        _frame(_chain(), "transition_state", "ts0_1_1"),
    ]


def test_a_structure_the_model_fails_on_is_voided_alone(out_dir: Path) -> None:
    """
    NaN and a failed flag for that structure; every other one is still checked.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    _write_found_frames(out_dir, 0, _far_and_near_frames())
    with pytest.warns(UserWarning, match="the model failed here"):
        calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    checks = _checks(out_dir, 0)
    assert checks["failed"].tolist() == [True, True, False, False]
    assert np.isnan(checks["max_force"][:2]).all()
    assert np.isfinite(checks["max_force"][2:]).all()
    for label in ("min00", "ts0_1_0"):
        assert checks[f"hessian_{label}"].shape == (9, 9)
        assert np.isnan(checks[f"hessian_{label}"]).all()
    for label in ("min01", "ts0_1_1"):
        assert np.isfinite(checks[f"hessian_{label}"]).all()


def test_checks_stand_while_the_found_file_is_unchanged(out_dir: Path) -> None:
    """
    Unchanged content is not checked again, whatever its timestamps say.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    found = _write_found_frames(out_dir, 0, _mixed_frames())
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))
    checks = _path(out_dir, 0, CHECKS)
    written = checks.read_bytes()
    # Stamped, so that rewriting it, even unchanged, would show.
    os.utime(checks, (2e9, 2e9))

    # Re-synced copies of the same file, newer and older than the checks. A
    # rerun with this model would replace every number with NaN.
    for stamp in (4e9, 1e9):
        os.utime(found, (stamp, stamp))
        calc_landscape_search.test_check_found_structures((TOY, _broken_model()))
        assert checks.read_bytes() == written
        assert checks.stat().st_mtime == 2e9


def test_checks_are_redone_when_the_found_file_changes(out_dir: Path) -> None:
    """
    A replaced found file never inherits the checks of the one before.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    found = _write_found_frames(out_dir, 0, _mixed_frames())
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    # Older than the checks, so only the content can show it changed.
    replacement = [_frame(_triangle(), "minimum", "min00")]
    write(found, replacement, format="extxyz")
    os.utime(found, (1e9, 1e9))
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    redone = _checks(out_dir, 0)
    assert redone["labels"].tolist() == ["min00"]
    digest = hashlib.sha256(found.read_bytes()).hexdigest()
    assert str(redone["found_sha256"]) == digest


def test_checks_with_a_failure_are_redone(out_dir: Path) -> None:
    """
    A failure may be transient, an out-of-memory GPU say, so it is retried.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    _write_found_frames(out_dir, 0, _mixed_frames())
    with pytest.warns(UserWarning, match="out of memory"):
        calc_landscape_search.test_check_found_structures((TOY, _broken_model()))
    assert _checks(out_dir, 0)["failed"].all()

    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))
    retried = _checks(out_dir, 0)
    assert not retried["failed"].any()
    assert np.isfinite(retried["max_force"]).all()


def test_a_deterministic_failure_costs_only_its_own_structures(
    out_dir: Path,
) -> None:
    """
    Structures that were checked are not checked again for a neighbour's sake.

    A model that always fails on one structure would otherwise redo the whole
    file on every run, forever.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    _write_found_frames(out_dir, 0, _far_and_near_frames())
    with pytest.warns(UserWarning, match="the model failed here"):
        calc_landscape_search.test_check_found_structures((TOY, _toy_model()))
    first = _checks(out_dir, 0)

    model = _toy_model()
    with pytest.warns(UserWarning, match="the model failed here"):
        calc_landscape_search.test_check_found_structures((TOY, model))

    # One evaluation for each failing structure, which raises at once.
    assert model.calculator.calls == 2
    second = _checks(out_dir, 0)
    assert set(second) == set(first)
    for key in first:
        np.testing.assert_array_equal(second[key], first[key])


def test_only_the_structures_that_failed_are_retried(out_dir: Path) -> None:
    """
    A retry fills in what failed and keeps what had been checked.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    """
    _write_found_frames(out_dir, 0, _far_and_near_frames())
    with pytest.warns(UserWarning, match="the model failed here"):
        calc_landscape_search.test_check_found_structures((TOY, _toy_model()))
    first = _checks(out_dir, 0)

    # This model fails on exactly the structures that were checked, so
    # checking them again would void them.
    healed = _ToyModel(_FailsNearCalculator(epsilon=LJ_EPSILON))
    calc_landscape_search.test_check_found_structures((TOY, healed))

    second = _checks(out_dir, 0)
    assert second["failed"].tolist() == [False] * 4
    assert np.isfinite(second["max_force"]).all()
    np.testing.assert_array_equal(second["max_force"][2:], first["max_force"][2:])
    for label in ("min01", "ts0_1_1"):
        key = f"hessian_{label}"
        np.testing.assert_array_equal(second[key], first[key])
    for label in ("min00", "ts0_1_0"):
        assert np.isfinite(second[f"hessian_{label}"]).all()


@pytest.mark.parametrize("old", ["garbage", "no fingerprint"])
def test_unreadable_or_older_checks_are_redone(out_dir: Path, old: str) -> None:
    """
    Checks that cannot show which found file they belong to are not trusted.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    old
        What sits at the checks path beforehand.
    """
    _write_found_frames(out_dir, 0, _mixed_frames())
    checks = _path(out_dir, 0, CHECKS)
    if old == "garbage":
        checks.write_bytes(b"not an npz")
    else:
        np.savez(checks, labels=np.array(["min01"]), failed=np.array([False]))
    calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    assert _checks(out_dir, 0)["labels"].tolist() == [
        "min01",
        "ts0_1_0",
        "min00",
        "min02",
    ]


@pytest.mark.parametrize("content", ["3\nnot extxyz\nAr\n", ""])
def test_an_unreadable_found_file_is_skipped_with_a_warning(
    out_dir: Path, content: str
) -> None:
    """
    Nothing is written for it, and the other starts are still checked.

    An empty file counts as unreadable: there is nothing in it to check.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    content
        What the unreadable found file holds.
    """
    _write_found_frames(out_dir, 0, _mixed_frames())
    _path(out_dir, 1, FOUND).write_text(content)
    with pytest.warns(UserWarning, match="start1"):
        calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    assert _path(out_dir, 0, CHECKS).exists()
    assert not _path(out_dir, 1, CHECKS).exists()


def test_a_crash_while_saving_checks_leaves_nothing_at_the_final_path(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A half-written checks file must never land where the analysis reads it.

    Parameters
    ----------
    out_dir
        The toy model's output directory.
    monkeypatch
        Pytest monkeypatch fixture.
    """

    def crash(file, **arrays) -> None:
        """
        Write a few bytes, then fail.

        Parameters
        ----------
        file
            Where the arrays are saved.
        **arrays
            Arrays to save.
        """
        file.write(b"PK")
        raise OSError("disk full")

    _write_found_frames(out_dir, 0, _mixed_frames())
    monkeypatch.setattr(calc_landscape_search.np, "savez", crash)
    with pytest.raises(OSError, match="disk full"):
        calc_landscape_search.test_check_found_structures((TOY, _toy_model()))

    assert not _path(out_dir, 0, CHECKS).exists()
    assert not list(out_dir.glob("*.partial"))


def test_the_check_step_runs_with_the_search_it_judges() -> None:
    """
    Parametrised and speed-marked as the search is.

    The analysis voids a model without checks, so a flag that runs the search
    but not the checks would run the benchmark for nothing.
    """
    marks = {
        mark.name: mark
        for mark in calc_landscape_search.test_check_found_structures.pytestmark
    }
    search_marks = {
        mark.name: mark
        for mark in calc_landscape_search.test_landscape_search.pytestmark
    }
    from ml_peg.app.utils.speed import SPEED_ORDER

    speeds = set(SPEED_ORDER)
    assert set(marks) & speeds == set(search_marks) & speeds == {"multi_day"}
    assert marks["parametrize"].args == search_marks["parametrize"].args
