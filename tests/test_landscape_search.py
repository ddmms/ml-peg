"""Unit tests for the landscape search distorted-minima metric."""

from __future__ import annotations

from ase import Atoms
from ase.build import molecule
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.landscapes.landscape_search import analyse_landscape_search
from ml_peg.analysis.landscapes.landscape_search.analyse_landscape_search import (
    ALIGNMENT_SEED,
    DEFAULT_THRESHOLDS,
    DEFAULT_TOOLTIPS,
    DISTORTED_SUBDIR,
    NEAR_MISS_FACTOR,
    SAME_STRUCTURE,
    UNMATCHABLE,
    _assign,
    _bond_deviations,
    _bond_lengths,
    _bonds,
    _distances,
    _in_reference_labelling,
    _is_distorted,
    _merge,
    _near_missed,
    _nearest,
    _seeded,
    _unphysical,
    _write_distorted,
)

# Paracetamol's phenol fragment, the three atoms MACE-OFF-23-L gets wrong.
# Real numbers throughout: the DFT reference has C7-O8 at 1.2795 Å and O8-H17
# at 0.937, the model's good minima put C7-O8 at 1.3635-1.3690, and the minima
# it converges with the bond crushed put it at 0.8572-0.8605.
REFERENCE_CO = 1.2795
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
    return _bond_lengths(reference, _bonds(reference))


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
    The model's good minima relax C-O by 7%, which must pass.

    Parameters
    ----------
    c_o
        Length of the C-O bond in one of those minima, in Å.
    """
    assert not _is_distorted(_fragment(c_o), reference_lengths)


@pytest.mark.parametrize("c_o", CRUSHED_CO)
def test_crushed_minima_are_flagged(c_o, reference_lengths) -> None:
    """
    The model's crushed minima compress C-O by 33%, which must fail.

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
    lengths = _bond_lengths(reference, _bonds(reference))
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


def test_the_metric_is_registered() -> None:
    """Without a metrics.yml entry the column has no threshold and no tooltip."""
    assert "Distorted minima" in DEFAULT_THRESHOLDS
    assert "Distorted minima" in DEFAULT_TOOLTIPS
    assert DEFAULT_TOOLTIPS["Distorted minima"]


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
    just_outside = SAME_STRUCTURE + 0.008
    matrix = np.array([[just_outside]])
    assert _assign(matrix) == {}
    assert _nearest(matrix) == pytest.approx([just_outside])


def test_nearest_is_not_constrained_by_the_assignment() -> None:
    """Two references may name the same nearest structure.

    ``_assign`` is one-to-one and must give that structure to one of them,
    structure and the other is dropped. The
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
    inside = SAME_STRUCTURE - 0.01
    band = SAME_STRUCTURE + 0.01
    far = NEAR_MISS_FACTOR * SAME_STRUCTURE + 0.01
    assert _near_missed([inside, band, far]) == 1


def test_near_miss_boundaries_are_half_open() -> None:
    """Exactly at the radius is a near miss; exactly at the outer edge is not."""
    assert _near_missed([SAME_STRUCTURE]) == 1
    assert _near_missed([NEAR_MISS_FACTOR * SAME_STRUCTURE]) == 0


def test_near_miss_ignores_references_with_nothing_found() -> None:
    """An infinite distance is not a near miss."""
    assert _near_missed([float("inf"), float("inf")]) == 0


def test_the_paracetamol_near_misses_are_counted() -> None:
    """The case this measure exists for.

    mace-mp-0a put two of paracetamol's four reference minima at 0.308 and
    0.352 Å and matched none of the four, which reads as total failure.
    """
    assert _near_missed([0.308, 0.352, 0.586, 0.571]) == 2


def test_the_new_metrics_are_registered() -> None:
    """Every new table row needs a threshold and a tooltip."""
    for name in (
        "Minima near-missed",
        "Nearest minima RMSD (median)",
        "Nearest saddle RMSD (median)",
    ):
        assert name in DEFAULT_THRESHOLDS
        assert name in DEFAULT_TOOLTIPS


# --- excluding unphysical structures from the network ---------------------
# A structure the model cannot physically hold is not a node of the landscape.
# It is counted and written out, then kept out of matching and connectivity.


def test_an_excluded_column_can_never_be_matched() -> None:
    """Pricing a column at UNMATCHABLE keeps it out of the assignment.

    The exclusion works by column rather than by removing list entries,
    because the merge indices that connectivity follows would otherwise shift.
    """
    import numpy as np

    matrix = np.array([[0.05, 0.9], [0.9, 0.06]])
    assert len(_assign(matrix)) == 2
    matrix[:, 0] = np.inf
    priced = np.where(np.isinf(matrix), UNMATCHABLE, matrix)
    assigned = _assign(priced)
    assert 0 not in {found for found, _ in assigned.values()}
    assert len(assigned) == 1


def test_excluding_leaves_the_other_indices_in_place() -> None:
    """Column 1 stays column 1 after column 0 is excluded.

    This is the property that makes connectivity survive the exclusion: a
    saddle's endpoint is followed through the merge by index.
    """
    import numpy as np

    matrix = np.array([[0.9, 0.05], [0.06, 0.9]])
    matrix[:, 0] = np.inf
    assigned = _assign(np.where(np.isinf(matrix), UNMATCHABLE, matrix))
    assert assigned == {0: (1, pytest.approx(0.05))}


def test_nearest_ignores_excluded_structures() -> None:
    """Nearest reports the closest structure the model may actually hold."""
    import numpy as np

    matrix = np.array([[0.05, 0.40]])
    matrix[:, 0] = np.inf
    assert _nearest(matrix) == pytest.approx([0.40])


def test_unphysical_flags_a_rebonded_structure(reference_lengths) -> None:
    """Bonding that differs from the reference is enough on its own."""
    good = _fragment(REFERENCE_CO)
    rebonded = _fragment(CRUSHED_CO[0])
    reference_bonds = _bonds(good)
    assert _unphysical([good], reference_bonds, reference_lengths) == []
    assert _unphysical([good, rebonded], reference_bonds, reference_lengths) == [1]


def test_unphysical_flags_a_distorted_structure(reference_lengths) -> None:
    """So is a reference bond held at an impossible length."""
    good = _fragment(REFERENCE_CO)
    reference_bonds = _bonds(good)
    stretched = _fragment(REFERENCE_CO * 1.5)
    flagged = _unphysical([good, stretched], reference_bonds, reference_lengths)
    assert flagged == [1]


def test_unphysical_returns_ascending_indices(reference_lengths) -> None:
    """The indices address columns of the distance matrix, so order matters."""
    good = _fragment(REFERENCE_CO)
    bad = _fragment(CRUSHED_CO[0])
    reference_bonds = _bonds(good)
    structures = [bad, good, bad, good, bad]
    flagged = _unphysical(structures, reference_bonds, reference_lengths)
    assert flagged == sorted(flagged)
    assert flagged == [0, 2, 4]


def test_compare_keeps_unphysical_structures_out_of_the_match(monkeypatch) -> None:
    """The exclusion is actually wired into the comparison.

    Guards the integration rather than the mechanism: each piece can be
    correct while nothing applies them, which unit tests on the matrix
    helpers cannot see. The structure here is chosen to sit 0.147 A from the
    reference -- inside the 0.3 A matching radius, so it matches unless it is
    excluded -- while holding its C-O bond 25% long, so it is flagged.
    """
    results = _compare_one(
        monkeypatch, _fragment(REFERENCE_CO), [_fragment(REFERENCE_CO * 1.25)]
    )
    assert results["excluded_minima"] == 1
    # Without the exclusion this would be 1: the structure is within the radius.
    assert results["minima_matched"] == 0


def _compare_one(monkeypatch, reference: Atoms, found: list[Atoms]) -> dict:
    """
    Run the comparison against one reference minimum and a given search result.

    Parameters
    ----------
    monkeypatch
        Pytest's monkeypatch fixture.
    reference
        The only reference minimum, and the structure reference bonding is read from.
    found
        The minima the search is to have found. It found no transition states.

    Returns
    -------
    dict
        The comparison's results for the one molecule.
    """
    symbols = reference.get_chemical_symbols()
    entry = analyse_landscape_search.MOLECULES[0]
    monkeypatch.setattr(analyse_landscape_search, "MOLECULES", (entry,))
    monkeypatch.setattr(
        analyse_landscape_search,
        "_reference",
        lambda _m: (symbols, [reference.get_positions().flatten()], [], []),
    )
    monkeypatch.setattr(
        analyse_landscape_search,
        "_found",
        lambda _m, _k, kind: found if kind == "minimum" else [],
    )
    monkeypatch.setattr(
        analyse_landscape_search, "_merge", lambda frames, _s: (frames, {})
    )
    monkeypatch.setattr(analyse_landscape_search, "_searches", lambda _m, _k: 1)
    monkeypatch.setattr(
        analyse_landscape_search, "_write_distorted", lambda *a, **k: None
    )
    monkeypatch.setattr(
        analyse_landscape_search, "read", lambda *a, **k: reference.copy()
    )
    return analyse_landscape_search._compare("any-model")[entry.key]


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


def test_a_relabelled_copy_is_not_unphysical() -> None:
    """
    Exchanging equivalent atoms changes nothing about the molecule.

    The match is permutation-invariant, so the copy sits on the reference at
    zero distance, and excluding it discards a correct match. Three salicylic
    acid minima were lost that way, at 0.0065, 0.030 and 0.054 A.
    """
    reference, relabelled = _relabelled_methanol()
    bonds = _bonds(reference)
    assert _unphysical([relabelled], bonds, _bond_lengths(reference, bonds)) == []


def test_a_migration_is_still_unphysical() -> None:
    """A migration is still flagged: the fix must not weaken the test."""
    reference, migrated = _migrated_methanol()
    bonds = _bonds(reference)
    assert _unphysical([migrated], bonds, _bond_lengths(reference, bonds)) == [0]


def test_compare_matches_a_relabelled_copy(monkeypatch) -> None:
    """The copy is matched, and counted as neither off-reference nor distorted."""
    reference, relabelled = _relabelled_methanol()
    results = _compare_one(monkeypatch, reference, [relabelled])
    assert results["excluded_minima"] == 0
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
    assert not _is_distorted(restored, _bond_lengths(reference, bonds))


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


@pytest.mark.parametrize("score", ["distances", "merge"])
def test_scoring_leaves_the_global_generator_alone(score) -> None:
    """Fixing the alignment's seed must not disturb anything else drawing from NumPy."""
    reference, found, symbols = _scored_pair()
    frames = [Atoms(symbols, positions=p.reshape(-1, 3)) for p in reference + found]
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    if score == "distances":
        _distances(reference, found, symbols)
    else:
        _merge(frames, symbols)
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
