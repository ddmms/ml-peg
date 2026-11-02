"""Unit tests for the landscape search distorted-minima metric."""

from __future__ import annotations

from ase import Atoms
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.landscapes.landscape_search import analyse_landscape_search
from ml_peg.analysis.landscapes.landscape_search.analyse_landscape_search import (
    DEFAULT_THRESHOLDS,
    DEFAULT_TOOLTIPS,
    DISTORTED_SUBDIR,
    NEAR_MISS_FACTOR,
    SAME_STRUCTURE,
    _assign,
    _bond_deviations,
    _bond_lengths,
    _bonds,
    _is_distorted,
    _near_missed,
    _nearest,
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
