"""Tests for the ACWF equation-of-state benchmark."""

from __future__ import annotations

from ase.build import bulk
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.bulk_crystal.equation_of_state_acwf.acwf_utils import (
    birch_murnaghan,
    delta,
    epsilon,
    fit_birch_murnaghan,
    nu,
)
from ml_peg.analysis.bulk_crystal.equation_of_state_acwf.analyse_equation_of_state_acwf import (  # noqa: E501
    compare_curve,
    is_flagged,
    unphysical_share,
)
from ml_peg.calcs.bulk_crystal.equation_of_state_acwf.calc_equation_of_state_acwf import (  # noqa: E501
    scaled_structures,
)

# Tested and reference (V0, B0, B1), in Å³/atom and eV/Å³.
TESTED = (20.4, 0.62, 4.9)
REFERENCE = (20.0, 0.60, 4.6)


def test_fit_recovers_parameters():
    """The fit recovers the parameters of an exact Birch-Murnaghan curve."""
    volumes = REFERENCE[0] * np.linspace(0.94, 1.06, 7)
    energies = birch_murnaghan(volumes, -3.5, *REFERENCE)

    fit = fit_birch_murnaghan(volumes, energies)

    assert fit["E0"] == pytest.approx(-3.5, abs=1e-10)
    assert fit["V0"] == pytest.approx(REFERENCE[0], rel=1e-10)
    assert fit["B0"] == pytest.approx(REFERENCE[1], rel=1e-8)
    assert fit["B1"] == pytest.approx(REFERENCE[2], rel=1e-6)
    assert fit["residuals"] == pytest.approx(0.0, abs=1e-12)


def test_fit_without_minimum():
    """A monotonic curve or too few points cannot be fitted."""
    volumes = np.linspace(10.0, 12.0, 7)
    assert fit_birch_murnaghan(volumes, -volumes) is None
    assert fit_birch_murnaghan(volumes[:3], (volumes[:3] - 11) ** 2) is None


def test_measures_vanish_for_identical_curves():
    """ε, ν and Δ are zero for identical curves, up to rounding in ε and Δ."""
    assert nu(*REFERENCE, *REFERENCE) == 0.0
    assert delta(*REFERENCE, *REFERENCE) < 1e-3
    assert epsilon(*REFERENCE, *REFERENCE) < 2e-3


def test_measures_match_published_implementation():
    """
    The measures match the ACWF implementation.

    Expected values were computed with ``acwf_paper_plots/quantities_for_comparison``
    of aiidateam/acwf-verification-scripts at commit 8eb1e9a.
    """
    assert delta(*TESTED, *REFERENCE) == pytest.approx(8.702788891366652, rel=1e-6)
    assert epsilon(*TESTED, *REFERENCE) == pytest.approx(1.0913340016917514, rel=1e-9)
    assert nu(*TESTED, *REFERENCE) == pytest.approx(1.9870349773494371, rel=1e-12)


def test_nu_volume_only():
    """With equal B0 and B1, ν is 100 times the relative difference in V0."""
    assert nu(20.2, 0.6, 4.6, 20.0, 0.6, 4.6) == pytest.approx(100 * 0.2 / 20.1)


def test_scaled_structures():
    """Structures are scaled in volume from 0.94 to 1.06, keeping the cell shape."""
    atoms = bulk("Fe", "bcc", a=2.83)

    structures = scaled_structures(atoms, "Fe-BCC")

    ratios = [struct.get_volume() / atoms.get_volume() for struct in structures]
    assert ratios == pytest.approx([0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06])
    assert all(struct.info["charge"] == 0 for struct in structures)
    assert all(struct.info["spin"] == 1 for struct in structures)
    assert all(struct.info["label"] == "Fe-BCC" for struct in structures)
    assert structures[0].cell.angles() == pytest.approx(atoms.cell.angles())


@pytest.mark.parametrize(
    ("energies", "v0_shift", "status"),
    [
        ("bm", 1.0, "ok"),
        ("bm", 1.1, "v0_outside"),
        ("linear", 1.0, "no_minimum"),
        ("nan", 1.0, "failed"),
    ],
)
def test_compare_curve_status(energies: str, v0_shift: float, status: str):
    """
    Curves are classified by whether a usable minimum is found.

    Parameters
    ----------
    energies
        Kind of E(V) curve to build.
    v0_shift
        Ratio of the tested V0 to the central volume.
    status
        Expected status.
    """
    reference = dict(zip(("V0", "B0", "B1"), REFERENCE, strict=True))
    volumes = REFERENCE[0] * np.linspace(0.94, 1.06, 7)
    curve = {
        "bm": birch_murnaghan(volumes, 0.0, REFERENCE[0] * v0_shift, *REFERENCE[1:]),
        "linear": -volumes,
        "nan": np.full(7, np.nan),
    }[energies]

    result = compare_curve(volumes, curve, reference)

    assert result["status"] == status
    assert (result["ε"] is None) == (status in {"no_minimum", "failed"})
    if status == "ok":
        assert result["ε"] < 2e-3
        assert result["ν"] == pytest.approx(0.0, abs=1e-6)


def test_label_survives_extxyz(tmp_path):
    """
    Labels are read back unchanged, including for fluorine.

    Parameters
    ----------
    tmp_path
        Temporary directory.
    """
    structures = scaled_structures(bulk("F", "sc", a=2.0), "F-SC")
    write(tmp_path / "eos.extxyz", structures)

    assert {atoms.info["label"] for atoms in read(tmp_path / "eos.extxyz", ":")} == {
        "F-SC"
    }


def test_flagged_structures():
    """Magnetic elements and +U oxides are flagged."""
    assert is_flagged("Fe", "BCC")
    assert is_flagged("Gd", "X2O3")
    assert is_flagged("V", "XO2")
    assert not is_flagged("V", "BCC")
    assert not is_flagged("Si", "Diamond")


def test_unphysical_share():
    """Only failed calculations and curves without a minimum are unphysical."""
    statuses = ["ok", "v0_outside", "no_minimum", "failed"]
    comparison = {i: {"status": status} for i, status in enumerate(statuses)}

    assert unphysical_share(comparison) == 50.0
    assert unphysical_share(None) is None
