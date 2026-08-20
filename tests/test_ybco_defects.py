"""Unit tests for the YBCO defect formation-energy benchmark helpers."""

from __future__ import annotations

from ase import Atoms
import pytest

from ml_peg.analysis.defect.YBCO_defects import analyse_ybco_defects as A  # noqa: N812
from ml_peg.models.mock import MockCalculator


def test_element() -> None:
    """Site map to the correct element symbol."""
    assert A._element("Cu1") == "Cu"
    assert A._element("O2") == "O"
    assert A._element("Ba") == "Ba"
    assert A._element("Y") == "Y"


def test_name_class() -> None:
    """Trajectory stems split into (defect name, class), incl. underscored antisites."""
    assert A._name_class("vac_O1-traj") == ("O1", "vacancy")
    assert A._name_class("anti_Ba_Cu1-traj") == ("Ba_Cu1", "antisite")
    assert A._name_class("int_Oint1-traj") == ("Oint1", "interstitial")


def test_reference_complete() -> None:
    """The reference set has the expected 8 vacancies, 8 antisites, 7 interstitials."""
    keys = set(A.REFERENCE_FE)
    assert len({k for k in keys if k.startswith("Oint")}) == 7
    # Oint4 collapses into the Oint2 configuration -> shares its formation energy.
    assert A.REFERENCE_FE["Oint4"] == A.REFERENCE_FE["Oint2"]
    antisites = {k for k in keys if "_" in k}
    assert len(antisites) == 8
    vacancies = keys - antisites - {k for k in keys if k.startswith("Oint")}
    assert len(vacancies) == 8
    # _element resolves vacancy and antisite site tokens to real elements.
    for name in vacancies:
        assert A._element(name) in {"O", "Cu", "Ba", "Y"}


def test_reactions_mu_independent() -> None:
    """Reactions pair an antisite with its reverse, so chemical potentials cancel."""
    assert len(A.REACTIONS) == len(A.REACTION_REF) == 5
    for _name, (pair, ref) in A.REACTIONS.items():
        a, b = pair
        # both members are antisites present in the reference set
        assert a in A.REFERENCE_FE and b in A.REFERENCE_FE
        # the swapped elements match (A_B paired with X_A so mu terms cancel)
        assert A._element(a.split("_")[1]) == A._element(b.split("_")[0])
        assert ref > 0


def test_class_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """RMSD is zero for exact predictions and equals a constant offset otherwise."""
    defects = ["O1", "O2", "Ba_Cu1", "Oint1"]
    classes = ["vacancy", "vacancy", "antisite", "interstitial"]
    monkeypatch.setattr(A, "MODELS", ["m_exact", "m_offset"])
    monkeypatch.setattr(A, "CLASSES", classes)
    monkeypatch.setattr(A, "DEFECTS", defects)
    n = len(defects)
    data = {"ref": [1.0] * n, "m_exact": [1.0] * n, "m_offset": [1.5] * n}
    errors = A._class_errors(data, None)
    assert errors["m_exact"] == pytest.approx(0.0)
    assert errors["m_offset"] == pytest.approx(0.5)
    # A constant offset gives the same RMSD when restricted to one class.
    assert A._class_errors(data, "vacancy")["m_offset"] == pytest.approx(0.5)


def test_mock_calculator_runs() -> None:
    """The mock calculator returns a finite (zero) energy for a YBCO-like cell."""
    atoms = Atoms(
        "YBaCuO",
        positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
        cell=[6, 6, 6],
        pbc=True,
    )
    atoms.calc = MockCalculator()
    assert atoms.get_potential_energy() == pytest.approx(0.0)
