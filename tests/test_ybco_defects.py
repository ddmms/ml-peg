"""Tests for the YBCO defect benchmark helper functions."""

from __future__ import annotations

from ml_peg.analysis.defect.YBCO_defects import analyse_ybco_defects as A  # noqa: N812


def test_element():
    """A site label with its number removed is just the element."""
    # "Cu1" is a copper site, so _element("Cu1") must give "Cu"
    assert A._element("Cu1") == "Cu"
    assert A._element("O2") == "O"
    assert A._element("Ba") == "Ba"


def test_name_class():
    """A defect file name splits into (defect, type)."""
    # "vac_O1" is the O1 vacancy, "anti_Ba_Cu1" is a Ba-on-Cu1 antisite, etc.
    assert A._name_class("vac_O1-traj") == ("O1", "vacancy")
    assert A._name_class("anti_Ba_Cu1-traj") == ("Ba_Cu1", "antisite")
    assert A._name_class("int_Oint1-traj") == ("Oint1", "interstitial")


def test_reactions():
    """There are exactly 5 antisite exchange reactions in the benchmark."""
    assert len(A.REACTIONS) == 5


def test_reference():
    """There are 23 defects, and Oint4 shares Oint2's reference energy."""
    # 8 vacancies + 8 antisites + 7 interstitials = 23; Oint4 relaxes onto Oint2
    assert len(A.REFERENCE_FE) == 23
    assert A.REFERENCE_FE["Oint4"] == A.REFERENCE_FE["Oint2"]
