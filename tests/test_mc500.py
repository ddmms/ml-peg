"""Tests for the MC500 molecular-crystal benchmark."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.molecular_crystal.MC500.analyse_MC500 import (
    rms_cartesian_displacement,
)
from ml_peg.calcs.molecular_crystal.MC500 import cif_utils


def test_rmscd_identical_structures() -> None:
    """Identical structures have zero RMSCD."""
    atoms = Atoms(
        "CH",
        scaled_positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        cell=[5.0, 6.0, 7.0],
    )
    atoms.pbc = True

    rmscd, rmscd_no_h, displacements = rms_cartesian_displacement(atoms, atoms)

    assert rmscd == pytest.approx(0.0)
    assert rmscd_no_h == pytest.approx(0.0)
    assert displacements == pytest.approx([0.0, 0.0])


def test_rmscd_averages_reference_and_relaxed_cells() -> None:
    """The Cartesian displacement is averaged over the two cells."""
    reference = Atoms("C", scaled_positions=[[0.1, 0.0, 0.0]], cell=[2.0] * 3)
    relaxed = Atoms("C", scaled_positions=[[0.2, 0.0, 0.0]], cell=[4.0] * 3)
    reference.pbc = True
    relaxed.pbc = True

    rmscd, rmscd_no_h, displacements = rms_cartesian_displacement(reference, relaxed)

    assert rmscd == pytest.approx(0.3)
    assert rmscd_no_h == pytest.approx(0.3)
    assert displacements == pytest.approx([0.3])


def test_rmscd_rejects_different_atom_order() -> None:
    """RMSCD requires matching atoms in matching order."""
    reference = Atoms("CH", positions=np.zeros((2, 3)), cell=[5.0] * 3)
    relaxed = Atoms("HC", positions=np.zeros((2, 3)), cell=[5.0] * 3)

    with pytest.raises(ValueError, match="different atom order"):
        rms_cartesian_displacement(reference, relaxed)


def test_read_mc500_cif_matches_original_reader(monkeypatch) -> None:
    """The default MC500 reader uses ASE and retains the original CIF labels."""
    atoms = Atoms(
        "CC",
        scaled_positions=[[0.0, 0.2, 0.3], [0.5, 0.6, 0.7]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
        info={
            "_atom_site_label": ["C1"],
            "_atom_site_occupancy": [1.0],
        },
    )
    atoms.new_array("spacegroup_kinds", np.array([0, 0]))
    read_kwargs = {}

    def fake_read(filename, **kwargs):
        read_kwargs.update(kwargs)
        return atoms.copy()

    monkeypatch.setattr(cif_utils, "read", fake_read)
    result = cif_utils.read_mc500_cif("structure.cif")

    assert read_kwargs["reader"] == "ase"
    assert read_kwargs["store_tags"] is True
    assert result.arrays[cif_utils.CIF_LABEL_ARRAY].tolist() == ["C1", "C1"]
    assert not np.isclose(
        result.get_scaled_positions(wrap=False),
        0.0,
        atol=cif_utils.BOUNDARY_TOLERANCE,
        rtol=0.0,
    ).any()


def test_read_mc500_cif_pycodcif_is_retained(monkeypatch) -> None:
    """The alternative reader selects the optional pycodcif backend."""
    atoms = Atoms(
        "C",
        scaled_positions=[[0.1, 0.2, 0.3]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
        info={
            "_atom_site_label": ["C1"],
            "_atom_site_occupancy": [1.0],
        },
    )
    atoms.new_array("spacegroup_kinds", np.array([0]))
    read_kwargs = {}

    def fake_read(filename, **kwargs):
        read_kwargs.update(kwargs)
        return atoms.copy()

    monkeypatch.setattr(cif_utils, "read", fake_read)
    cif_utils.read_mc500_cif_pycodcif("structure.cif")

    assert read_kwargs["reader"] == "pycodcif"


def test_read_mc500_cif_accepts_implicit_full_occupancy(monkeypatch) -> None:
    """Missing occupancy tags use the CIF default of full occupancy."""
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=[5.0] * 3, pbc=True)
    atoms.info = {"_atom_site_label": ["C1"]}
    atoms.new_array("spacegroup_kinds", np.array([0]))
    monkeypatch.setattr(cif_utils, "read", lambda *args, **kwargs: atoms)

    result = cif_utils.read_mc500_cif("structure.cif")

    assert len(result) == 1


def test_read_mc500_cif_rejects_partial_occupancy(monkeypatch) -> None:
    """The MC500 reader rejects disordered sites with partial occupancy."""
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=[5.0] * 3, pbc=True)
    atoms.info = {
        "_atom_site_label": ["C1"],
        "_atom_site_occupancy": [0.5],
    }
    atoms.new_array("spacegroup_kinds", np.array([0]))
    monkeypatch.setattr(cif_utils, "read", lambda *args, **kwargs: atoms)

    with pytest.raises(ValueError, match="partial occupancies"):
        cif_utils.read_mc500_cif("structure.cif")
