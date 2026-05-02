"""Tests for fast extraction of metadata from Extended XYZ files."""

from __future__ import annotations

from ase import Atoms
from ase.io import write

from ml_peg.analysis.utils.utils import read_extxyz_info_fast


def test_read_extxyz_info_fast(tmp_path):
    """Test that read_extxyz_info_fast correctly extracts info dict from file."""
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)])
    atoms.info["model_energy"] = -1.23
    atoms.info["system"] = "H2"

    xyz_path = tmp_path / "test.xyz"
    write(xyz_path, atoms)

    info = read_extxyz_info_fast(xyz_path)
    assert info["model_energy"] == -1.23
    assert info["system"] == "H2"
