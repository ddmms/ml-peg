"""Generated GPUMD inputs: run.in and model.xyz (written through calorine)."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from calorine.gpumd import read_runfile
import gpumd_io
import numpy as np
import pytest

DATA = Path(__file__).parent / "data"

RUN_IN_SNAPSHOT = """\
potential nep.txt
time_step 1.0
ensemble nve
dump_thermo 100
run 100
ensemble nve
dump_thermo 100
run 1000
"""

SINGLE_POINT_SNAPSHOT = """\
potential nep.txt
time_step 0
ensemble nve
dump_xyz 1 step0.xyz precision double force potential \n\
run 1
"""


def test_run_in_snapshot(tmp_path):
    """The timed run.in matches the snapshot."""
    path = tmp_path / "run.in"
    gpumd_io.write_run_in(path, gpumd_io.run_parameters(1.0, 100, 1000, 100))
    assert path.read_text() == RUN_IN_SNAPSHOT


def test_run_in_hydrogen_timestep(tmp_path):
    """Half-fs steps are written without rounding."""
    path = tmp_path / "run.in"
    gpumd_io.write_run_in(path, gpumd_io.run_parameters(0.5, 100, 1000, 100))
    assert ("time_step", 0.5) in read_runfile(str(path))


def test_run_in_has_no_forbidden_keywords():
    """Nothing rescales velocities, writes trajectories or computes extras."""
    keywords = {key for key, _ in gpumd_io.run_parameters(1.0, 100, 1000, 100)}
    assert keywords == {"potential", "time_step", "ensemble", "dump_thermo", "run"}


def test_single_point_snapshot(tmp_path):
    """The step-0 run.in matches the snapshot and the one used for the capture."""
    path = tmp_path / "run.in"
    gpumd_io.write_run_in(path, gpumd_io.single_point_parameters())
    assert path.read_text() == SINGLE_POINT_SNAPSHOT
    assert read_runfile(str(DATA / "step0_run.in")) == read_runfile(str(path))


@pytest.mark.parametrize(
    ("log", "warmup", "measured", "expected"),
    [(100, 100, 1000, 100), (100, 100, 250, 50), (100, 20, 20, 20)],
)
def test_thermo_interval(log, warmup, measured, expected):
    """The thermo interval lands a row on both run ends."""
    assert gpumd_io.thermo_interval(log, warmup, measured) == expected


@pytest.fixture
def two_atoms():
    """
    Build a two-atom triclinic cell with unit ASE velocities.

    Returns
    -------
    ase.Atoms
        The structure.
    """
    atoms = Atoms(
        "SiO",
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.0, 0.0, 6.0]],
        pbc=[True, True, False],
    )
    atoms.set_masses([28.0, 16.0])
    atoms.set_momenta([[28.0, 0.0, 0.0], [0.0, -16.0, 0.0]])  # v = 1 ASE unit
    return atoms


def test_model_xyz_content(two_atoms, tmp_path):
    """model.xyz carries cell, PBC, masses and Å/fs velocities GPUMD can read."""
    path = tmp_path / "model.xyz"
    gpumd_io.write_model_xyz(two_atoms, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "2"
    header = lines[1]
    assert 'Lattice="5.0 0.0 0.0 0.5 5.0 0.0 0.0 0.0 6.0"' in header
    assert 'pbc="T T F"' in header
    props = header.split("Properties=")[1].split()[0]
    assert props.startswith("species:S:1:pos:R:3:")
    assert ":mass:R:1" in props and ":vel:R:3" in props
    names = props.split(":")[::3]
    tokens = lines[2].split()
    assert tokens[0] == "Si"
    width = {"species": 1, "pos": 3, "masses": 1, "momenta": 3, "mass": 1, "vel": 3}
    offset = sum(width[n] for n in names[: names.index("vel")])
    # 1 ASE velocity unit = ase.units.fs A/fs = 0.0982269... A/fs
    assert float(tokens[offset]) == pytest.approx(0.09822695, abs=1e-8)
    assert float(lines[3].split()[offset + 1]) == pytest.approx(-0.09822695, abs=1e-8)
    assert len(lines) == 4


def test_model_xyz_drops_free_text_info(two_atoms, tmp_path):
    """Input info (e.g. a comment containing 'pbc=') never reaches GPUMD."""
    two_atoms.info["comment"] = "LAMMPS data, pbc=F F F properties=junk"
    path = tmp_path / "model.xyz"
    gpumd_io.write_model_xyz(two_atoms, path)
    header = path.read_text().splitlines()[1]
    assert "comment" not in header and "junk" not in header


def test_model_xyz_requires_momenta(tmp_path):
    """A structure without momenta is rejected rather than silently zeroed."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(ValueError, match="momenta"):
        gpumd_io.write_model_xyz(atoms, tmp_path / "model.xyz")


def test_model_xyz_rejects_zero_velocities(two_atoms, tmp_path):
    """Calorine drops all-zero velocities; that must not reach GPUMD."""
    two_atoms.set_momenta(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="re-thermalize"):
        gpumd_io.write_model_xyz(two_atoms, tmp_path / "model.xyz")


def test_nep_elements():
    """Species are read from the model with calorine."""
    assert gpumd_io.nep_elements(DATA / "nep89-H-O-Si.txt") == ["H", "O", "Si"]
