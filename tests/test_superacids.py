"""Unit tests for superacids benchmark utilities."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import Trajectory
import numpy as np
import pytest

from ml_peg.analysis.superacids.HF_SbF5_density.analyse_HF_SbF5_density import (
    compute_density,
)
from ml_peg.analysis.superacids.HF_structure.analyse_HF_structure import (
    compute_r_factor,
)
from ml_peg.calcs.superacids.HF_SbF5_density.calc_HF_SbF5_density import (
    OUT_FREQ,
    read_restart,
)


def write_trajectory(traj_path: Path, separations: tuple[float, ...]) -> None:
    """
    Write a trajectory of two atoms with a total mass of 100 amu.

    Parameters
    ----------
    traj_path
        Path to write the trajectory to.
    separations
        Separation between the two atoms in each frame, in Angstrom.
    """
    with Trajectory(str(traj_path), "w") as traj:
        for separation in separations:
            atoms = Atoms(
                "H2",
                positions=[[0, 0, 0], [separation, 0, 0]],
                cell=[10, 10, 10],
                pbc=True,
            )
            atoms.set_masses([50.0, 50.0])
            traj.write(atoms)


def test_read_restart_without_trajectory(tmp_path: Path) -> None:
    """Test nothing is restarted from if the trajectory does not exist."""
    atoms, nsteps = read_restart(tmp_path / "does_not_exist.traj")

    assert atoms is None
    assert nsteps == 0


def test_read_restart_returns_last_frame(tmp_path: Path) -> None:
    """Test the last frame of the trajectory, and its step, are returned."""
    traj_path = tmp_path / "X_0.traj"
    write_trajectory(traj_path, (1.0, 2.0, 3.0))

    atoms, nsteps = read_restart(traj_path)

    # Frames are written at steps 0, OUT_FREQ and 2 * OUT_FREQ
    assert nsteps == 2 * OUT_FREQ
    assert atoms.get_distance(0, 1) == pytest.approx(3.0)


def test_read_restart_ignores_corrupt_trajectory(tmp_path: Path) -> None:
    """Test an unreadable trajectory is warned about, rather than raising."""
    traj_path = tmp_path / "X_0.traj"
    traj_path.write_bytes(b"not a trajectory")

    with pytest.warns(UserWarning, match="unreadable trajectory"):
        atoms, nsteps = read_restart(traj_path)

    assert atoms is None
    assert nsteps == 0


def test_compute_density(tmp_path: Path) -> None:
    """Test density is computed from the production half of the volumes."""
    traj_path = tmp_path / "X_0.traj"
    write_trajectory(traj_path, (1.0,))

    # Equilibration at 1000 A^3, production at 200 A^3
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        "# step  volume_A3\n0  1000\n200  1000\n400  200\n600  200\n"
    )

    # 100 amu * 1.66053906660e-24 g/amu / (200 A^3 * 1e-24 cm^3/A^3)
    assert compute_density(traj_path, volume_path) == pytest.approx(0.830269533)


def test_compute_density_too_few_samples(tmp_path: Path) -> None:
    """Test NaN is returned if the production window is too short."""
    traj_path = tmp_path / "X_0.traj"
    write_trajectory(traj_path, (1.0,))

    # Half of two samples is one sample, fewer than MIN_SAMPLES
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text("# step  volume_A3\n0  1000\n200  200\n")

    assert np.isnan(compute_density(traj_path, volume_path))


def test_compute_r_factor_identical_curves() -> None:
    """Test the R-factor of a structure factor against itself is zero."""
    q = np.arange(1.0, 4.01, 0.05)
    sq = np.exp(-q)

    assert compute_r_factor(q, sq, q, sq) == pytest.approx(0.0)


def test_compute_r_factor_constant_offset() -> None:
    """Test the R-factor of a structure factor offset by a known amount."""
    q = np.array([1.0, 2.0, 3.0, 4.0])
    sq_ref = np.ones(4)
    sq_calc = np.full(4, 1.5)

    # sum|1 - 1.5| = 2, sum|1| = 4
    assert compute_r_factor(q, sq_ref, q, sq_calc) == pytest.approx(0.5)


def test_compute_r_factor_without_overlap() -> None:
    """Test NaN is returned if the two structure factors do not overlap."""
    q_ref = np.array([1.0, 2.0, 3.0, 4.0])
    q_calc = np.array([10.0, 11.0, 12.0])

    assert np.isnan(compute_r_factor(q_ref, np.ones(4), q_calc, np.ones(3)))
