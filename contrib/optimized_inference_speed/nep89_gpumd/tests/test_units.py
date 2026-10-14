"""ASE momenta -> GPUMD velocities -> kinetic energy round trip."""

from __future__ import annotations

from ase import units
from calorine.gpumd import read_xyz
import gpumd_io
import numpy as np


def read_model_xyz_like_gpumd(path):
    """
    Parse ``model.xyz`` the way GPUMD's read_xyz.cu does for mass and vel.

    Columns are located from ``properties=`` by name, with offsets summed over
    all preceding columns, exactly as GPUMD does; unknown columns are skipped.

    Parameters
    ----------
    path
        Path to ``model.xyz``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Masses (amu) and velocities in GPUMD natural units.
    """
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0])
    header = lines[1].lower()
    props = header.split("properties=")[1].split()[0].split(":")
    offsets, col = {}, 0
    for name, _kind, width in zip(props[::3], props[1::3], props[2::3], strict=True):
        offsets[name] = col
        col += int(width)
    rows = [ln.split() for ln in lines[2 : 2 + n_atoms]]
    assert all(len(row) == col for row in rows)
    masses = np.array([float(row[offsets["mass"]]) for row in rows])
    vel = np.array(
        [[float(x) for x in row[offsets["vel"] : offsets["vel"] + 3]] for row in rows]
    )
    # read_xyz.cu: velocity in model.xyz is A/fs, times TIME_UNIT_CONVERSION
    return masses, vel * gpumd_io.GPUMD_TIME_UNIT_CONVERSION


def test_kinetic_energy_round_trip(sio2_atoms, tmp_path):
    """KE from ASE equals 1/2 m v^2 from the velocities GPUMD reads."""
    path = tmp_path / "model.xyz"
    gpumd_io.write_model_xyz(sio2_atoms, path)
    masses, velocities = read_model_xyz_like_gpumd(path)

    ke_ase = sio2_atoms.get_kinetic_energy()
    ke_gpumd = 0.5 * np.sum(masses[:, None] * velocities**2)
    # GPUMD's 10.18051 vs ASE's 1/fs (4e-7 relative) plus the 8-decimal
    # '%16.8f' extxyz format calorine writes (~1e-6 relative per component).
    assert np.isclose(ke_gpumd, ke_ase, rtol=1e-5, atol=0)
    # The temperature is what MaxwellBoltzmann at 300 K should give (768 atoms).
    temperature = 2 * ke_ase / (3 * len(sio2_atoms) * units.kB)
    assert 250 < temperature < 350


def test_calorine_velocity_round_trip(sio2_atoms, tmp_path):
    """Calorine reads back the velocities it wrote, in ASE units."""
    path = tmp_path / "model.xyz"
    gpumd_io.write_model_xyz(sio2_atoms, path)
    back = read_xyz(str(path))
    # 8 decimals in A/fs = 5e-9 A/fs = 5e-8 ASE velocity units.
    np.testing.assert_allclose(
        back.get_velocities(), sio2_atoms.get_velocities(), atol=1e-7, rtol=0
    )
    assert np.isclose(
        back.get_kinetic_energy(), sio2_atoms.get_kinetic_energy(), rtol=1e-5
    )


def test_masses_written(sio2_atoms, tmp_path):
    """Masses in model.xyz equal the ASE masses."""
    path = tmp_path / "model.xyz"
    gpumd_io.write_model_xyz(sio2_atoms, path)
    masses, _ = read_model_xyz_like_gpumd(path)
    np.testing.assert_allclose(masses, sio2_atoms.get_masses(), atol=1e-8)
