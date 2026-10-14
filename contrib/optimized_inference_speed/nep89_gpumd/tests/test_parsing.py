"""Parsing of captured GPUMD v5.9.1 stdout, stderr and thermo.out."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from calorine.gpumd import read_runfile
import gpumd_io
import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


def feed_fixture(clock):
    """
    Feed the stdout fixture through a tracker with a synthetic clock.

    Parameters
    ----------
    clock
        Callable mapping a line to its read time.

    Returns
    -------
    gpumd_io.StdoutTracker
        The populated tracker.
    """
    tracker = gpumd_io.StdoutTracker()
    for line in (DATA / "gpumd_stdout_v5.9.1.txt").read_text().splitlines(True):
        tracker.feed(line, clock(line))
    return tracker


def test_tracker_extracts_runs_and_settings():
    """Run timers, speeds, device info and NEP settings are captured."""
    tracker = feed_fixture(lambda line: 0.0)
    assert tracker.run_seconds == [0.201768, 1.52943]
    assert tracker.speeds == [380635.0, 502147.0]
    assert tracker.n_gpus == 1
    assert tracker.devices == ["NVIDIA GeForce RTX 5080"]
    assert tracker.compute_capabilities == ["12.0"]
    assert tracker.settings == {
        "radial_cutoff": 6.0,
        "angular_cutoff": 5.0,
        "MN_radial": 274.0,
        "enlarged_MN_radial": 343.0,
        "enlarged_MN_angular": 198.0,
    }


def test_derive_timestamps():
    """Boundaries come from the stdout clock minus GPUMD's run timers."""
    seen = iter([1000.202, 1001.8])

    def clock(line):
        return next(seen) if line.startswith("Time used for this run") else 0.0

    stamps = gpumd_io.derive_timestamps(feed_fixture(clock))
    assert stamps.measure_end == pytest.approx(1001.8)
    assert stamps.warmup_end == pytest.approx(1001.8 - 1.52943)
    assert stamps.setup_end == pytest.approx(1001.8 - 1.52943 - 0.201768)
    assert stamps.run1_end_seen == pytest.approx(1000.202)
    assert stamps.run1_offset_s == pytest.approx(1001.8 - 1.52943 - 1000.202)


def test_derive_timestamps_needs_two_runs():
    """An incomplete stdout is an error, not a silent result."""
    tracker = gpumd_io.StdoutTracker()
    tracker.feed("Time used for this run = 0.1 second.\n", 1.0)
    with pytest.raises(ValueError, match="2 completed runs"):
        gpumd_io.derive_timestamps(tracker)


def test_thermo_energies():
    """Total energy is KE + PE at the last warm-up row and the last row."""
    e_start, e_end = gpumd_io.thermo_energies(DATA / "thermo_v5.9.1.out", 1, 10)
    assert e_start == pytest.approx(20.599839586 - 5994.1960652, abs=1e-9)
    assert e_end == pytest.approx(-5973.593674397, abs=1e-9)


def test_thermo_energies_row_count_mismatch():
    """A thermo.out that does not match the run lengths is rejected."""
    with pytest.raises(ValueError, match="11 rows"):
        gpumd_io.thermo_energies(DATA / "thermo_v5.9.1.out", 1, 20)


def test_detect_oom():
    """CUDA OOM is recognised; other errors are not."""
    assert gpumd_io.detect_oom("", (DATA / "stderr_oom_v5.9.1.txt").read_text())
    assert not gpumd_io.detect_oom(
        (DATA / "gpumd_stdout_v5.9.1.txt").read_text(),
        (DATA / "stderr_input_error_v5.9.1.txt").read_text(),
    )


def _write_step0(path, atoms, energy, forces, shift=None):
    pos = atoms.positions if shift is None else atoms.positions + shift
    lattice = " ".join(f"{x:.10f}" for x in atoms.cell.array.ravel())
    lines = [
        str(len(atoms)),
        f'Time=0.00000000 pbc="T T T" Lattice="{lattice}" energy={energy:.16e} '
        'virial="0 0 0 0 0 0 0 0 0" stress="0 0 0 0 0 0 0 0 0" '
        "Properties=species:S:1:pos:R:3:forces:R:3:energy_atom:R:1",
    ]
    for s, p, f in zip(atoms.get_chemical_symbols(), pos, forces, strict=True):
        lines.append(
            f"{s} {p[0]:.16e} {p[1]:.16e} {p[2]:.16e} "
            f"{f[0]:.16e} {f[1]:.16e} {f[2]:.16e} 0.0"
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def small():
    """
    Build a three-atom periodic structure.

    Returns
    -------
    ase.Atoms
        The structure.
    """
    return Atoms(
        "SiO2",
        positions=[[0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [4.9, 0.0, 2.0]],
        cell=[5, 5, 5],
        pbc=True,
    )


def test_read_step0(tmp_path, small):
    """Energy and forces are read; wrapped positions are accepted."""
    forces = np.arange(9.0).reshape(3, 3)
    path = tmp_path / "step0.xyz"
    _write_step0(
        path,
        small,
        -12.5,
        forces,
        shift=np.array([[5.0, 0, 0], [0, 0, 0], [-5.0, 0, 0]]),
    )
    energy, read_forces = gpumd_io.read_step0_xyz(path, small)
    assert energy == -12.5
    np.testing.assert_array_equal(read_forces, forces)


def test_read_step0_rejects_moved_atoms(tmp_path, small):
    """A frame that is not at the input positions is rejected."""
    path = tmp_path / "step0.xyz"
    _write_step0(path, small, -1.0, np.zeros((3, 3)), shift=1e-3)
    with pytest.raises(ValueError, match="moved"):
        gpumd_io.read_step0_xyz(path, small)


def test_read_real_step0_capture(sio2_atoms):
    """The real v5.9.1 dump (time_step 0, wrapped positions) parses and matches."""
    assert read_runfile(str(DATA / "step0_run.in")) == (
        gpumd_io.single_point_parameters()
    )
    energy, forces = gpumd_io.read_step0_xyz(DATA / "step0_v5.9.1.xyz", sio2_atoms)
    assert energy == pytest.approx(-6003.6029186248779, abs=1e-9)
    assert forces.shape == (768, 3)
    # Forces sum to ~0 for a periodic system (fp32 accumulation noise only).
    assert np.abs(forces.sum(axis=0)).max() < 1e-3
