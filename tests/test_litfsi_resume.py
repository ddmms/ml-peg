"""Test resuming a WiSE MD stage from the output of a job killed mid-write."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

from ase.build import bulk
from ase.io import read, write
import pytest

from ml_peg.calcs.electrolytes.litfsi_h2o_21m import calc_litfsi_h2o_21m as calc_module
from ml_peg.calcs.electrolytes.litfsi_h2o_21m.litfsi_utils import (
    atomic_path,
    resume_state,
    run_stage,
)

STATS_EVERY = 10
TRAJ_EVERY = 20
RESTART_EVERY = 100
STEPS = 450


def frame_at(step: int):
    """
    Build a frame tagged with its step, as janus-core writes it.

    Parameters
    ----------
    step
        MD step of the frame.

    Returns
    -------
    Atoms
        Small copper cell with ``step`` in its info.
    """
    frame = bulk("Cu", cubic=True)
    frame.info["step"] = step
    return frame


def as_text(frame) -> str:
    """
    Serialise a frame to extxyz.

    Parameters
    ----------
    frame
        Structure to serialise.

    Returns
    -------
    str
        The frame in extxyz format.
    """
    buffer = StringIO()
    write(buffer, frame, format="extxyz")
    return buffer.getvalue()


def write_stage(prefix: Path, last_step: int) -> None:
    """
    Write the statistics, trajectory and restart files of a stage up to a step.

    Parameters
    ----------
    prefix
        Prefix of the stage's output files.
    last_step
        Last step written before the job stopped.
    """
    lines = ["# Step | Time [fs]\n"]
    lines += [
        f"{step:10d} {step:.2f}\n" for step in range(0, last_step + 1, STATS_EVERY)
    ]
    Path(f"{prefix}-stats.dat").write_text("".join(lines))
    frames = [frame_at(step) for step in range(0, last_step + 1, TRAJ_EVERY)]
    write(f"{prefix}-traj.extxyz", frames, format="extxyz")
    for step in range(RESTART_EVERY, last_step + 1, RESTART_EVERY):
        write(f"{prefix}-res-{step}.extxyz", frame_at(step), format="extxyz")


def append(path: str, text: str) -> None:
    """
    Append text to a file.

    Parameters
    ----------
    path
        File to append to.
    text
        Text to append.
    """
    with open(path, "a", encoding="utf8") as file:
        file.write(text)


def torn_frame(prefix: Path) -> int:
    """
    Kill the job while it writes the trajectory frame of step 360.

    Parameters
    ----------
    prefix
        Prefix of the stage's output files.

    Returns
    -------
    int
        Step the stage should resume from.
    """
    write_stage(prefix, 340)
    text = as_text(frame_at(360))
    append(f"{prefix}-traj.extxyz", text[: len(text) // 2])
    return 300


def torn_first_frame_past_restart(prefix: Path) -> int:
    """
    Kill the job while it writes the first frame after the restart step.

    Parameters
    ----------
    prefix
        Prefix of the stage's output files.

    Returns
    -------
    int
        Step the stage should resume from.
    """
    write_stage(prefix, 310)
    text = as_text(frame_at(320))
    append(f"{prefix}-traj.extxyz", text[: len(text) // 3])
    return 300


def torn_restart(prefix: Path) -> int:
    """
    Kill the job while it writes the restart file of step 400.

    Parameters
    ----------
    prefix
        Prefix of the stage's output files.

    Returns
    -------
    int
        Step the stage should resume from.
    """
    write_stage(prefix, 400)
    restart = Path(f"{prefix}-res-400.extxyz")
    text = restart.read_text()
    restart.write_text(text[: len(text) // 2])
    return 300


def torn_stats_line(prefix: Path) -> int:
    """
    Kill the job while it writes the statistics line of step 310.

    Parameters
    ----------
    prefix
        Prefix of the stage's output files.

    Returns
    -------
    int
        Step the stage should resume from.
    """
    write_stage(prefix, 300)
    append(f"{prefix}-stats.dat", "         3")
    return 300


def missing_frame(prefix: Path) -> int:
    """
    Lose the trajectory frame of a restart step that is itself intact.

    Parameters
    ----------
    prefix
        Prefix of the stage's output files.

    Returns
    -------
    int
        Step the stage should resume from.
    """
    write_stage(prefix, 300)
    frames = read(f"{prefix}-traj.extxyz", index=":")
    write(f"{prefix}-traj.extxyz", frames[:-1], format="extxyz")
    return 200


@pytest.mark.parametrize(
    "damage",
    [
        torn_frame,
        torn_first_frame_past_restart,
        torn_restart,
        torn_stats_line,
        missing_frame,
    ],
)
def test_resume_after_kill(tmp_path, damage):
    """Test the stage resumes from the last step its output is intact up to."""
    prefix = tmp_path / "stage"
    expected = damage(prefix)

    resumed, step = resume_state(prefix, STEPS, traj_every=TRAJ_EVERY)

    assert step == expected
    assert resumed.info["step"] == expected
    frames = read(f"{prefix}-traj.extxyz", index=":")
    assert [frame.info["step"] for frame in frames] == list(
        range(0, expected + 1, TRAJ_EVERY)
    )
    stats = Path(f"{prefix}-stats.dat").read_text().splitlines()
    assert [int(line.split()[0]) for line in stats[1:]] == list(
        range(0, expected + 1, STATS_EVERY)
    )


def test_no_resume_across_missing_interior_frame(tmp_path):
    """Test a trajectory with a gap before the restart step is not resumed."""
    prefix = tmp_path / "stage"
    write_stage(prefix, 100)
    frames = read(f"{prefix}-traj.extxyz", index=":")
    del frames[2]
    write(f"{prefix}-traj.extxyz", frames, format="extxyz")

    assert resume_state(prefix, STEPS, traj_every=TRAJ_EVERY) is None


def test_incomplete_final_file_resumes(tmp_path):
    """Test an unreadable final file falls back to the restart files."""
    prefix = tmp_path / "stage"
    write_stage(prefix, 300)
    Path(f"{prefix}-final.extxyz").write_text("2\n")
    seen = {}

    class Ensemble:
        """Stand-in for a janus-core ensemble that records its arguments."""

        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.struct = kwargs["struct"]

        def run(self):
            """Do nothing."""

    start = frame_at(0)
    start.calc = None
    run_stage(Ensemble, start, steps=STEPS, file_prefix=prefix, traj_every=TRAJ_EVERY)

    assert seen["restart"] is True
    assert seen["struct"].info["step"] == 300


class BrokenModel:
    """Model whose calculator cannot be built."""

    def get_calculator(self, **kwargs):
        """
        Fail as a model with a missing dependency would.

        Parameters
        ----------
        **kwargs
            Ignored.
        """
        raise ImportError("no backend")


def test_calculator_failure_is_recorded(tmp_path: Path, monkeypatch) -> None:
    """
    Check a model whose calculator cannot be built is marked as failed.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    monkeypatch
        Pytest fixture, used to redirect the output directory.
    """
    monkeypatch.setattr(calc_module, "OUT_PATH", tmp_path)

    with pytest.warns(UserWarning, match="Could not build the calculator"):
        failed = calc_module.run_reference_md("broken", BrokenModel())

    assert failed == [calc_module.CELL_P64, calc_module.CELL_P16]
    for cell in failed:
        marker = json.loads((tmp_path / "broken" / f"failed_{cell}.json").read_text())
        assert marker["cell"] == cell
        assert "no backend" in marker["error"]


def test_calculator_failure_keeps_complete_data(tmp_path: Path, monkeypatch) -> None:
    """
    Check a calculator failure does not invalidate data already complete.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    monkeypatch
        Pytest fixture, used to redirect the output directory.
    """
    monkeypatch.setattr(calc_module, "OUT_PATH", tmp_path)
    data_dir = tmp_path / "broken"
    data_dir.mkdir()
    with atomic_path(data_dir / "nvt_trajectory.extxyz") as tmp_file:
        write(tmp_file, [frame_at(0), frame_at(TRAJ_EVERY)], format="extxyz")

    with pytest.warns(UserWarning, match="Could not build the calculator"):
        failed = calc_module.run_reference_md("broken", BrokenModel())

    assert failed == [calc_module.CELL_P16]
    assert not (data_dir / f"failed_{calc_module.CELL_P64}.json").exists()
    assert (data_dir / f"failed_{calc_module.CELL_P16}.json").exists()


def test_calculator_failure_marks_incomplete_product(
    tmp_path: Path, monkeypatch
) -> None:
    """
    Check a product interrupted mid-write does not count as complete.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    monkeypatch
        Pytest fixture, used to redirect the output directory.
    """
    monkeypatch.setattr(calc_module, "OUT_PATH", tmp_path)
    data_dir = tmp_path / "broken"
    data_dir.mkdir()
    product = data_dir / "nvt_trajectory.extxyz"

    with pytest.raises(KeyboardInterrupt), atomic_path(product) as tmp_file:
        tmp_file.write_text("1534\nLattice=")
        raise KeyboardInterrupt

    assert not product.exists()

    with pytest.warns(UserWarning, match="Could not build the calculator"):
        failed = calc_module.run_reference_md("broken", BrokenModel())

    assert failed == [calc_module.CELL_P64, calc_module.CELL_P16]
    assert (data_dir / f"failed_{calc_module.CELL_P64}.json").exists()
