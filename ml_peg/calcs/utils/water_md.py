"""Resume Janus water MD from the last saved positions and momenta."""

from __future__ import annotations

from mmap import ACCESS_READ, mmap
from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.io import read
import numpy as np


def _trajectory_end(path: Path, step: int, atom_count: int) -> int:
    """
    Locate the end of the checkpoint frame without loading the trajectory.

    Parameters
    ----------
    path
        Janus production trajectory in extended XYZ format.
    step
        Global MD step recorded in the checkpoint.
    atom_count
        Number of atom records per frame.

    Returns
    -------
    int
        Byte offset immediately after the checkpoint frame.
    """
    # Janus writes the global step in the extxyz comment line. Search backwards
    # since a time-limited job usually stops just beyond its last checkpoint.
    token = f" step={step}".encode()
    with (
        path.open("rb") as stream,
        mmap(stream.fileno(), 0, access=ACCESS_READ) as data,
    ):
        end = len(data)
        while (offset := data.rfind(token, 0, end)) >= 0:
            if data[offset + len(token) : offset + len(token) + 1].isspace():
                data.seek(offset)
                data.readline()
                for _ in range(atom_count):
                    if not data.readline().endswith(b"\n"):
                        raise ValueError(f"Incomplete frame at step {step} in {path}")
                return data.tell()
            end = offset
    raise ValueError(f"Missing frame at step {step} in {path}")


def water_md_restart(
    initial: Atoms, directory: Path, equil_steps: int, total_steps: int
) -> dict:
    """
    Prepare Janus restart arguments and align existing outputs to the checkpoint.

    Parameters
    ----------
    initial
        Initial structure with the model calculator, masses and constraints set.
    directory
        Model output directory containing files with the Janus prefix `md`.
    equil_steps
        Global step at which production recording starts.
    total_steps
        Target number of equilibration and production steps combined.

    Returns
    -------
    dict
        Structure, restart flags and local trajectory start for the NVT run.
    """
    checkpoints = {
        int(path.stem.split("-")[-1]): path
        for path in directory.glob("md-res-*.extxyz")
        if path.stem.removeprefix("md-res-").isdigit()
    }
    for step in sorted(checkpoints, reverse=True):
        if step > total_steps:
            raise ValueError(f"Checkpoint step {step} exceeds target {total_steps}")
        checkpoint = checkpoints[step]
        try:
            saved = read(checkpoint)
            if saved.info.get("step") != step or not saved.has("momenta"):
                raise ValueError("missing checkpoint step or momenta")
            if (
                not np.isfinite(saved.positions).all()
                or not np.isfinite(saved.get_momenta()).all()
            ):
                raise ValueError("non-finite checkpoint positions or momenta")
        except OSError:
            raise
        except Exception as exc:
            # Janus may append to an existing restart file when replaying this
            # step. Keep the damaged copy outside its checkpoint filename.
            checkpoint.rename(checkpoint.with_suffix(".extxyz.invalid"))
            warn(
                f"Skipping unreadable checkpoint {checkpoints[step]}: {exc}",
                stacklevel=2,
            )
            continue

        if not np.array_equal(saved.numbers, initial.numbers) or not np.allclose(
            saved.cell, initial.cell
        ):
            raise ValueError(
                f"Checkpoint does not match the initial structure: {checkpoints[step]}"
            )

        stats = directory / "md-stats.dat"
        trajectory = directory / "md-traj.extxyz"
        # Janus reads its offset from the last stats row with restart_auto=False.
        with stats.open("r+b") as stream:
            for line in stream:
                if line.split(maxsplit=1)[:1] == [str(step).encode()]:
                    break
            else:
                raise ValueError(f"Missing statistics for checkpoint step {step}")
            end = (
                _trajectory_end(trajectory, step, len(saved))
                if step >= equil_steps
                else 0
            )
            stream.truncate(stream.tell())
        # Janus requires the trajectory file to exist even during equilibration.
        with trajectory.open("ab") as stream:
            stream.truncate(end)

        initial.set_positions(saved.positions, apply_constraint=False)
        initial.set_momenta(saved.get_momenta(), apply_constraint=False)
        return {
            "struct": initial,
            "restart": True,
            "restart_auto": False,
            "traj_start": max(0, equil_steps - step),
        }

    return {"struct": initial, "traj_start": equil_steps}
