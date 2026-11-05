"""Stream water trajectories into a reusable, disk-backed ASE cache."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tempfile
from warnings import warn

from ase.io import iread
from ase.io.trajectory import Trajectory

# Temporary caches live only for this analysis process. No cached run survives a rerun.
_CACHE = tempfile.TemporaryDirectory(prefix="mlpeg-water-")


@lru_cache(maxsize=8)
def _cached_frames(path: str, size: int, modified: int) -> Trajectory:
    """
    Convert one extxyz revision without retaining ASE frames in memory.

    Parameters
    ----------
    path : str
        Absolute path to the source extxyz trajectory.
    size : int
        Source file size in bytes, used to identify the cached revision.
    modified : int
        Source modification time in nanoseconds, used to identify the cached revision.

    Returns
    -------
    ase.io.trajectory.TrajectoryReader
        Disk-backed trajectory containing the source frames.
    """
    with tempfile.NamedTemporaryFile(
        dir=_CACHE.name, suffix=".traj", delete=False
    ) as f:
        target = f.name
    try:
        with Trajectory(target, "w") as output:
            for frame in iread(path, index=":"):
                output.write(frame)
        return Trajectory(target, "r")
    except Exception:
        Path(target).unlink(missing_ok=True)
        raise


def cached_frames(path: Path) -> Trajectory:
    """
    Return random-access frames, invalidating the cache when the source changes.

    Parameters
    ----------
    path : pathlib.Path
        Path to the source extxyz trajectory.

    Returns
    -------
    ase.io.trajectory.TrajectoryReader
        Disk-backed trajectory for the current source revision.
    """
    stat = path.stat()
    return _cached_frames(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def validated_frames(path: Path, expected_frames: int):
    """
    Return complete-run frames or None for missing, failed or corrupt outputs.

    A run may include the production boundary frame, so the expected number is a
    minimum rather than an exact count. The in-progress marker also catches kills.

    Parameters
    ----------
    path : pathlib.Path
        Path to the model's extxyz trajectory.
    expected_frames : int
        Minimum number of production frames required for a successful run.

    Returns
    -------
    ase.io.trajectory.TrajectoryReader or None
        Validated disk-backed frames, or None when the run cannot be scored.
    """
    if (path.parent / "md-failed").exists() or not path.is_file():
        return None
    try:
        frames = cached_frames(path)
        if len(frames) < expected_frames or not len(frames):
            raise ValueError(
                f"Expected at least {expected_frames} frames, got {len(frames)}"
            )
        return frames
    except Exception as exc:
        warn(f"Invalid MD trajectory {path}: {exc}", stacklevel=2)
        return None


@lru_cache(maxsize=8)
def _cached_arrays(path: str, size: int, modified: int):
    """
    Convert positions and velocities together into compact disk-backed arrays.

    Parameters
    ----------
    path : str
        Absolute path to the source extxyz trajectory.
    size : int
        Source file size in bytes, used to identify the cached revision.
    modified : int
        Source modification time in nanoseconds, used to identify the cached revision.

    Returns
    -------
    numpy.memmap
        Positions in nm, with shape (frames, atoms, 3).
    numpy.memmap
        ASE velocities scaled by 0.1 to nm per ASE time unit, with the same shape.
        Missing velocities are represented by NaN.
    """
    import numpy as np

    frames = _cached_frames(path, size, modified)
    shape = (len(frames), len(frames[0]), 3)
    arrays = []
    for _ in range(2):
        with tempfile.NamedTemporaryFile(
            dir=_CACHE.name, suffix=".npy", delete=False
        ) as f:
            arrays.append(
                np.lib.format.open_memmap(
                    f.name, mode="w+", dtype="float32", shape=shape
                )
            )
    positions, velocities = arrays
    for i, atoms in enumerate(frames):
        positions[i] = atoms.positions * 0.1
        velocity = atoms.get_velocities()
        velocities[i] = velocity * 0.1 if velocity is not None else np.nan
    for array in arrays:
        array.flush()
    return positions, velocities


def trajectory_arrays(path: Path):
    """
    Read converted arrays without retaining ASE frame objects.

    Parameters
    ----------
    path : pathlib.Path
        Path to the source extxyz trajectory.

    Returns
    -------
    numpy.memmap
        Positions in nm, with shape (frames, atoms, 3).
    numpy.memmap
        ASE velocities scaled by 0.1 to nm per ASE time unit, with the same shape.
        Missing velocities are represented by NaN.
    """
    stat = path.stat()
    return _cached_arrays(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
