"""
Pytest plugin: cross-job lock files so concurrent SLURM jobs share the work.

Multiple independent jobs can run the same pytest invocation against the same
checkout. Before a test runs, the job atomically claims a lock file under
``MLPEG_LOCK_DIR`` (``O_CREAT | O_EXCL`` on shared storage); if another job
already holds the claim, the test is skipped and the job moves on to the next
one. Together with ml-peg's completion markers this makes the jobs share the
(model, benchmark) pool without ever computing the same pair twice.

Locks are released on test teardown whether the test passed or failed, so
only jobs killed mid-test (e.g. by the walltime) leave locks behind; locks
older than ``MLPEG_LOCK_STALE_HOURS`` (default 25, just over the 24 h
walltime) are treated as such leftovers and reclaimed.

The plugin is a no-op unless ``MLPEG_LOCK_DIR`` is set. Load it with
``pytest -p mlpeg_job_lock`` (the scripts directory must be on PYTHONPATH).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import time

import pytest
from pytest import Item

# Locks claimed by this process, released in pytest_runtest_teardown
_held_locks: set[Path] = set()


def _lock_dir() -> Path | None:
    """
    Get the shared lock directory, if configured.

    Returns
    -------
    Path | None
        Directory from the MLPEG_LOCK_DIR environment variable, or None.
    """
    value = os.environ.get("MLPEG_LOCK_DIR")
    return Path(value) if value else None


def _lock_path(lock_dir: Path, nodeid: str) -> Path:
    """
    Map a test node ID to its lock file.

    Parameters
    ----------
    lock_dir
        Shared lock directory.
    nodeid
        Pytest test node ID, e.g. "path/to/calc_x.py::test_x[model]".

    Returns
    -------
    Path
        Lock file path: a readable slug plus a hash of the full node ID.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", nodeid)[-80:]
    digest = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
    return lock_dir / f"{slug}.{digest}.lock"


def _try_claim(lock: Path) -> bool:
    """
    Atomically create a lock file claiming a test for this job.

    Parameters
    ----------
    lock
        Lock file path.

    Returns
    -------
    bool
        Whether the claim succeeded (False if the lock already exists).
    """
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf8") as file:
        file.write(f"{os.environ.get('SLURM_JOB_ID', os.getpid())}\n")
    return True


def pytest_runtest_setup(item: Item) -> None:
    """
    Claim the test's lock, or skip the test if another job holds it.

    Parameters
    ----------
    item
        Pytest test item.
    """
    lock_dir = _lock_dir()
    if lock_dir is None:
        return
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(lock_dir, item.nodeid)

    if not _try_claim(lock):
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            # Lock released between the claim attempt and stat
            age = None
        stale = float(os.environ.get("MLPEG_LOCK_STALE_HOURS", "25")) * 3600
        if age is not None and age < stale:
            pytest.skip(f"claimed by another job ({lock.name})")
        # Stale or vanished lock: reclaim it. Unlink + create is not atomic;
        # the worst case is two jobs redoing one test, which is harmless.
        try:
            lock.unlink()
        except OSError:
            pass
        if not _try_claim(lock):
            pytest.skip(f"claimed by another job ({lock.name})")

    _held_locks.add(lock)


def pytest_runtest_teardown(item: Item) -> None:
    """
    Release the test's lock if this job claimed it.

    Parameters
    ----------
    item
        Pytest test item.
    """
    lock_dir = _lock_dir()
    if lock_dir is None:
        return
    lock = _lock_path(lock_dir, item.nodeid)
    if lock in _held_locks:
        _held_locks.discard(lock)
        try:
            lock.unlink()
        except OSError:
            pass
