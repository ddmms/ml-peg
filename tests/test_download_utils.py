"""Tests for the cached download and extraction utilities."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import zipfile

import pytest

import ml_peg.calcs.utils.utils as utils
from ml_peg.calcs.utils.utils import (
    cache_lock,
    download_github_data,
    extract_zip,
)

MEMBER_NAME = "data/payload.txt"
MEMBER_CONTENT = b"0123456789abcdef" * 65536  # 1 MiB


def make_zip(path: Path) -> None:
    """
    Create a zip file containing a single known member.

    Parameters
    ----------
    path
        Path of the zip file to create.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zip_ref:
        zip_ref.writestr(MEMBER_NAME, MEMBER_CONTENT)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the benchmark data cache at a temporary directory."""
    monkeypatch.setattr(utils, "BENCHMARK_DATA_DIR", tmp_path)
    return tmp_path


def test_extract_zip_non_zip_returns_parent(tmp_path):
    """Non-zip files are returned untouched without raising."""
    path = tmp_path / "data.txt"
    path.write_text("not a zip")
    assert extract_zip(path) == tmp_path


def test_cached_zip_extracted_once(cache_dir):
    """A cached zip is extracted on first use only, then never rewritten."""
    make_zip(cache_dir / "data.zip")

    data_dir = download_github_data("data.zip", "https://unused.invalid")
    extracted = data_dir / MEMBER_NAME
    assert extracted.read_bytes() == MEMBER_CONTENT
    assert (cache_dir / "data.zip.extracted").exists()

    # A second call must not re-extract over the cache: corrupt the extracted
    # file and check it is left alone
    extracted.write_bytes(b"corrupted")
    download_github_data("data.zip", "https://unused.invalid")
    assert extracted.read_bytes() == b"corrupted"


class FakeResponse:
    """Minimal stand-in for requests.Response serving fixed bytes."""

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        """Match the requests.Response interface."""


def test_force_re_downloads_and_re_extracts(cache_dir, tmp_path, monkeypatch):
    """force=True re-downloads, removes the marker and repairs the data."""
    make_zip(cache_dir / "data.zip")

    data_dir = download_github_data("data.zip", "https://unused.invalid")
    extracted = data_dir / MEMBER_NAME
    extracted.write_bytes(b"corrupted")

    zip_bytes = (cache_dir / "data.zip").read_bytes()
    monkeypatch.setattr(utils.requests, "get", lambda uri: FakeResponse(zip_bytes))
    download_github_data("data.zip", "https://unused.invalid", force=True)

    assert extracted.read_bytes() == MEMBER_CONTENT


def _hammer(cache_dir_str: str, barrier, errors) -> None:
    """
    Repeatedly resolve the cached zip while asserting its contents are intact.

    Parameters
    ----------
    cache_dir_str
        Benchmark cache directory to use.
    barrier
        Barrier synchronising all workers to maximise contention.
    errors
        Queue collecting assertion failures from workers.
    """
    try:
        utils.BENCHMARK_DATA_DIR = Path(cache_dir_str)
        barrier.wait(timeout=30)
        for _ in range(50):
            data_dir = download_github_data("data.zip", "https://unused.invalid")
            content = (data_dir / MEMBER_NAME).read_bytes()
            if content != MEMBER_CONTENT:
                errors.put(f"read {len(content)} bytes, expected full member")
                return
    except Exception as err:
        errors.put(repr(err))


def test_concurrent_workers_see_complete_files(cache_dir):
    """Concurrent workers never observe empty or partially extracted files."""
    make_zip(cache_dir / "data.zip")

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(4)
    errors = ctx.Queue()
    workers = [
        ctx.Process(target=_hammer, args=(str(cache_dir), barrier, errors))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)

    assert errors.empty(), errors.get()


def _try_lock(lock_path_str: str, result) -> None:
    """
    Attempt a non-blocking lock from a child process.

    Parameters
    ----------
    lock_path_str
        Path of the lock file to try.
    result
        Queue receiving True if the lock was acquired, False if it was busy.
    """
    import fcntl

    with open(lock_path_str, "w") as lock_file:
        try:
            fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            result.put(False)
        else:
            result.put(True)


def test_cache_lock_is_exclusive(cache_dir):
    """Another process cannot take the lock while it is held."""
    path = cache_dir / "data.zip"
    lock_path = cache_dir / "data.zip.lock"
    ctx = multiprocessing.get_context("fork")
    result = ctx.Queue()

    with cache_lock(path):
        proc = ctx.Process(target=_try_lock, args=(str(lock_path), result))
        proc.start()
        proc.join(timeout=30)
    assert result.get(timeout=10) is False

    # And it is acquirable again once released
    proc = ctx.Process(target=_try_lock, args=(str(lock_path), result))
    proc.start()
    proc.join(timeout=30)
    assert result.get(timeout=10) is True
