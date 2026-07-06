"""Utility functions for running calculations."""

from __future__ import annotations

import contextlib
import fcntl
import os
import pathlib
from pathlib import Path
import zipfile

import requests

from ml_peg.calcs.utils.completion import record_data_file
from ml_peg.data.data import download

# Local cache directory
BENCHMARK_DATA_DIR = pathlib.Path.home() / ".cache" / "ml_peg"


@contextlib.contextmanager
def cache_lock(path: Path):
    """
    Hold an exclusive inter-process lock while downloading/extracting a file.

    Blocks until any other process holding the lock for the same path releases
    it. The lock is tied to an open file descriptor, so it is released
    automatically if the holding process dies. The `<name>.lock` file exists
    whether or not anyone holds the lock, so its presence is not informative.
    But deletion is not necessarily safe: fcntl locks live on the inode, so a
    process locking a recreated file is not excluded by holders of the old one.

    Parameters
    ----------
    path
        Path to the cached file to lock.
    """
    lock_path = path.parent / (path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.lockf(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(lock_file, fcntl.LOCK_UN)


def _extracted_marker(path: Path) -> Path:
    """
    Get the marker file recording that a cached file was fully extracted.

    Parameters
    ----------
    path
        Path to the cached file.

    Returns
    -------
    Path
        Path to the marker file.
    """
    return path.parent / (path.name + ".extracted")


def download_s3_data(
    key: str,
    filename: str | Path,
    bucket: str = "ml-peg-data",
    endpoint: str = "https://s3.echo.stfc.ac.uk",
    force: bool = False,
) -> Path:
    """
    Download data from an S3 bucket.

    Parameters
    ----------
    key
        Name of file in S3 bucket to download to cache directory.
    filename
        Name of file to save download as locally.
    bucket
        Name of S3 bucket. Default is "ml-peg-data".
    endpoint
        Endpoint URL. Default is "https://s3.echo.stfc.ac.uk".
    force
        Whether to ignored cached download. Default is False.

    Returns
    -------
    Path
        Path to directory containing extracted data.
    """
    local_path = Path(BENCHMARK_DATA_DIR) / filename
    marker = _extracted_marker(local_path)

    if force:
        marker.unlink(missing_ok=True)

    if marker.exists():
        print(f"[cache] Found cached file: {local_path.name}")
    else:
        # Only one process downloads and extracts; the rest block on the lock,
        # then find the marker and skip
        with cache_lock(local_path):
            if not marker.exists():
                if force or not local_path.exists():
                    print(f"[download] Downloading {endpoint}/{bucket}/{key}")
                    download(
                        key=key, filename=local_path, bucket=bucket, endpoint=endpoint
                    )
                else:
                    print(f"[cache] Found cached file: {local_path.name}")
                extract_zip(local_path)
                marker.touch()

    record_data_file(local_path)
    return local_path.parent


def download_github_data(filename: str, github_uri: str, force: bool = False) -> Path:
    """
    Retrieve benchmark data from a GitHub repository.

    If it's a .zip, download and extract it.

    Parameters
    ----------
    filename
        Name of benchmark data file to download to cache directory.
    github_uri
        Name of GitHub URI to download data from.
    force
        Whether to ignore cached download. Default is False.

    Returns
    -------
    Path
        Path to directory containing extracted data.
    """
    uri = f"{github_uri}/{filename}"
    local_path = Path(BENCHMARK_DATA_DIR) / filename
    marker = _extracted_marker(local_path)

    if force:
        marker.unlink(missing_ok=True)

    if marker.exists():
        print(f"[cache] Found cached file: {local_path.name}")
    else:
        # Only one process downloads and extracts; the rest block on the lock,
        # then find the marker and skip
        with cache_lock(local_path):
            if not marker.exists():
                if force or not local_path.exists():
                    print(f"[download] Downloading {filename} from {uri}")

                    response = requests.get(uri)
                    response.raise_for_status()
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write via a temporary file so a crash mid-download cannot
                    # leave a plausible-looking partial file behind
                    part_path = local_path.parent / (local_path.name + ".part")
                    with open(part_path, "wb") as f_out:
                        f_out.write(response.content)
                    part_path.replace(local_path)
                else:
                    print(f"[cache] Found cached file: {local_path.name}")
                extract_zip(local_path)
                marker.touch()

    record_data_file(local_path)
    return local_path.parent


def extract_zip(filename: Path) -> Path:
    """
    Attempt to extract a zip file.

    Parameters
    ----------
    filename
        Name of potential zip file to extract.

    Returns
    -------
    Path
        Parent directory of unziped file.
    """
    extract_dir = filename.parent

    # If it's a zip, try to extract it
    if filename.suffix == ".zip":
        try:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        except (ValueError, RuntimeError, zipfile.BadZipFile) as err:
            raise ValueError(f"Unable to unzip file: {filename}") from err
    return extract_dir


@contextlib.contextmanager
def chdir(path: Path):
    """
    Change working directory and return to previous on exit.

    Parameters
    ----------
    path
        Path to temporarily change to.
    """
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
