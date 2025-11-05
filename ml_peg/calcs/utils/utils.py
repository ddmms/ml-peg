"""Utility functions for running calculations."""

from __future__ import annotations

import contextlib
import os
import pathlib
from pathlib import Path
import zipfile

import requests

from ml_peg.data.data import download

# Local cache directory
BENCHMARK_DATA_DIR = pathlib.Path.home() / ".cache" / "ml_peg"


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

    # Download file if not already cached or if force is True
    if force or not local_path.exists():
        print(f"[download] Downloading {endpoint}/{bucket}/{key}")
        download(key=key, filename=local_path, bucket=bucket, endpoint=endpoint)
    else:
        print(f"[cache] Found cached file: {local_path.name}")

    # Extract contents if necessary and return path
    return extract_zip(local_path)


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

    # Download file if not already cached or if force is True
    if force or not local_path.exists():
        print(f"[download] Downloading {filename} from {uri}")

        response = requests.get(uri)
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f_out:
            f_out.write(response.content)

    else:
        print(f"[cache] Found cached file: {local_path.name}")

    # Extract contents if necessary and return path
    return extract_zip(local_path)


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
    # If it's a zip, try to extract it
    if filename.suffix == ".zip":
        extract_dir = filename.parent
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
