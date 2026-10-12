"""
Build the input archive for the ACWF equation-of-state benchmark.

The archive collects, from pinned upstream commits:

* the PBE input structures (XSF files at the central volume) of the ACWF
  verification set, for the unaries and the oxides (Z = 1-96);
* the PBE all-electron reference Birch-Murnaghan parameters published with
  Bosoni et al. (2024), averaged over WIEN2k and FLEUR.

The resulting ``equation_of_state_acwf.zip`` is written to the ML-PEG cache, so the
calculation and analysis can run before it is uploaded to the ML-PEG S3 bucket as
``inputs/bulk_crystal/equation_of_state_acwf/equation_of_state_acwf.zip``.

Run with::

    python ml_peg/calcs/bulk_crystal/equation_of_state_acwf/prepare_inputs.py
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import requests

from ml_peg.calcs.utils.utils import BENCHMARK_DATA_DIR

ARCHIVE_NAME = "equation_of_state_acwf"

ACWF_REPO = "aiidateam/acwf-verification-scripts"
ACWF_COMMIT = "8eb1e9a35b512780c4462ceb8f3db815096c25cf"
STRUCTURE_DIRS = {
    "unaries": "0-preliminary-do-not-run/unaries/xsfs-unaries-verification-PBE-v1",
    "oxides": "0-preliminary-do-not-run/oxides/xsfs-oxides-verification-PBE-v1",
}

REFERENCE_DIR = "acwf_paper_plots/code-data"
REFERENCE_FILES = {
    "unaries": "results-unaries-verification-PBE-v1-AE-average.json",
    "oxides": "results-oxides-verification-PBE-v1-AE-average.json",
}

README = f"""\
ACWF equation-of-state benchmark inputs for ML-PEG.

structures/{{unaries,oxides}}/*.xsf
    PBE input structures at the central volume, from
    https://github.com/{ACWF_REPO} (commit {ACWF_COMMIT}, MIT licence):
    {", ".join(STRUCTURE_DIRS.values())}

reference/*.json
    PBE all-electron reference Birch-Murnaghan parameters (average of WIEN2k and
    FLEUR), from the same repository and commit, {REFERENCE_DIR}.

Protocol and metrics: E. Bosoni et al., Nat. Rev. Phys. 6, 45 (2024),
https://doi.org/10.1038/s42254-023-00655-3
"""


def _raw_url(repo: str, commit: str, path: str) -> str:
    """
    Get the URL of a raw file in a GitHub repository.

    Parameters
    ----------
    repo
        Repository, as ``owner/name``.
    commit
        Commit hash.
    path
        Path of the file in the repository.

    Returns
    -------
    str
        URL of the raw file.
    """
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def _list_files(repo: str, commit: str, directory: str) -> list[str]:
    """
    List the files in a directory of a GitHub repository at a given commit.

    Parameters
    ----------
    repo
        Repository, as ``owner/name``.
    commit
        Commit hash.
    directory
        Directory in the repository.

    Returns
    -------
    list[str]
        Paths of the files in the directory, relative to the repository root.
    """
    response = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{directory}",
        params={"ref": commit},
        timeout=60,
    )
    response.raise_for_status()
    return sorted(item["path"] for item in response.json() if item["type"] == "file")


def build_archive(out_dir: Path = BENCHMARK_DATA_DIR) -> Path:
    """
    Download the inputs and pack them into the benchmark archive.

    Parameters
    ----------
    out_dir
        Directory in which to write the archive. Default is the ML-PEG cache.

    Returns
    -------
    Path
        Path to the archive.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{ARCHIVE_NAME}.zip"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_archive = Path(tmp) / archive.name
        with zipfile.ZipFile(tmp_archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"{ARCHIVE_NAME}/README.txt", README)

            for dataset, filename in REFERENCE_FILES.items():
                url = _raw_url(ACWF_REPO, ACWF_COMMIT, f"{REFERENCE_DIR}/{filename}")
                print(f"[{dataset}] {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                zip_file.writestr(
                    f"{ARCHIVE_NAME}/reference/{filename}", response.content
                )

            for dataset, directory in STRUCTURE_DIRS.items():
                paths = _list_files(ACWF_REPO, ACWF_COMMIT, directory)
                print(f"[{dataset}] {len(paths)} structures from {directory}")
                for path in paths:
                    response = requests.get(
                        _raw_url(ACWF_REPO, ACWF_COMMIT, path), timeout=60
                    )
                    response.raise_for_status()
                    zip_file.writestr(
                        f"{ARCHIVE_NAME}/structures/{dataset}/{Path(path).name}",
                        response.content,
                    )

        tmp_archive.replace(archive)

    print(f"Written {archive}")
    return archive


if __name__ == "__main__":
    build_archive()
