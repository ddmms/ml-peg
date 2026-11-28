"""Resolve and verify author-hosted AM26 data without an ML-PEG data mirror."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

from ml_peg.calcs.utils.utils import download_github_data

REFERENCE_REVISION = "9218ee127f8db5ff06edecdf209f741d0eafb7c6"
REFERENCE_SHA256 = "a071f537305e78e3bb1acf5dbffaa9acf1f79e2eb5081cb59218beddaf5a802a"
REFERENCE_FILE = "data/am26.extxyz"
REFERENCE_URI = (
    f"https://raw.githubusercontent.com/vldgroup/AM26/{REFERENCE_REVISION}/data"
)
DATA_PATH_ENV = "AM26_DATA_PATH"


def get_reference_path(data_path: str | Path | None = None) -> Path:
    """
    Select the canonical file and reject any uncompressed checksum mismatch.

    Parameters
    ----------
    data_path
        Explicit local reference file, read in place without copying. Otherwise,
        use ``AM26_DATA_PATH`` when set, or ML-PEG's GitHub download/cache helper.

    Returns
    -------
    Path
        Verified uncompressed reference file. Existing cache files are checked too.

    Raises
    ------
    ValueError
        If the selected file does not match the pinned canonical SHA-256. The file
        is left untouched; it is never deleted or automatically overwritten.
    """
    selected = data_path if data_path is not None else os.environ.get(DATA_PATH_ENV)
    if selected is not None:
        path = Path(selected)
    else:
        directory = download_github_data(
            filename=Path(REFERENCE_FILE).name, github_uri=REFERENCE_URI
        )
        path = directory / Path(REFERENCE_FILE).name
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != REFERENCE_SHA256:
        raise ValueError(
            f"AM26 SHA-256 mismatch for {path}: expected {REFERENCE_SHA256}, "
            f"found {actual}. No data were accepted; the file was left untouched."
        )
    return path
