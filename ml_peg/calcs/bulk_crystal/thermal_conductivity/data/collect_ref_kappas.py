"""
Collect reference thermal conductivity data from per-material HDF5 files.

Rebuild aggregates from the generated results so newly completed materials are
included, without creating empty full-mesh outputs after fast-only generation.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import h5py
import pandas as pd

DATA_PATH = Path(__file__).parent

sys.path.append(str(DATA_PATH.parent))
tc = importlib.import_module("thermal_conductivity")

PBE_DATA_PATH = DATA_PATH / "PBE"


def collect_reference_kappas(filename_no_ext: str, output_name_no_ext: str) -> None:
    """
    Rebuild a reference aggregate from per-material results when data exists.

    Parameters
    ----------
    filename_no_ext : str
        Base name of the per-material HDF5 files.
    output_name_no_ext : str
        Base name of the aggregate JSON and HDF5 files.
    """
    print(f"Loading {filename_no_ext} from {PBE_DATA_PATH} subdirectories...")
    dicts = tc.load_hdf5_subdir_dicts(PBE_DATA_PATH, f"{filename_no_ext}.hdf5")
    if not dicts:
        print(f"No {filename_no_ext} data found. Skipping collection.")
        return

    df = pd.DataFrame(dicts).T
    df.index.name = tc.TCKeys.mat_id
    df.reset_index().to_json(PBE_DATA_PATH / f"{output_name_no_ext}.json.gz")
    with h5py.File(PBE_DATA_PATH / f"{output_name_no_ext}.hdf5", "w") as f:
        tc.dict_to_hdf5(dicts, f)


collect_reference_kappas("fast_kappa", "fast_kappas")
collect_reference_kappas("kappa", "kappas")
