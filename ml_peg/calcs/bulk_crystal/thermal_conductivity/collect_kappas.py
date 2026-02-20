"""Collects thermal conductivity data from HDF5 files."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import pandas as pd

from ml_peg.calcs.bulk_crystal.thermal_conductivity import thermal_conductivity as tc

OUT_PATH = Path(__file__).parent / "outputs"


if len(sys.argv) == 1:
    raise ValueError(
        "Please provide the path to the directory containing the kappa.hdf5 files "
        "as a command-line argument."
    )

models = sys.argv[1:]


def collect_kappas(parent_dir: Path, filename_no_ext: str, output_name_no_ext: str):
    """
    Collect kappa data from HDF5 files in the directory and save as JSON and HDF5.

    Parameters
    ----------
    parent_dir : Path
        The directory containing the HDF5 files to collect.
    filename_no_ext : str
        The base name of the HDF5 files to look for (without extension).
    output_name_no_ext : str
        The base name to use for the output JSON and HDF5 files (without extension).
    """
    print(f"Loading {filename_no_ext} from {parent_dir} subdirectories...")
    dicts = tc.load_hdf5_subdir_dicts(parent_dir, filename_no_ext + ".hdf5")
    print("Loading finished.")

    df = pd.DataFrame(dicts).T
    df.index.name = tc.TCKeys.mat_id
    df.reset_index().to_json(OUT_PATH / f"{output_name_no_ext}.json.gz")
    with h5py.File(parent_dir / f"{output_name_no_ext}.hdf5", "w") as f:
        tc.dict_to_hdf5(dicts, f)


for model in models:
    model_dir = OUT_PATH / model
    if not model_dir.exists():
        print(f"Model directory {model_dir} does not exist. Skipping.")
        continue

    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]

    if not any(not (d / "fast_kappa.hdf5").exists() for d in subdirs):
        try:
            collect_kappas(model_dir, "fast_kappa", "fast_kappa")
        except Exception as exc:
            print(f"Error collecting fast kappas for {model}: {exc}")
    else:
        print(
            "Fast kappa files not found in all subdirectories."
            "Skipping fast kappa collection."
        )

    if not any(not (d / "kappa.hdf5").exists() for d in subdirs):
        try:
            collect_kappas(model_dir, "kappa", "kappa")
        except Exception as exc:
            print(f"Error collecting kappas for {model}: {exc}")
    else:
        print("Kappa files not found in all subdirectories. Skipping kappa collection.")
