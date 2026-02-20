"""
Collects reference thermal conductivity data.

It loads the data from JSON and HDF5 files, processes it into pandas DataFrames,
and saves it back to disk in both formats for easy access in future analyses.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import h5py
import pandas as pd

DATA_PATH = Path(__file__).parent

tc = importlib.import_module(DATA_PATH.parent / "thermal_conductivity.py")

PBE_DATA_PATH = DATA_PATH / "PBE"

FAST_JSON_PATH = PBE_DATA_PATH / "fast_kappas.json.gz"
FAST_HDF5_PATH = PBE_DATA_PATH / "fast_kappas.hdf5"

ORIGINAL_JSON_PATH = PBE_DATA_PATH / "kappas.json.gz"
ORIGINAL_HDF5_PATH = PBE_DATA_PATH / "kappas.hdf5"


if FAST_JSON_PATH.exists():
    print(f"Loading fast kappa dicts from {FAST_JSON_PATH}...")
    df_fast = pd.read_json(FAST_JSON_PATH)
    fast_dicts = df_fast.set_index(tc.TCKeys.mat_id).to_dict(orient="index")
    print("Loading finished.")

elif FAST_HDF5_PATH.exists():
    print(f"Loading fast kappa dicts from {FAST_HDF5_PATH}...")
    with h5py.File(FAST_HDF5_PATH, "r") as f:
        fast_dicts = tc.hdf5_to_dict(f)
    print("Loading finished.")
else:
    print(f"Loading fast kappa dicts from {PBE_DATA_PATH} subdirectories...")
    fast_dicts = tc.load_hdf5_subdir_dicts(PBE_DATA_PATH, "fast_kappa.hdf5")
    print("Loading finished.")

if ORIGINAL_JSON_PATH.exists():
    print(f"Loading original kappa dicts from {ORIGINAL_JSON_PATH}...")
    df_original = pd.read_json(ORIGINAL_JSON_PATH)
    original_dicts = df_original.set_index(tc.TCKeys.mat_id).to_dict(orient="index")
    print("Loading finished.")

elif ORIGINAL_HDF5_PATH.exists():
    print(f"Loading original kappa dicts from {ORIGINAL_HDF5_PATH}...")
    with h5py.File(ORIGINAL_HDF5_PATH, "r") as f:
        original_dicts = tc.hdf5_to_dict(f)
    print("Loading finished.")
else:
    print(f"Loading original kappa dicts from {PBE_DATA_PATH} subdirectories...")
    original_dicts = tc.load_hdf5_subdir_dicts(PBE_DATA_PATH, "kappa.hdf5")
    print("Loading finished.")


df_fast = pd.DataFrame(fast_dicts).T
df_original = pd.DataFrame(original_dicts).T

df_fast.index.name = tc.TCKeys.mat_id
df_original.index.name = tc.TCKeys.mat_id
df_fast.reset_index().to_json(FAST_JSON_PATH)
df_original.reset_index().to_json(ORIGINAL_JSON_PATH)

with h5py.File(FAST_HDF5_PATH, "w") as f:
    tc.dict_to_hdf5(fast_dicts, f)
with h5py.File(ORIGINAL_HDF5_PATH, "w") as f:
    tc.dict_to_hdf5(original_dicts, f)
