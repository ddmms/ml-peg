"""Shared utility functions for the carbon benchmarks."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
import numpy as np

from ml_peg.calcs.utils.utils import download_github_data

GITHUB_URI = "https://raw.githubusercontent.com/patrickwrowe/Carbon_GAP/main/ml_peg_benchmark_data"


def load_carbon_systems(benchmark: str) -> tuple[Path, list[str]]:
    """
    Download a carbon benchmark's reference data and read the systems it ships.

    Parameters
    ----------
    benchmark
        Name of the benchmark, matching both its zip file and the directory that
        zip extracts to.

    Returns
    -------
    tuple[Path, list[str]]
        Path to the extracted data, and the system names read from its `list` file.
    """
    data_dir = (
        download_github_data(filename=f"{benchmark}.zip", github_uri=GITHUB_URI)
        / benchmark
    )
    with open(data_dir / "list") as file:
        return data_dir, file.read().splitlines()


def energy_at(atoms: Atoms, calc: Calculator, label: str) -> float:
    """
    Get the single-point potential energy of a copy of `atoms` with `calc` attached.

    Parameters
    ----------
    atoms
        Reference structure to evaluate, unmodified.
    calc
        ASE calculator to attach.
    label
        Description of the structure being evaluated, for warning messages.

    Returns
    -------
    float
        Potential energy in eV, or `np.nan` on failure.
    """
    struct = atoms.copy()
    struct.info.setdefault("charge", 0)
    struct.info.setdefault("spin", 1)
    struct.calc = copy(calc)
    try:
        return struct.get_potential_energy()
    except Exception as exc:
        warn(f"Error computing energy for {label}: {exc}", stacklevel=2)
        return np.nan
