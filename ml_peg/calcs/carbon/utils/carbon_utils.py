"""Shared utility functions for the carbon benchmarks."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
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


def get_dispersion_variants(model: Any, calc: Calculator) -> tuple[Calculator, ...]:
    """
    Get the calculator variants to evaluate one model with.

    A model already trained on dispersion is evaluated once; any other model is
    evaluated twice, without and with a D3 correction.

    Parameters
    ----------
    model
        Model instance providing the dispersion correction.
    calc
        Uncorrected calculator for `model`.

    Returns
    -------
    tuple[Calculator, ...]
        The uncorrected calculator alone, or the uncorrected and D3-corrected
        calculators.
    """
    if model.trained_on_dispersion:
        return (calc,)
    return (calc, model.add_d3_calculator(copy(calc)))


def write_variant_frame(
    struct_file: Path, atoms: Atoms, variant_index: int, n_variants: int
) -> None:
    """
    Write one variant's result, keeping every output file two frames long.

    Frame 0 holds the uncorrected result and frame 1 the D3-corrected one, so a
    model evaluated once has its single result written to both.

    Parameters
    ----------
    struct_file
        Extxyz file to write to.
    atoms
        Structure to write.
    variant_index
        Index of the variant being written.
    n_variants
        Number of variants being evaluated for this model.
    """
    write(struct_file, atoms, append=variant_index > 0)
    if n_variants == 1:
        write(struct_file, atoms, append=True)
