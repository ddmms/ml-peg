"""Utilities for the MOF adsorption enthalpy benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase import units
import numpy as np

from ml_peg.calcs.utils.utils import download_s3_data

S3_KEY = "inputs/porous_materials/mof_adsorption_enthalpy/mof_adsorption_enthalpy.zip"
S3_FILENAME = "mof_adsorption_enthalpy.zip"

# Energy given by FLAMES to insertions rejected by its overlap check (eV)
OVERLAP_ENERGY = 1.0e5
# Insertions below this energy are counted as unphysical and given zero weight (eV)
ENERGY_FLOOR = -1.25

R_GAS = units.kB / units.J * units.mol  # J/mol/K
EV_TO_KJ_MOL = 1.0 / (units.kJ / units.mol)


def get_data_dir() -> Path:
    """
    Get directory containing the benchmark input data.

    Returns
    -------
    Path
        Path to the extracted input data.
    """
    return (
        download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "mof_adsorption_enthalpy"
    )


def load_reference(data_dir: Path | None = None) -> dict[str, Any]:
    """
    Load framework list and experimental reference values.

    Parameters
    ----------
    data_dir
        Directory containing ``reference.json``. Default is the downloaded data.

    Returns
    -------
    dict[str, Any]
        Reference data.
    """
    data_dir = get_data_dir() if data_dir is None else data_dir
    with open(data_dir / "reference.json", encoding="utf8") as file:
        return json.load(file)


def scored_frameworks(reference: dict[str, Any]) -> list[str]:
    """
    Get frameworks with an experimental enthalpy.

    Parameters
    ----------
    reference
        Reference data.

    Returns
    -------
    list[str]
        Names of frameworks with a reference value.
    """
    return sorted(
        name
        for name, entry in reference["frameworks"].items()
        if entry["enthalpy"] is not None
    )


def widom_estimates(
    energies: np.ndarray,
    temperature: float,
    density: float,
    energy_floor: float = ENERGY_FLOOR,
) -> dict[str, float]:
    """
    Calculate Henry coefficient and enthalpy of adsorption from insertions.

    Overlapping and unphysical insertions are given zero Boltzmann weight.

    Parameters
    ----------
    energies
        Interaction energy of each trial insertion, in eV.
    temperature
        Temperature in K.
    density
        Framework density in kg/m^3.
    energy_floor
        Energy below which an insertion is counted as unphysical, in eV.

    Returns
    -------
    dict[str, float]
        Henry coefficient (mol/kg/Pa), enthalpy (kJ/mol), and fractions of
        overlapping and unphysical insertions.
    """
    energies = np.asarray(energies, dtype=float)
    overlap = ~np.isfinite(energies) | (energies >= OVERLAP_ENERGY)
    unphysical = ~overlap & (energies < energy_floor)
    valid = energies[~overlap & ~unphysical]
    n_evaluated = np.count_nonzero(~overlap)

    results = {
        "henry_coefficient": np.nan,
        "enthalpy": np.nan,
        "overlap_fraction": np.count_nonzero(overlap) / max(energies.size, 1),
        "unphysical_fraction": np.count_nonzero(unphysical) / max(n_evaluated, 1),
    }
    if valid.size == 0:
        return results

    # Shift by the minimum energy to avoid overflow
    beta = 1.0 / (units.kB * temperature)
    e_min = valid.min()
    weights = np.exp(-beta * (valid - e_min))
    mean_weight = weights.sum() / energies.size * np.exp(-beta * e_min)

    results["henry_coefficient"] = mean_weight / (density * R_GAS * temperature)
    results["enthalpy"] = (
        np.average(valid, weights=weights) - units.kB * temperature
    ) * EV_TO_KJ_MOL
    return results
