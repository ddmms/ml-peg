"""Utillity functions to process logs and compute thermodynamic properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_peg.analysis.utils.thermodynamics import (
    density,
    evaporation_enthalpy,
    heat_capacity_cp,
    isothermal_compressibility,
    thermal_expansion,
)


def read_property_from_log(
    fname: Path,
    property_name: str,
    equil_time_ps: float = 0.0,
) -> tuple[np.ndarray, str | None]:
    """
    Read a property time series and unit from a log file.

    Parameters
    ----------
    fname
        Path to the log file.
    property_name
        Name of the property to extract from the log.
    equil_time_ps
        Equilibration time in ps. Values recorded before this time are ignored.

    Returns
    -------
    np.ndarray
        Property values recorded after the equilibration time.
    str | None
        Unit associated with the property.
    """
    values = []
    unit = None

    with open(fname) as lines:
        for line in lines:
            items = line.strip().split()

            try:
                time_index = items.index("t:")
                property_index = items.index(f"{property_name}:")

                time_ps = float(items[time_index + 1])
                value = float(items[property_index + 1])

                if property_index + 2 < len(items):
                    unit = items[property_index + 2]

            except (ValueError, IndexError):
                continue

            if time_ps >= equil_time_ps:
                values.append(value)

    return np.asarray(values), unit


def analyse_liquid(
    log_file_liq: Path,
    log_file_gas: Path,
    temperature: float,
    pressure: float,
    n_molecules: int,
    equilib_time_ps: float,
    block_size: int,
) -> dict[str, tuple[float, float]]:
    """
    Analyse thermodynamic properties for one liquid simulation.

    Parameters
    ----------
    log_file_liq
        Path to the NPT liquid phase production log.
    log_file_gas
        Path to the NVT gas phase production log.
    temperature
        Simulation temperature in K.
    pressure
        Simulation pressure in bar.
    n_molecules
        Number of molecules in the liquid simulation cell.
    equilib_time_ps
        Initial length of trajectory, in ps, to be disregarded.
    block_size
        Number of logged samples in each statistical block.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mean and standard error for each thermodynamic observable.
    """
    density_series, density_units = read_property_from_log(
        log_file_liq,
        "density",
        equil_time_ps=equilib_time_ps,
    )
    volume_series, volume_units = read_property_from_log(
        log_file_liq,
        "volume",
        equil_time_ps=equilib_time_ps,
    )
    pot_energy_series, pot_energy_units = read_property_from_log(
        log_file_liq,
        "Epot",
        equil_time_ps=equilib_time_ps,
    )
    kin_energy_series, kin_energy_units = read_property_from_log(
        log_file_liq,
        "Ekin",
        equil_time_ps=equilib_time_ps,
    )

    pot_energy_series_gas, pot_energy_units_gas = read_property_from_log(
        log_file_gas,
        "Epot",
        equil_time_ps=equilib_time_ps,
    )

    return {
        "density": density(
            density_series,
            block_size=block_size,
        ),
        "cp": heat_capacity_cp(
            pot_energy_series,
            kin_energy_series,
            volume_series,
            temperature=temperature,
            pressure=pressure,
            n_molecules=n_molecules,
            block_size=block_size,
        ),
        "evaporation_enthalpy": evaporation_enthalpy(
            pot_energy_series,
            pot_energy_series_gas,
            volume_series,
            temperature=temperature,
            pressure=pressure,
            n_molecules=n_molecules,
            block_size=block_size,
        ),
        "compressibility": isothermal_compressibility(
            volume_series,
            temperature=temperature,
            block_size=block_size,
        ),
        "alpha": thermal_expansion(
            pot_energy_series,
            kin_energy_series,
            volume_series,
            temperature=temperature,
            pressure=pressure,
            block_size=block_size,
        ),
    }
