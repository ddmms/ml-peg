"""General methods to compute thermodynamic quantites."""

from __future__ import annotations

from ase import units
import numpy as np

from ml_peg.analysis.utils.utils import block_estimate, correlator

EV_TO_J_MOL = 96485.33212
EV_TO_KJ_MOL = EV_TO_J_MOL / 1000.0
ANG3_PER_EV_TO_GPA_INV = 0.006241509074


def density(values, block_size: int) -> tuple[float, float]:
    """
    Compute density and its standard error.

    Parameters
    ----------
    values
        Density time series in g/cm3.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Mean density in g/cm3.
    stderr
        Standard error in g/cm3.
    """
    return block_estimate(values, block_size=block_size)


def heat_capacity_cp(
    enthalpy, temperature: float, n_molecules: int, block_size: int
) -> tuple[float, float]:
    """
    Compute constant-pressure heat capacity and its standard error.

    Parameters
    ----------
    enthalpy
        Total-system enthalpy time series in eV.
    temperature
        Temperature in K.
    n_molecules
        Number of molecules in the simulation box.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Constant-pressure heat capacity in J/mol/K.
    stderr
        Standard error in J/mol/K.
    """
    return block_estimate(
        enthalpy,
        block_size=block_size,
        estimator=lambda h: (
            correlator(h, h) / (units.kB * temperature**2) * EV_TO_J_MOL / n_molecules
        ),
    )


def isothermal_compressibility(
    volume, temperature: float, block_size: int
) -> tuple[float, float]:
    """
    Compute isothermal compressibility and its standard error.

    Parameters
    ----------
    volume
        Volume time series in Angstrom^3.
    temperature
        Temperature in K.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Isothermal compressibility in GPa^-1.
    stderr
        Standard error in GPa^-1.
    """
    return block_estimate(
        volume,
        block_size=block_size,
        estimator=lambda v: (
            correlator(v, v)
            / (units.kB * temperature * np.mean(v))
            * ANG3_PER_EV_TO_GPA_INV
        ),
    )


def thermal_expansion(
    volume, enthalpy, temperature: float, block_size: int
) -> tuple[float, float]:
    """
    Compute thermal expansion coefficient and its standard error.

    Parameters
    ----------
    volume
        Volume time series in Angstrom^3.
    enthalpy
        Total-system enthalpy time series in eV.
    temperature
        Temperature in K.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Thermal expansion coefficient in K^-1.
    stderr
        Standard error in K^-1.
    """
    return block_estimate(
        volume,
        enthalpy,
        block_size=block_size,
        estimator=lambda v, h: (
            correlator(v, h) / (units.kB * temperature**2 * np.mean(v))
        ),
    )


def enthalpy_of_vaporization(
    liquid_potential_energy,
    gas_potential_energy,
    temperature: float,
    n_molecules: int,
    block_size: int,
) -> tuple[float, float]:
    """
    Compute enthalpy of vaporization and its standard error.

    Parameters
    ----------
    liquid_potential_energy
        Total liquid potential-energy time series in eV.
    gas_potential_energy
        Single-molecule gas potential-energy time series in eV.
    temperature
        Temperature in K.
    n_molecules
        Number of molecules in the liquid simulation box.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Enthalpy of vaporization in kJ/mol.
    stderr
        Standard error in kJ/mol.
    """
    liquid_mean, liquid_stderr = block_estimate(
        liquid_potential_energy, block_size=block_size
    )
    gas_mean, gas_stderr = block_estimate(gas_potential_energy, block_size=block_size)

    delta_h = gas_mean - liquid_mean / n_molecules + units.kB * temperature

    stderr = np.sqrt(gas_stderr**2 + (liquid_stderr / n_molecules) ** 2)

    return (delta_h * EV_TO_KJ_MOL, stderr * EV_TO_KJ_MOL)
