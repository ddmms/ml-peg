"""General methods to compute thermodynamic quantites."""

from __future__ import annotations

from ase import units
import numpy as np

from ml_peg.analysis.utils.utils import block_estimate, correlator

AU_TO_G_L = 1e27 / units.mol
EV_TO_J_MOL = 96485.33212
EV_TO_KJ_MOL = EV_TO_J_MOL / 1000.0
ANG3_PER_EV_TO_GPA_INV = 0.006241509074
MILLI = 1.0e-3


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
        Mean density in g/L.
    stderr
        Standard error in g/L.
    """
    return block_estimate(values, block_size=block_size)


def volume(values, block_size: int) -> tuple[float, float]:
    """
    Compute volume and its standard error.

    Parameters
    ----------
    values
        Volume time series in A^3.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Mean volume in A^3.
    stderr
        Standard error in A^3.
    """
    return block_estimate(values, block_size=block_size)


def heat_capacity_cp(
    pot_energy,
    kin_energy,
    volume,
    pressure: float,
    temperature: float,
    n_molecules: int,
    block_size: int,
) -> tuple[float, float]:
    """
    Compute constant-pressure heat capacity and its standard error.

    Parameters
    ----------
    pot_energy
        Total-system potential-energy time series in eV.
    kin_energy
        Total-system kinetic-energy time series in eV.
    volume
        Simulation-cell volume time series in Angstrom^3.
    pressure
        Pressure in bar.
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
    enthalpy = (
        np.asarray(pot_energy)
        + np.asarray(kin_energy)
        + pressure * units.bar * np.asarray(volume)
    )

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
    pot_energy, kin_energy, volume, temperature: float, pressure: float, block_size: int
) -> tuple[float, float]:
    """
    Compute thermal expansion coefficient and its standard error.

    Parameters
    ----------
    pot_energy
        Total-system potential energy time series in eV.
    kin_energy
        Total-system potential energy time series in eV.
    volume
        Volume time series in Angstrom^3.
    temperature
        Temperature in K.
    pressure
        Pressure in bar.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Thermal expansion coefficient in 1e-3 K^-1.
    stderr
        Standard error in 1e-3 K^-1.
    """
    enthalpy = (
        np.asarray(pot_energy)
        + np.asarray(kin_energy)
        + pressure * units.bar * np.asarray(volume)
    )

    return block_estimate(
        volume,
        enthalpy,
        block_size=block_size,
        estimator=lambda v, h: (
            correlator(v, h) / (MILLI * units.kB * temperature**2 * np.mean(v))
        ),
    )


def evaporation_enthalpy(
    liquid_pot_energy,
    gas_pot_energy,
    liquid_volume,
    pressure: float,
    temperature: float,
    n_molecules: int,
    block_size: int,
) -> tuple[float, float]:
    """
    Compute evaporation enthalpy and its standard error.

    The gas phase is treated as ideal, so its pV contribution is k_B T
    per molecule. The liquid pV contribution is calculated explicitly.
    Note that this expression is valid only far from the critical point
    where the ideal gas equation of state is valid.

    Parameters
    ----------
    liquid_pot_energy
        Total liquid potential-energy time series in eV.
    gas_pot_energy
        Single-molecule gas potential-energy time series in eV.
    liquid_volume
        Liquid simulation-cell volume time series in A^3.
    pressure
        Pressure in bar.
    temperature
        Temperature in K.
    n_molecules
        Number of molecules in the liquid simulation box.
    block_size
        Number of samples in each block.

    Returns
    -------
    mean
        Evaporation enthalpy in kJ/mol.
    stderr
        Standard error in kJ/mol.
    """
    liquid_mean, liquid_stderr = block_estimate(
        liquid_pot_energy,
        block_size=block_size,
    )
    gas_mean, gas_stderr = block_estimate(
        gas_pot_energy,
        block_size=block_size,
    )
    volume_mean, volume_stderr = block_estimate(
        liquid_volume,
        block_size=block_size,
    )

    delta_h = (
        gas_mean
        - liquid_mean / n_molecules
        + units.kB * temperature
        - pressure * units.bar * volume_mean / n_molecules
    )

    stderr = np.sqrt(
        gas_stderr**2
        + (liquid_stderr / n_molecules) ** 2
        + (pressure * units.bar * volume_stderr / n_molecules) ** 2
    )

    return delta_h * EV_TO_KJ_MOL, stderr * EV_TO_KJ_MOL
