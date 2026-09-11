"""Regression checks for the calculation of thermodynamic quantities."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from ase import units
import numpy as np
import pytest

from ml_peg.analysis.molecular_dynamics.thermodynamic_properties import (
    analyse_liquid,
    read_property_from_log,
)
from ml_peg.analysis.utils.thermodynamics import (
    ANG3_PER_EV_TO_GPA_INV,
    AU_TO_G_L,
    EV_TO_J_MOL,
    EV_TO_KJ_MOL,
    MILLI,
    density,
    evaporation_enthalpy,
    heat_capacity_cp,
    isothermal_compressibility,
    thermal_expansion,
)
from ml_peg.calcs.molecular_dynamics.thermodynamic_properties.calc_properties import (
    log_md,
)

# Statistical/thermodynamic estimator tests


def test_density():
    """Test density calculation."""
    values = np.array([0.8, 1.0, 1.0, 1.2, 1.2, 1.4, 1.4, 1.6])
    mean, stderr = density(values, block_size=2)
    block_values = np.array([0.9, 1.1, 1.3, 1.5])
    assert mean == pytest.approx(np.mean(block_values))
    assert stderr == pytest.approx(
        np.std(block_values, ddof=1) / np.sqrt(len(block_values))
    )


def test_heat_capacity_cp():
    """Test constant pressure heat capacity calculation."""
    penergy = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0]) / 3.0
    kenergy = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0]) / 3.0
    vol = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0]) / units.bar / 3.0
    temperature, press, n_molecules = 300.0, 1.0, 10

    cp, stderr = heat_capacity_cp(
        pot_energy=penergy,
        kin_energy=kenergy,
        volume=vol,
        temperature=temperature,
        pressure=press,
        n_molecules=n_molecules,
        block_size=2,
    )

    # Every block has the same population variance: 1 eV^2.
    expected = 1.0 / (units.kB * temperature**2) * EV_TO_J_MOL / n_molecules
    assert cp == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)


def test_isothermal_compressibility():
    """Test isothermal compressibility calculation."""
    volume = np.array([9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0])
    temperature = 300.0
    kappa, stderr = isothermal_compressibility(
        volume, temperature=temperature, block_size=2
    )
    # variance = 1 Angstrom^6, mean volume = 10 Angstrom^3.
    mean_volume = 10.0
    expected = 1.0 / (units.kB * temperature * mean_volume) * ANG3_PER_EV_TO_GPA_INV
    assert kappa == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)


def test_thermal_expansion():
    """Test thermal expansion coefficient calculation."""
    volume = np.array([9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0])
    pot_energy, kin_energy = 2 * volume, 3 * volume
    temp, press = 300.0, 1.0
    alpha, stderr = thermal_expansion(
        pot_energy, kin_energy, volume, temperature=temp, pressure=press, block_size=2
    )
    mean_volume, mean_cov = 10.0, 5.0 + press * units.bar
    # cov(V, H) = cov(V, 2V) = 2 var(V) = 2.
    expected = mean_cov / (units.kB * temp**2 * mean_volume) / MILLI
    assert alpha == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)


def test_evaporation_enthalpy():
    """Test enthalpy of evaporation calculation."""
    liquid = np.array([-101.0, -99.0, -101.0, -99.0, -101.0, -99.0, -101.0, -99.0])
    volume = np.array([1001.0, 999.0, 1001.0, 999.0, 1001.0, 999.0, 1001.0, 999.0])

    gas = np.array([-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0])
    temp, press, n_mol = 300.0, 1.0, 10

    hvap, stderr = evaporation_enthalpy(
        liquid,
        gas,
        volume,
        pressure=press,
        temperature=temp,
        n_molecules=n_mol,
        block_size=2,
    )

    # Each liquid block has mean U -100 and every gas block mean U -4,
    # the volume mean is 1000.
    expected = (
        -4.0 - (-100.0 / 10 + 1.0 * units.bar * 1000.0 / 10.0) + units.kB * temp
    ) * EV_TO_KJ_MOL
    assert hvap == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)


# Analysis/log parsing tests


def write_test_log(path, phase):
    """
    Write a synthetic log for the liquid phase.

    Parameters
    ----------
    path
        Path to the log.
    phase
        The phase, 'liq' or 'gas'.
    """
    if phase == "liq":
        txt = [
            "t: 0.000 ps  Walltime:   76.114 s T: 300.0 K Epot:"
            + "  -673204.666 eV Ekin: 49.6358 eV"
            + "  volume: 2496.3 A^3 density: 809.0 g/L",
            "t: 0.001 ps  Walltime:  229.436 s T: 302.4 K Epot:"
            + "  -673207.534 eV Ekin: 50.0329 eV"
            + "  volume: 2480.9 A^3 density: 810.0 g/L",
            "t: 0.002 ps  Walltime:  306.222 s T: 306.2 K Epot:"
            + "  -673205.233 eV Ekin: 50.6616 eV"
            + "  volume: 2505.6 A^3 density: 802.0 g/L",
            "t: 0.003 ps  Walltime:  236.222 s T: 299.2 K Epot:"
            + "  -673206.667 eV Ekin: 49.5035 eV"
            + "  volume: 2490.2 A^3 density: 807.0 g/L",
        ]
    elif phase == "gas":
        txt = [
            "t: 0.000 ps  Walltime: 0.001 s T: 0.000 K Epot:"
            + " -5259.12744 eV Ekin: 0.0000 eV"
            + " volume: 1000000.0 A^3 density: 9.644 g/L",
            "t: 0.001 ps  Walltime: 0.002 s T: 200.0 K Epot:"
            + " -5259.12745 eV Ekin: 0.2585 eV"
            + " volume: 1000000.0 A^3 density: 9.644 g/L",
            "t: 0.002 ps  Walltime: 0.003 s T: 300.0 K Epot:"
            + " -5259.12743 eV Ekin: 0.3878 eV"
            + " volume: 1000000.0 A^3 density: 9.644 g/L",
            "t: 0.003 ps  Walltime: 0.004 s T: 298.0 K Epot:"
            + " -5259.12742 eV Ekin: 0.3852 eV"
            + " volume: 1000000.0 A^3 density: 9.644 g/L",
        ]
    else:
        raise ValueError("phase must be either 'liq' or 'gas'")

    path.write_text("\n".join(txt) + "\n")


def test_read_property_from_log(tmp_path):
    """
    Test reading thermodynamic properties from a log.

    Parameters
    ----------
    path
        Path to the log.
    """
    log_file = tmp_path / "test.log"
    write_test_log(log_file, "liq")

    volume, units = read_property_from_log(
        log_file,
        "volume",
        equil_time_ps=0.0,
    )

    assert np.allclose(volume, [2496.3, 2480.9, 2505.6, 2490.2])


def test_analyse_liquid(tmp_path):
    """
    Test analysing thermodynamic properties from a log.

    Parameters
    ----------
    path
        Path to the log.
    """
    log_file_liq = tmp_path / "test-liq.log"
    write_test_log(log_file_liq, "liq")
    log_file_gas = tmp_path / "test-gas.log"
    write_test_log(log_file_gas, "gas")

    results = analyse_liquid(
        log_file_liq=log_file_liq,
        log_file_gas=log_file_gas,
        temperature=300.0,
        pressure=1.0,
        n_molecules=128,
        equilib_time_ps=0.0,
        block_size=2,
    )

    assert set(results) == {
        "density",
        "cp",
        "compressibility",
        "evaporation_enthalpy",
        "alpha",
    }
    for value, stderr in results.values():
        assert np.isfinite(value)
        assert np.isfinite(stderr)

    assert results["density"][0] == pytest.approx(807.0)
    assert results["cp"][0] == pytest.approx(155.8059)
    assert results["compressibility"][0] == pytest.approx(0.00574133)
    assert results["evaporation_enthalpy"][0] == pytest.approx(30.92115)
    assert results["alpha"][0] == pytest.approx(0.50401255)


# Test the interface between calc and analysis


def test_log_md_matches_analysis_reader(tmp_path, caplog):
    """Test that MD logging provides all properties required by analysis."""

    class DummyAtoms:
        def get_potential_energy(self):
            return -12.51234

        def get_kinetic_energy(self):
            return 4.8123

        def get_volume(self):
            return 1000.0

        def get_temperature(self):
            return 300.0

        def get_masses(self):
            return np.array([1.0, 1.0])

    dyn = SimpleNamespace(
        atoms=DummyAtoms(),
        get_time=lambda: 500.0 * 1000.0 * units.fs,
    )

    with caplog.at_level(logging.INFO):
        log_md(dyn, start_time=0.0)

    log_file = tmp_path / "test.log"
    log_file.write_text(caplog.text)
    expected = {
        "Epot": (-12.51234, "eV"),
        "volume": (1000.0, "A^3"),
        "density": (2.0 / 1000.0 * AU_TO_G_L, "g/L"),
    }

    for property_name in ["Epot", "volume", "density"]:
        values, unit = read_property_from_log(
            log_file,
            property_name,
            equil_time_ps=0.0,
        )
        assert np.isclose(values[0], expected[property_name][0])
        assert len(values) == 1
        assert np.isfinite(values[0])
        assert unit is not None
