"""Run calculations for water slab dipole tests."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import numpy as np
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


@pytest.mark.parametrize("mlip", MODELS.items())
def test_water_dipole(mlip: tuple[str, Any]) -> None:
    """
    Run water dipole test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test (for models where applicable)
    calc = model.add_d3_calculator(calc)

    out_name = "slab"

    md_t = 200  # 200000 # number of timesteps, here dt = 1fs
    md_dt = 10  # 500  # intervals for printing energy, T, etc
    th_dt = 10  # 500  # intervals for printing structures
    temp = 300  # Kelvin
    pres = 1.013  # bar

    ttime = 100 * units.fs  # timescale themostat

    print("Reading start_config from ", DATA_PATH / "init_38A_slab.xyz")
    start_config = read(DATA_PATH / "init_38A_slab.xyz", "-1")
    start_config.set_cell(np.triu(start_config.get_cell()))  # why?
    start_config.info["charge"] = 0.0
    start_config.info["spin"] = 1

    start_config.calc = calc

    velocities = start_config.get_velocities()
    start_config.set_velocities(velocities)
    MaxwellBoltzmannDistribution(start_config, temperature_K=300)

    md = NPT(
        atoms=start_config,
        timestep=1 * units.fs,
        temperature_K=temp,
        externalstress=pres * units.bar,
        ttime=ttime,
        pfactor=None,
    )

    # start_config_positions = start_config.positions.copy()
    # start_config_com = start_config.get_center_of_mass().copy()

    # Write output structures
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    print("Writing to ", write_dir)

    thermo_traj = open(write_dir / (out_name + ".thermo"), "w")  # file for output
    coord_traj_name = write_dir / (out_name + ".xyz")  # file for coordinate output

    def print_traj(a=start_config):
        """
        Output trajectory (xyz in coord_traj_name, steps, T, and E in thermo_traj).

        Parameters
        ----------
        a
            Structure to evaluate. Defaults to start_config.
        """
        calc_time = md.get_time() / units.fs
        calc_temp = a.get_temperature()
        # calc_dens = np.sum(a.get_masses())/a.get_volume()*densfact
        # calc_pres = - \
        #    np.trace(a.get_stress(
        #        include_ideal_gas=True, voigt=False))/3/units.bar
        calc_epot = a.get_potential_energy()
        # calc_msd = (((a.positions-a.get_center_of_mass()) -
        #            (start_config_positions-start_config_com))**2).mean(0).sum(0)
        # calc_drft = ((a.get_center_of_mass()-start_config_com)**2).sum(0)
        # calc_tens = -a.get_stress(include_ideal_gas=True, voigt=True)/units.bar
        if md.nsteps % th_dt == 0:
            thermo_traj.write(
                ("%12d" + " %17.6f" * 2 + "\n") % (calc_time, calc_temp, calc_epot)
            )
            thermo_traj.flush()
        if md.nsteps % md_dt == 0:
            print(f"Step {md.nsteps}")
            write(coord_traj_name, a, append=True)

    # print_traj could also be done using MDLogger and write_traj

    thermo_traj.write(
        "# ASE Dynamics. Date: " + date.today().strftime("%d %b %Y") + ", started: " + datetime.now().strftime("%H:%M:%S") + "\n"
    )
    thermo_traj.write("#   Time(fs)      Temperature(K)       Energy(eV)  \n")
    open(coord_traj_name, "w").close()
    print_traj(start_config)

    print("Starting MD")
    md.attach(print_traj)
    md.run(md_t)

    thermo_traj.write(
            "# Date: " + date.today().strftime("%d %b %Y") + ", finished: " + datetime.now().strftime("%H:%M:%S") + "\n"
    )
    thermo_traj.close()
