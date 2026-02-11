"""Run calculations for bulk benchmark."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import time
from typing import Any

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"


def run_md_water(
    start_config: Atoms,
    calc: Calculator,
    write_dir: Path,
    teqb: int = 25000,
    trun: int = 300000,
    th_dt: int = 1,
    md_dt: int = 1,
    timestep: float = 1,
    temperature: int = 330,
    friction: float = 0.05,
) -> None:
    """
    Run MD to study bulk water.

    Parameters
    ----------
    start_config : ase.Atoms
        Initial Atoms structure.
    calc : ase.calculators.calculator.Calculator
        Calculator to use to evaluate structure energy.
    write_dir : pathlib.Path
        Directory to write output files to.
    teqb : int, optional
        Total number of MD steps for equilibration run. Default is 5000.
    trun : int, optional
        Total number of MD steps to run. Default is 300000.
    th_dt : int, optional
        Logfile output interval in number of MD steps (md.thermo). Default is 1.
    md_dt : int, optional
        Coordinate output interval in number of MD steps (md.xyz). Default is 1.
    timestep : float, optional
        MD timestep in fs. Default is 1.
    temperature : int, optional
        MD temperature in K. Default is 330.
    friction : float, optional
        Friction coefficient for Langevin thermostat. Default is 0.05.
    """
    MaxwellBoltzmannDistribution(start_config, temperature_K=temperature)

    start_config.calc = calc
    start_config.pbc = [True, True, True]

    # Open coordinates file and thermo file
    thermo_traj = open(write_dir / "md.thermo", "w")
    coord_traj_name = write_dir / "md-pos.xyz"
    velc_traj_name = write_dir / "md-velc.xyz"
    # Remove any existing .xyz files to avoid appending to old data
    if coord_traj_name.exists():
        coord_traj_name.unlink()
    if velc_traj_name.exists():
        velc_traj_name.unlink()

    def print_traj(config=start_config):
        """
        Print trajectory information during MD.

        Parameters
        ----------
        config : ase.Atoms, optional
            Atoms configuration to print. Default is start_config.
        """
        calc_time = (dyn.get_time()) / units.fs
        calc_temp = config.get_temperature()
        calc_epot = config.get_potential_energy()
        calc_walltime = time.time() - glob_start_time

        if dyn.nsteps % th_dt == 0:
            thermo_traj.write(
                ("%17.2f" + " %17.6f" * 4 + "\n")
                % (calc_time, calc_temp, calc_epot, calc_walltime)
            )
            thermo_traj.flush()
        if dyn.nsteps % md_dt == 0:
            coords_write = Atoms(
                numbers=config.numbers,
                positions=config.get_positions(),
                cell=config.cell,
                pbc=config.pbc,
            )
            write(coord_traj_name, coords_write, append=True)
            velocities_write = Atoms(
                numbers=config.numbers,
                positions=config.get_velocities(),
                cell=config.cell,
                pbc=config.pbc,
            )
            write(velc_traj_name, velocities_write, append=True)

    thermo_traj.write(
        "# ASE Dynamics. Date: " + date.today().strftime("%d %b %Y") + "\n"
    )
    thermo_traj.write(
        "#   Time(fs)      Temperature(K)      Energy(eV)        Walltime(s)      \n"
    )

    open(coord_traj_name, "w").close()
    glob_start_time = time.time()

    # RUN MD WITH LANGEVIN THERMOSTAT
    dyn = Langevin(
        start_config,
        timestep * units.fs,
        temperature_K=temperature,
        friction=friction,
    )
    dyn.run(teqb)  # Equilibration run
    dyn.attach(print_traj)
    print_traj(start_config)
    dyn.run(trun)

    thermo_traj.close()


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bulk_water(mlip: tuple[str, Any]) -> None:
    """
    Run Bulk Water test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    # Setup calculator with d3 correction
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    # Get copper water benchmark data
    data_dir = (
        download_s3_data(
            filename="bulk_water.zip",
            key="inputs/aqueous-solutions/bulk_water/bulk_water.zip",
        )
        / "bulk_water"
    )

    # Load initial structure
    start_config = read(data_dir / "init.xyz")
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Run MD
    run_md_water(start_config, calc, write_dir=write_dir)
