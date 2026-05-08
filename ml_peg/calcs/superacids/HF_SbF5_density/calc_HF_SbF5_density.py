"""Run calculations for HF/SbF5 density tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
from ase.md.logger import MDLogger
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import FIRE
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Simulation parameters
TEMPERATURE_K = 288.6
PRESSURE_ATM = 1
DT_FS = 0.5  # DT in femtoseconds
N_MIN_STEPS = 1000  # maximum minimization steps
N_NPT_STEPS = 200000  # NPT production steps
OUT_FREQ = 1000

# Conversions
ATM_TO_GPA = 1.01325e-4  # 1 atm = 0.000101325 GPa
PRESSURE_AU = PRESSURE_ATM * ATM_TO_GPA * units.GPa
DT = 0.5 * units.fs
TDAMP = 100 * DT_FS * units.fs
PDAMP = 1000 * DT_FS * units.fs

# Systems
SYSTEMS = ["X_0", "X_10", "X_100"]


@pytest.mark.parametrize("mlip", MODELS.items(), ids=lambda x: x[0])
@pytest.mark.parametrize("system", SYSTEMS)
def test_hf_sbf5_density(mlip: tuple[str, Any], system: str) -> None:
    """
    Run HF/SbF5 mixture density test.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    system
        System identifier (X_0, X_10, X_100).
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # Download dataset
    hf_sbf5_density_dir = (
        download_s3_data(
            key="inputs/superacids/HF_SbF5_density/HF_SbF5_density.zip",
            filename="HF_SbF5_density.zip",
        )
        / "HF_SbF5_density"
    )

    print(f"Simulating {system} with model {model_name}")

    atoms = read(hf_sbf5_density_dir / system / "start.xyz")
    atoms.calc = calc

    write_dir = OUT_PATH / model_name / system
    write_dir.mkdir(parents=True, exist_ok=True)

    # Minimization
    opt = FIRE(atoms, logfile=str(write_dir / "opt.log"))
    opt.run(fmax=0.05, steps=N_MIN_STEPS)
    write(write_dir / "minimised.xyz", atoms)

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = IsotropicMTKNPT(
        atoms=atoms,
        timestep=DT,
        temperature_K=TEMPERATURE_K,
        pressure_au=PRESSURE_AU,
        tdamp=TDAMP,
        pdamp=PDAMP,
    )

    dyn.attach(
        MDLogger(dyn, atoms, str(write_dir / "md.log"), header=True, mode="w"),
        interval=OUT_FREQ,
    )

    # Volume logger: step and volume only
    vol_file = open(write_dir / "volume.dat", "w")
    vol_file.write("# step  volume_A3\n")

    def write_volume(_dyn=dyn, _atoms=atoms, _f=vol_file) -> None:
        """
        Write current step and volume to file.

        Parameters
        ----------
        _dyn : IsotropicMTKNPT
            The dynamics object.
        _atoms : Atoms
            The ASE atoms object.
        _f : TextIOWrapper
            The open file handle for volume data.
        """
        step = _dyn.nsteps
        vol = _atoms.get_volume()
        _f.write(f"{step}  {vol:.6f}\n")
        _f.flush()

    write_volume()  # step 0
    dyn.attach(write_volume, interval=OUT_FREQ)

    # Run NPT
    dyn.run(N_NPT_STEPS)

    vol_file.close()

    # Save final structure
    atoms.info["system"] = system
    write(write_dir / f"{system}.xyz", atoms)

    print(f"  {system} done")
