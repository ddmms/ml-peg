"""Run calculations for HF/SbF5 tests."""

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
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


# Simulation parameters
TEMPERATURE_K = 288.6
PRESSURE_ATM = 1
DT_FS = 0.5  # DT in femtoseconds

N_MIN_STEPS = 100  # maximum minimization steps
N_NPT_STEPS = 200000  # NPT production steps

OUT_FREQ = 1000


# conversions
ATM_TO_GPA = 1.01325e-4  # 1 atm = 0.000101325 GPa
PRESSURE_AU = PRESSURE_ATM * ATM_TO_GPA * units.GPa
DT = 0.5 * units.fs
TDAMP = 100 * DT_FS * units.fs
PDAMP = 1000 * DT_FS * units.fs


@pytest.mark.parametrize("mlip", MODELS.items())
def test_hf_sbf5_density(mlip: tuple[str, Any]) -> None:
    """
    Run HF/SbF5 mixture density test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # download dataset
    hf_sbf5_density_dir = (
        download_s3_data(
            key="inputs/superacids/HF_SbF5_density/HF_SbF5_density.zip",
            filename="HF_SbF5_density.zip",
        )
        / "HF_SbF5_density"
    )

    with open(hf_sbf5_density_dir / "list") as f:
        systems = f.read().splitlines()

    for system in systems:
        print(f"simulando {system} with model {model_name}")

        atoms = read(hf_sbf5_density_dir / system / "start.xyz")
        atoms.calc = calc

        write_dir = OUT_PATH / model_name / system
        write_dir.mkdir(parents=True, exist_ok=True)

        # MINIMIZATIOn
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
