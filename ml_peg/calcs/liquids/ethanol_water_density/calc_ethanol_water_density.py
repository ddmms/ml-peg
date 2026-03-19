"""Calculate ethanol-water density curves."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any

from ase import units
from ase.io import Trajectory, read
from ase.md import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

BENCH_ROOT = Path(__file__).resolve().parent
OUT_PATH = BENCH_ROOT / "outputs"

MODELS = load_models(current_models)
MODEL_INDEX = {name: i for i, name in enumerate(MODELS)}

DATA_PATH = BENCH_ROOT / "data"

NUM_NPT_STEPS = 1000_000
NUM_NVT_STEPS = 50_000
TIMESTEP = 1 * units.fs
LOG_INTERVAL = 100
ATM = 1.01325 * units.bar
TEMPERATURE = 298.15
LANGEVIN_FRICTION = 1 / (500 * units.fs)


@dataclass(frozen=True)
class CompositionCase:
    """
    Map composition value to structure filename.

    Attributes
    ----------
    x_ethanol : float
        Ethanol mole fraction for the case.
    filename : str
        Structure filename associated with the composition.
    """

    x_ethanol: float
    filename: str


def load_compositions() -> list[CompositionCase]:
    """
    Load composition grid from ``compositions.csv``.

    Returns
    -------
    list[CompositionCase]
        Parsed composition cases ordered as in the CSV file.
    """
    comps_file = DATA_PATH / "compositions.csv"
    cases: list[CompositionCase] = []
    with comps_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                CompositionCase(
                    x_ethanol=float(row["x_ethanol"]),
                    filename=row["filename"],
                )
            )
    if not cases:
        raise RuntimeError("No compositions found in compositions.csv")
    return cases


COMPOSITIONS = load_compositions()


def run_one_case(
    struct_path: str | Path,
    calc: Any,
    output_fname: str | Path,
):
    """
    Run the full MD workflow for one composition case.

    Parameters
    ----------
    struct_path : str | pathlib.Path
        Input structure path.
    calc : Any
        ASE-compatible calculator.
    output_fname : str | pathlib.Path
        The output file path.

    Returns
    -------
    collections.abc.Iterable[float]
        Density time series in g/cm^3.
    """
    atoms = read(struct_path)
    atoms.set_pbc(True)
    atoms.wrap()
    atoms.calc = calc

    # velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
    Stationary(atoms)
    ZeroRotation(atoms)
    if not os.path.exists(output_fname):
        # NVT
        dyn = Langevin(
            atoms,
            timestep=TIMESTEP,
            temperature_K=TEMPERATURE,
            friction=LANGEVIN_FRICTION,
        )
        dyn.run(NUM_NVT_STEPS)

    # NPT
    run_npt(atoms, calc, output_fname)


def get_density_g_cm3(atoms):
    """
    Return density in g/cm^3.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration with periodic cell volume.

    Returns
    -------
    float
        Density in g/cm^3.
    """
    amu_to_kg = 1.66053906660e-27
    v_a3 = atoms.get_volume()
    v_m3 = v_a3 * 1e-30
    m_kg = atoms.get_masses().sum() * amu_to_kg
    rho_kg_m3 = m_kg / v_m3
    return rho_kg_m3 / 1000.0


def log_md(dyn, start_time):
    """
    Log molecular dynamics simulation.

    Parameters
    ----------
    dyn
        ASE molecular dynamics object.
    start_time
        Real time of the simulation start, in seconds.
    """
    current_time = time.time() - start_time
    energy = dyn.atoms.get_potential_energy()
    density = get_density_g_cm3(dyn.atoms)
    temperature = dyn.atoms.get_temperature()
    t = dyn.get_time() / (1000 * units.fs)
    logging.info(
        f"""t: {t:>8.3f} ps\
            Walltime: {current_time:>10.3f} s\
            T: {temperature:.1f} K\
            Epot: {energy:.2f} eV\
            density: {density:.3f} g/cm^3\
        """
    )


def run_npt(atoms, calc, output_fname):
    """
    Run NPT molecular dynamics using the isotropic MTK barostat.

    Parameters
    ----------
    atoms
        ASE Atoms of the system.
    calc
        ASE Calculator.
    output_fname
        File name to save the trajectory to.
    """
    if os.path.exists(output_fname):
        try:
            traj = Trajectory(output_fname)
            atoms = traj[-1]
            nsteps = (len(traj) - 1) * LOG_INTERVAL
        except Exception as e:
            print(e)
            nsteps = 0
    else:
        nsteps = 0

    atoms.calc = calc

    dyn = IsotropicMTKNPT(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=TEMPERATURE,
        pressure_au=ATM,
        tdamp=50 * units.fs,
        pdamp=500 * units.fs,
        trajectory=output_fname,
        loginterval=LOG_INTERVAL,
        append_trajectory=True,
    )
    dyn.nsteps = nsteps
    dyn.attach(log_md, interval=LOG_INTERVAL, dyn=dyn, start_time=time.time())
    dyn.run(steps=NUM_NPT_STEPS)


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items(), ids=list(MODELS.keys()))
def test_water_ethanol_density_curves(mlip: tuple[str, Any]) -> None:
    """
    Generate one density-curve case for a model and composition.

    Parameters
    ----------
    mlip : tuple[str, Any]
        Pair of model name and model object.

    Returns
    -------
    None
        This test writes output files for a single case.
    """
    for case in COMPOSITIONS:
        water_ethanol_density_curve_one_case(mlip, case)


def water_ethanol_density_curve_one_case(mlip: tuple[str, Any], case) -> None:
    """
    Run one MD simulation case and write its density time series.

    Parameters
    ----------
    mlip : tuple[str, Any]
        Pair of model name and model object.
    case : Any
        Composition case containing ``x_ethanol`` and ``filename``.

    Returns
    -------
    None
        This function writes outputs for one composition.
    """
    model_name, model = mlip

    model_out = OUT_PATH / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    # TODO: the data downloading thing here

    struct_path = DATA_PATH / case.filename
    if not struct_path.exists():
        raise FileNotFoundError(
            f"Missing structure for x={case.x_ethanol}: {struct_path}"
        )

    case_dir = model_out / f"x_ethanol_{case.x_ethanol:.2f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=case_dir / f"{model_name}.log",
        filemode="a",
        force=True,
    )
    run_one_case(struct_path, calc, case_dir / f"{model_name}.traj")


if __name__ == "__main__":  # TODO: delete this
    # run a very small simulation to see if it does something reasonable
    from ase import units
    from mace.calculators import mace_mp

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    calc = mace_mp(
        "data_old/mace-omat-0-small.model",
        dispersion=True,
        dispersion_cutoff=25 * units.Bohr,
    )
    run_one_case(
        "data/mix_xe_0.00.extxyz",
        calc,
        output_fname="debug/whatever.traj",
    )
