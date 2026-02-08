"""Run calculations for SSE RDF benchmark tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import Atoms, io, units
from ase.calculators.calculator import Calculator
from ase.io import Trajectory, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
import numpy as np
import pytest

# from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS: dict[str, Any] = load_models(models=current_models)

OUT_PATH: Path = Path(__file__).parent / "outputs"

# Benchmark parameters
NSTEPS: int = 1000
DELTA_T: float = 0.5
SEED: int = 0
BIN_SIZE: float = 0.05
FRAME_FREQUENCY: int = 5
N_EQUI_FRAMES: int = 20
TCHAIN: int = 10


def get_systems(data_dir: Path) -> list[tuple[Path, float, str]]:
    """
    Discover all SSE RDF systems from the extracted data directory.

    Walks the directory tree looking for POSCAR files under the structure
    ``{system}/stoichiometric/{temperature}K/POSCAR``.

    Parameters
    ----------
    data_dir
        Path to the top-level SSEs_data directory.

    Returns
    -------
    list[tuple[Path, float, str]]
        List of (poscar_dir, temperature, system_name) for each system.
    """
    systems: list[tuple[Path, float, str]] = []
    for poscar_file in sorted(data_dir.rglob(pattern="POSCAR")):
        temp_dir: Path = poscar_file.parent
        compound_dir: Path = temp_dir.parent.parent

        temperature = float(temp_dir.name.rstrip("K"))
        system_name = f"{compound_dir.name}_{temp_dir.parent.name}_{temp_dir.name}"
        systems.append((temp_dir, temperature, system_name))

    return systems


@pytest.mark.parametrize(argnames="mlip", argvalues=MODELS.items())
def test_rdf_benchmark(mlip: tuple[str, Any]) -> None:
    """
    Run SSE RDF benchmark test.

    Runs NVT molecular dynamics using a Nosé-Hoover chain thermostat
    for each system.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    """
    model_name, model = mlip
    calc: Calculator = model.get_calculator()

    timestep: float = DELTA_T * units.fs
    tdamp: float = 100 * timestep

    # TODO: switch back to S3 download for production
    # data_dir = (
    #     download_s3_data(
    #         key="inputs/electrolytes/SSE/SSEs_data.zip",
    #         filename="SSEs_data.zip",
    #     )
    #     / "SSEs_data"
    # )
    from ml_peg.calcs.utils.utils import extract_zip

    data_dir: Path = (
        extract_zip(filename=Path.home() / ".cache" / "ml-peg" / "SSEs_data.zip")
        / "SSEs_data"
    )

    systems: list[tuple[Path, int | float, str]] = get_systems(data_dir)

    for poscar_dir, temperature, system_name in systems:
        poscar_file: Path = poscar_dir / "POSCAR"
        atoms_initial: Atoms | list[Atoms] = io.read(
            filename=poscar_file, format="vasp"
        )

        atoms: Atoms = atoms_initial.copy()  # type: ignore[assignment]
        atoms.calc = copy(calc)

        rng = np.random.RandomState(seed=SEED)
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature, force_temp=True, rng=rng
        )
        Stationary(atoms)
        ZeroRotation(atoms)

        file_name = f"{system_name}_{model_name}"

        # Write output directory
        write_dir: Path = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        log_path: Path = write_dir / f"{file_name}.log"
        traj_path: Path = write_dir / f"{file_name}.traj"

        md_nvt = NoseHooverChainNVT(
            atoms=atoms,
            timestep=timestep,
            temperature_K=temperature,
            tdamp=tdamp,
            tchain=TCHAIN,
            logfile=str(log_path),
        )

        traj = Trajectory(filename=str(traj_path), mode="w", atoms=atoms)
        md_nvt.attach(function=traj.write, interval=10)

        md_nvt.run(steps=NSTEPS)

        # Read trajectory, skip equilibration frames and subsample
        ase_traj: Atoms | list[Atoms] = io.read(
            filename=str(traj_path), index=f"{N_EQUI_FRAMES}:"
        )[::FRAME_FREQUENCY]

        # Store metadata on each frame
        for frame in ase_traj:
            frame.info["system"] = system_name
            frame.info["temperature"] = temperature
            frame.info["delta_t"] = DELTA_T
            frame.info["nsteps"] = NSTEPS

        # Write trajectory frames as XYZ
        write(filename=write_dir / f"{file_name}.xyz", images=ase_traj)
