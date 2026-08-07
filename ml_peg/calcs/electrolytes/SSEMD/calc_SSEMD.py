"""Run calculations for SSE RDF benchmark tests."""

from __future__ import annotations

from collections.abc import Generator
from copy import copy
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms, io, units
from ase.calculators.calculator import Calculator
from ase.io import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS: dict[str, Any] = load_models(models=current_models)

OUT_PATH: Path = Path(__file__).parent / "outputs"

# Benchmark parameters
TOTAL_TIME_NS: float = 1.0  # ns
DELTA_T_FS: float = 0.5
SEED: int = 0
FRAME_FREQUENCY: int = 15
NSTEPS: int = int(TOTAL_TIME_NS * 1e6 / DELTA_T_FS)

EQUI_TIME_NS: float = 0.005  # 5 ps
N_EQUI_STEPS: int = int(EQUI_TIME_NS * 1e6 / DELTA_T_FS)
N_EQUI_FRAMES: int = N_EQUI_STEPS // FRAME_FREQUENCY
TCHAIN: int = 10

N_SYSTEMS: int = 49


def get_systems(data_dir: Path) -> Generator[tuple[Path, float, str], None, None]:
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
    Generator[tuple[Path, float, str], None, None]
        Generator yielding (poscar_dir, temperature, system_name) for each system.
    """
    for poscar_file in sorted(data_dir.rglob(pattern="POSCAR")):
        temp_dir: Path = poscar_file.parent
        compound_dir: Path = temp_dir.parent.parent

        temperature = float(temp_dir.name.rstrip("K"))
        system_name = f"{compound_dir.name}_{temp_dir.parent.name}_{temp_dir.name}"
        yield temp_dir, temperature, system_name


@pytest.mark.very_slow
@pytest.mark.parametrize(argnames="mlip", argvalues=MODELS.items())
def test_ssemd_benchmark(mlip: tuple[str, Any], system_id: int) -> None:
    """
    Run SSE RDF benchmark test.

    Runs NVT molecular dynamics using a Nosé-Hoover chain thermostat
    for each system.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    system_id
        Identifier of the SSE system to run MD on.
    """
    model_name, model = mlip
    calc: Calculator = model.get_calculator(precision="low")

    timestep: float = DELTA_T_FS * units.fs
    tdamp: float = 100 * timestep

    data_dir = (
        download_s3_data(
            key="inputs/electrolytes/SSEMD/SSEMD.zip",
            filename="SSEMD.zip",
        )
        / "SSEMD"
    )

    systems = list(get_systems(data_dir=data_dir))

    poscar_dir, temperature, system_name = systems[system_id]
    poscar_file: Path = poscar_dir / "POSCAR"

    file_name = f"{system_name}_{model_name}"

    # Write output directory
    write_dir: Path = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    log_path: Path = write_dir / f"{file_name}.log"
    traj_path: Path = write_dir / f"{file_name}.traj"

    # Restart if existing
    nsteps_done: int = 0
    if traj_path.exists():
        try:
            existing = Trajectory(filename=str(traj_path))
            atoms: Atoms = existing[-1]
            nsteps_done = (len(existing) - 1) * FRAME_FREQUENCY
            existing.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not restart from {traj_path}: {exc}")
            nsteps_done = 0

    if nsteps_done == 0:
        atoms_initial: Atoms | list[Atoms] = io.read(
            filename=poscar_file, format="vasp"
        )
        atoms = atoms_initial.copy()  # type: ignore[assignment]

        rng = np.random.RandomState(seed=SEED)
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature, force_temp=True, rng=rng
        )
        Stationary(atoms)
        ZeroRotation(atoms)

    # Set before the MD so the metadata is written out with every frame
    atoms.info.update(
        {
            "charge": 0,
            "spin": 1,
            "system": system_name,
            "temperature": temperature,
            "delta_t": DELTA_T_FS,
            "nsteps": NSTEPS,
        }
    )
    atoms.calc = copy(calc)

    md_nvt = NoseHooverChainNVT(
        atoms=atoms,
        timestep=timestep,
        temperature_K=temperature,
        tdamp=tdamp,
        tchain=TCHAIN,
        logfile=str(log_path),
        trajectory=str(traj_path),
        loginterval=FRAME_FREQUENCY,
        append_trajectory=True,
    )
    md_nvt.nsteps = nsteps_done

    try:
        md_nvt.run(steps=max(NSTEPS - nsteps_done, 0))
    except Exception as exc:
        warn(f"Error running MD: {exc}", stacklevel=2)
