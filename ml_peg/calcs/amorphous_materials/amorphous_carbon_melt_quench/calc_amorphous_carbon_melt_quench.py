"""Run melt-quench simulations for amorphous carbon benchmark."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase import Atoms, units
from ase.build import bulk
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

# Local directory for calculator outputs
OUT_PATH = Path(__file__).parent / "outputs"

# Benchmark settings
DENSITY_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]  # g/cm^3
SUPERCELL = (3, 3, 3)
DIAMOND_LATTICE = 3.5  # Angstrom
MELT_TEMP = 8000.0  # K
FINAL_TEMP = 300.0  # K
MELT_TIME_PS = 3.0  # ps
COOLING_RATE = 1000.0  # K/ps
DT_FS = 1.0  # fs
FRICTION = 0.001 / units.fs
TRAJ_INTERVAL = 100  # write a trajectory frame every N MD steps


def _build_diamond_supercell() -> Atoms:
    """
    Build a diamond supercell as a starting point.

    Returns
    -------
    Atoms
        Diamond structure replicated into a supercell.
    """
    atoms = bulk("C", "diamond", a=DIAMOND_LATTICE, cubic=True)
    atoms *= SUPERCELL
    atoms.set_pbc(True)
    return atoms


def _density_g_cm3(atoms: Atoms) -> float:
    """
    Calculate density in g/cm^3.

    Parameters
    ----------
    atoms
        Atomic configuration.

    Returns
    -------
    float
        Density in g/cm^3.
    """
    mass_amu = atoms.get_masses().sum()
    mass_g = mass_amu * 1.66053906660e-24
    volume_cm3 = atoms.get_volume() * 1e-24
    return float(mass_g / volume_cm3)


def _scale_to_density(atoms: Atoms, target_density: float) -> None:
    """
    Isotropically scale the cell to match the target density.

    Parameters
    ----------
    atoms
        Atomic configuration (mutated in-place).
    target_density
        Target density in g/cm^3.
    """
    current_density = _density_g_cm3(atoms)
    scale = (current_density / target_density) ** (1.0 / 3.0)
    atoms.set_cell(atoms.cell * scale, scale_atoms=True)


def _attach_trajectory_writer(dyn, atoms: Atoms, traj_path: Path) -> None:
    """
    Append the current structure to a trajectory every ``TRAJ_INTERVAL`` steps.

    Parameters
    ----------
    dyn
        MD dynamics object to attach the writer to.
    atoms
        Atomic configuration being propagated.
    traj_path
        Trajectory file to append snapshots to.
    """

    def _snapshot() -> None:
        """Append the current structure to the trajectory file."""
        frame = atoms.copy()
        frame.calc = None
        write(traj_path, frame, append=True, format="extxyz")

    dyn.attach(_snapshot, interval=TRAJ_INTERVAL)


def _melt(atoms: Atoms, steps: int, traj_path: Path) -> None:
    """
    Melt the structure at a fixed temperature.

    Parameters
    ----------
    atoms
        Atomic configuration (mutated in-place).
    steps
        Number of MD steps.
    traj_path
        Trajectory file to append snapshots to during the run.
    """
    MaxwellBoltzmannDistribution(atoms, temperature_K=MELT_TEMP)
    dyn = Langevin(
        atoms,
        timestep=DT_FS * units.fs,
        temperature=MELT_TEMP * units.kB,
        friction=FRICTION,
    )
    pbar = tqdm(total=steps, desc="melt", leave=False)
    dyn.attach(pbar.update, interval=1)
    _attach_trajectory_writer(dyn, atoms, traj_path)
    dyn.run(steps)
    pbar.close()


def _quench(atoms: Atoms, steps: int, temp_step: float, traj_path: Path) -> None:
    """
    Quench the structure by linearly reducing the temperature.

    Parameters
    ----------
    atoms
        Atomic configuration (mutated in-place).
    steps
        Number of MD steps.
    temp_step
        Temperature decrement per step (K).
    traj_path
        Trajectory file to append snapshots to during the run.
    """
    dyn = Langevin(
        atoms,
        timestep=DT_FS * units.fs,
        temperature=MELT_TEMP * units.kB,
        friction=FRICTION,
    )
    _attach_trajectory_writer(dyn, atoms, traj_path)

    for step in tqdm(range(steps), desc="quench", leave=False):
        dyn.run(1)
        target_temp = max(FINAL_TEMP, MELT_TEMP - temp_step * step)
        MaxwellBoltzmannDistribution(atoms, temperature_K=target_temp)
        current_temp = atoms.get_temperature()
        if current_temp > 0:
            atoms.set_momenta(atoms.get_momenta() * (target_temp / current_temp) ** 0.5)


def _run_density(model_name: str, model, density: float) -> None:
    """
    Run melt-quench simulation for a single density.

    Parameters
    ----------
    model_name
        Name of MLIP model.
    model
        Model wrapper exposing ``get_calculator``.
    density
        Target density in g/cm^3.

    Returns
    -------
    QuenchStats
        Summary metrics for this density.
    """
    print(f"Starting {model_name} melt-quench at density {density:.1f} g/cm^3")

    atoms = _build_diamond_supercell()
    _scale_to_density(atoms, density)

    atoms.calc = model.get_calculator()

    out_dir = OUT_PATH / model_name / f"density_{density:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"trajectory_density_{density:.1f}.extxyz"
    traj_path.unlink(missing_ok=True)

    melt_steps = int(MELT_TIME_PS * 1000.0 / DT_FS)
    dt_ps = DT_FS * 1.0e-3
    temp_step = COOLING_RATE * dt_ps
    quench_steps = int((MELT_TEMP - FINAL_TEMP) / temp_step)

    try:
        LBFGS(atoms).run(fmax=10.0, steps=500)
        _melt(atoms, melt_steps, traj_path)
        _quench(atoms, quench_steps, temp_step, traj_path)
        LBFGS(atoms).run(fmax=0.02, steps=1000)
        if not np.all(np.isfinite(atoms.get_positions())):
            raise ValueError("non-finite positions after quench")
    except Exception as exc:
        warn(
            f"{model_name} melt-quench crashed at density {density:.1f}: {exc!r}; "
            "recording NaN.",
            stacklevel=2,
        )
        atoms.info["failed"] = True

    atoms.calc = None
    write(out_dir / f"final_density_{density:.1f}.xyz", atoms)


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_amorphous_carbon_melt_quench(mlip: tuple[str, object], density_index) -> None:
    """
    Run amorphous carbon melt-quench benchmark for a single model and density.

    Parameters
    ----------
    mlip
        Tuple of model name and model object.
    density_index
        Index into ``DENSITY_GRID`` of the density to run.
    """
    assert density_index in range(len(DENSITY_GRID)), (
        f"density_index out of range: use 0 to {len(DENSITY_GRID) - 1}"
    )
    model_name, model = mlip
    _run_density(model_name, model, DENSITY_GRID[density_index])
