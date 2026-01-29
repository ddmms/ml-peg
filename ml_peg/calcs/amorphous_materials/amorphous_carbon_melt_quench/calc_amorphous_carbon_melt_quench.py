"""Run melt-quench simulations for amorphous carbon benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms, units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
from ase.io import write
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

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


def _melt(atoms: Atoms, steps: int) -> None:
    """
    Melt the structure at a fixed temperature.

    Parameters
    ----------
    atoms
        Atomic configuration (mutated in-place).
    steps
        Number of MD steps.
    """
    MaxwellBoltzmannDistribution(atoms, MELT_TEMP * units.kB)
    dyn = Langevin(
        atoms,
        timestep=DT_FS * units.fs,
        temperature=MELT_TEMP * units.kB,
        friction=FRICTION,
    )
    dyn.run(steps)


def _quench(atoms: Atoms, steps: int, temp_step: float) -> None:
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
    """
    dyn = Langevin(
        atoms,
        timestep=DT_FS * units.fs,
        temperature=MELT_TEMP * units.kB,
        friction=FRICTION,
    )

    for step in range(steps):
        dyn.run(1)
        target_temp = max(FINAL_TEMP, MELT_TEMP - temp_step * step)
        MaxwellBoltzmannDistribution(atoms, target_temp * units.kB)
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
    atoms = _build_diamond_supercell()
    _scale_to_density(atoms, density)

    atoms.calc = model.get_calculator()

    relaxer = LBFGS(atoms)
    relaxer.run(fmax=10.0)

    melt_steps = int(MELT_TIME_PS * 1000.0 / DT_FS)
    dt_ps = DT_FS * 1.0e-3
    temp_step = COOLING_RATE * dt_ps
    quench_steps = int((MELT_TEMP - FINAL_TEMP) / temp_step)

    _melt(atoms, melt_steps)
    _quench(atoms, quench_steps, temp_step)

    relaxer = LBFGS(atoms)
    relaxer.run(fmax=0.02)

    out_dir = OUT_PATH / model_name / f"density_{density:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / f"final_density_{density:.1f}.xyz", atoms)


@pytest.mark.parametrize("mlip", MODELS.items())
@pytest.mark.very_slow
def test_amorphous_carbon_melt_quench(mlip: tuple[str, object]) -> None:
    """
    Run amorphous carbon melt-quench benchmark for a single model.

    Parameters
    ----------
    mlip
        Tuple of model name and model object.
    """
    model_name, model = mlip
    for density in DENSITY_GRID:
        _run_density(model_name, model, density)
