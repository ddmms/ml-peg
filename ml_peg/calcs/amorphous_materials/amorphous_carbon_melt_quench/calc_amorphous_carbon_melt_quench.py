"""Run melt-quench simulations for amorphous carbon benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.optimize import LBFGS
from janus_core.calculations.md import NVT
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
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
QUENCH_TEMP_STEP = 100.0  # K
QUENCH_TEMP_TIME = 100.0  # fs, so 1000 K/ps cooling rate
DT_FS = 1.0  # fs
FRICTION = 0.001  # fs^-1
TRAJ_INTERVAL = 100  # write a trajectory frame every N MD steps


def _load_reference() -> dict[str, tuple[list[float], list[float]]]:
    """
    Download and load digitized DFT/Expt sp3-vs-density reference curves.

    Returns
    -------
    dict[str, tuple[list[float], list[float]]]
        Mapping of series ("DFT", "Expt.") to (densities, sp3 percentages),
        sorted by increasing density.
    """
    ref_dir = (
        download_s3_data(
            key="inputs/amorphous_materials/amorphous_carbon_melt_quench/"
            "amorphous_carbon_melt_quench.zip",
            filename="amorphous_carbon_melt_quench.zip",
        )
        / "amorphous_carbon_melt_quench"
    )
    curves: dict[str, tuple[list[float], list[float]]] = {}
    with open(ref_dir / "carbon_exp_dft_digitized.csv", newline="") as f:
        for row in csv.DictReader(f):
            dens, sp3 = curves.setdefault(row["series"], ([], []))
            dens.append(float(row["density_g_cm-3"]))
            sp3.append(float(row["sp3_count_percent"]))

    # Sort each series by density (required for np.interp in the analysis).
    for series, (dens, sp3) in curves.items():
        order = np.argsort(dens)
        curves[series] = ([dens[i] for i in order], [sp3[i] for i in order])
    return curves


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


def _melt(atoms: Atoms, out_dir: Path) -> None:
    """
    Melt the structure at a fixed temperature.

    Parameters
    ----------
    atoms
        Atomic configuration with calculator attached (mutated in-place).
    out_dir
        Directory for MD outputs.
    """
    NVT(
        struct=atoms,
        temp=MELT_TEMP,
        steps=int(MELT_TIME_PS * 1000.0 / DT_FS),
        timestep=DT_FS,
        friction=FRICTION,
        traj_every=TRAJ_INTERVAL,
        file_prefix=out_dir / "melt",
        enable_progress_bar=True,
    ).run()


def _quench(atoms: Atoms, out_dir: Path) -> None:
    """
    Quench the structure with a stepwise linear cooling ramp.

    Parameters
    ----------
    atoms
        Atomic configuration with calculator attached (mutated in-place).
    out_dir
        Directory for MD outputs.
    """
    NVT(
        struct=atoms,
        timestep=DT_FS,
        friction=FRICTION,
        temp_start=MELT_TEMP,
        temp_end=FINAL_TEMP,
        temp_step=QUENCH_TEMP_STEP,
        temp_time=QUENCH_TEMP_TIME,
        traj_every=TRAJ_INTERVAL,
        file_prefix=out_dir / "quench",
        enable_progress_bar=True,
    ).run()


def _run_density(
    model_name: str,
    model,
    density: float,
    reference: dict[str, tuple[list[float], list[float]]],
) -> None:
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
    reference
        DFT/Expt reference curves, embedded in the output for the analysis stage.
    """
    print(f"Starting {model_name} melt-quench at density {density:.1f} g/cm^3")

    atoms = _build_diamond_supercell()
    _scale_to_density(atoms, density)

    atoms.calc = model.get_calculator(precision="high")

    out_dir = OUT_PATH / model_name / f"density_{density:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        LBFGS(atoms).run(fmax=10.0, steps=500)
        _melt(atoms, out_dir)
        _quench(atoms, out_dir)
        LBFGS(atoms).run(fmax=0.02, steps=1000)
        finite = (
            np.all(np.isfinite(atoms.get_positions()))
            and np.isfinite(atoms.get_potential_energy())
            and np.all(np.isfinite(atoms.get_forces()))
        )
        if not finite:
            raise ValueError("non-finite positions, energy or forces after quench")
    except Exception as exc:
        warn(
            f"{model_name} melt-quench crashed at density {density:.1f}: {exc!r}; "
            "recording NaN.",
            stacklevel=2,
        )
        atoms.info["failed"] = True
        # Single line, as newlines in an info string corrupt the extxyz header
        atoms.info["failure"] = " ".join(f"{type(exc).__name__}: {exc}".split())

    atoms.calc = None
    atoms.info["ref_dft_density"] = np.array(reference["DFT"][0])
    atoms.info["ref_dft_sp3"] = np.array(reference["DFT"][1])
    atoms.info["ref_expt_density"] = np.array(reference["Expt."][0])
    atoms.info["ref_expt_sp3"] = np.array(reference["Expt."][1])
    write(out_dir / f"final_density_{density:.1f}.xyz", atoms)


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_amorphous_carbon_melt_quench(mlip: tuple[str, object]) -> None:
    """
    Run amorphous carbon melt-quench benchmark for a single model.

    Parameters
    ----------
    mlip
        Tuple of model name and model object.
    """
    model_name, model = mlip
    reference = _load_reference()
    for density in DENSITY_GRID:
        _run_density(model_name, model, density, reference)
