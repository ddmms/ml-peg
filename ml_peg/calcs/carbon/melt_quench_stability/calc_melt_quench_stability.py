"""Run carbon melt-quench stability simulations."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from janus_core.calculations.md import NVT_CSVR
import numpy as np
import pytest

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

CELL_REPETITIONS = 6
N_ATOMS = CELL_REPETITIONS**3
DENSITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)

# Integer counts nearest to the target C, C85H15, C75H15N10 and C75H15O10 ratios.
COMPOSITIONS = {
    "C": {"C": 216},
    "CH": {"C": 184, "H": 32},
    "CHN": {"C": 162, "H": 32, "N": 22},
    "CHO": {"C": 162, "H": 32, "O": 22},
}

BASE_SEED = 10
FS_PER_PS = 1000.0
INITIAL_TEMPERATURE_K = 100.0
MELT_TEMPERATURE_K = 8000.0
FINAL_TEMPERATURE_K = 300.0
TIMESTEP_PS = 0.0005
MELT_TIME_PS = 5.0
QUENCH_TIME_PS = 8.0
MELT_STEPS = round(MELT_TIME_PS / TIMESTEP_PS)
QUENCH_STEPS = round(QUENCH_TIME_PS / TIMESTEP_PS)
TOTAL_STEPS = MELT_STEPS + QUENCH_STEPS

THERMOSTAT_TIME_PS = 0.1
STATS_EVERY = 100
TRAJ_EVERY = 1000
CHECK_EVERY = 10
MAX_TEMPERATURE_K = 10 * MELT_TEMPERATURE_K

EXPECTED_OUTPUT_FILES = (
    "initial.extxyz",
    "melt-final.extxyz",
    "melt-stats.dat",
    "melt-traj.extxyz",
    "quench-final.extxyz",
    "quench-stats.dat",
    "quench-traj.extxyz",
)


def build_structure(
    composition: str,
    density: float,
    seed: int,
) -> Atoms:
    """
    Build a randomized simple-cubic carbon-material structure.

    Parameters
    ----------
    composition
        Key in ``COMPOSITIONS``.
    density
        Target mass density in g cm^-3.
    seed
        Random seed for species placement and atomic displacements.

    Returns
    -------
    ase.Atoms
        Periodic 6 x 6 x 6 simple-cubic structure.
    """
    if composition not in COMPOSITIONS:
        raise ValueError(f"Unknown composition: {composition}")
    if density <= 0:
        raise ValueError("Density must be positive")

    counts = COMPOSITIONS[composition]
    if sum(counts.values()) != N_ATOMS:
        raise ValueError(f"{composition} must contain {N_ATOMS} atoms")

    symbols = [symbol for symbol, count in counts.items() for _ in range(count)]
    rng = np.random.default_rng(seed)
    rng.shuffle(symbols)

    atoms = Atoms(symbols)
    volume = atoms.get_masses().sum() * 1e24 / (units.mol * density)
    cell_length = volume ** (1 / 3)
    spacing = cell_length / CELL_REPETITIONS

    grid = np.indices((CELL_REPETITIONS,) * 3).reshape(3, -1).T
    displacement = rng.uniform(-0.05 * spacing, 0.05 * spacing, grid.shape)
    atoms.set_positions(grid * spacing + displacement)
    atoms.set_cell([cell_length] * 3)
    atoms.set_pbc(True)
    atoms.wrap()
    atoms.info.update(
        {
            "composition": composition,
            "density_g_cm3": density,
            "seed": seed,
        }
    )
    return atoms


def check_md_state(atoms: Atoms) -> None:
    """
    Raise if an MD configuration has exploded numerically.

    Parameters
    ----------
    atoms
        Current MD configuration.

    Raises
    ------
    RuntimeError
        If positions, energy, forces or temperature are non-finite, or the
        instantaneous temperature exceeds ten times the melt temperature.
    """
    positions = atoms.get_positions()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    temperature = atoms.get_temperature()

    values_are_finite = (
        np.all(np.isfinite(positions))
        and np.isfinite(energy)
        and np.all(np.isfinite(forces))
        and np.isfinite(temperature)
    )
    if not values_are_finite:
        raise RuntimeError("MD state contains non-finite values")
    if temperature > MAX_TEMPERATURE_K:
        raise RuntimeError(
            f"MD temperature {temperature:.0f} K exceeds {MAX_TEMPERATURE_K:.0f} K"
        )


def _set_quench_temperature(md: NVT_CSVR) -> None:
    """
    Update the CSVR target temperature along the linear quench.

    Parameters
    ----------
    md
        Janus CSVR dynamics object.
    """
    progress = min(md.dyn.nsteps / QUENCH_STEPS, 1.0)
    temperature = MELT_TEMPERATURE_K + progress * (
        FINAL_TEMPERATURE_K - MELT_TEMPERATURE_K
    )
    md.temp = temperature
    # Janus' built-in ramp resets velocities at each temperature. Updating its
    # target directly preserves a continuous trajectory for this stability test.
    md._set_target_temperature(temperature)


def _run_stage(md: NVT_CSVR) -> tuple[int, str | None]:
    """
    Run one MD stage.

    Parameters
    ----------
    md
        Janus CSVR dynamics object.

    Returns
    -------
    tuple[int, str | None]
        Completed steps and any failure message.
    """
    md.dyn.attach(check_md_state, interval=CHECK_EVERY, atoms=md.struct)
    try:
        md.run()
    except Exception as exc:
        return md.dyn.nsteps, f"{type(exc).__name__}: {exc}"
    return md.dyn.nsteps, None


def _outputs_complete(output_dir: Path) -> bool:
    """
    Check whether all outputs for one trajectory are present.

    Parameters
    ----------
    output_dir
        Composition-density output directory.

    Returns
    -------
    bool
        Whether every expected output exists and is non-empty.
    """
    return all(
        (path := output_dir / filename).is_file() and path.stat().st_size > 0
        for filename in EXPECTED_OUTPUT_FILES
    )


def _load_previous_results(status_path: Path) -> dict[tuple[str, float], dict]:
    """
    Load existing status records by composition and density.

    Parameters
    ----------
    status_path
        Per-model status file.

    Returns
    -------
    dict[tuple[str, float], dict]
        Existing records keyed by composition and density.
    """
    if not status_path.exists():
        return {}
    try:
        records = json.loads(status_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(records, list):
        return {}
    return {
        (record["composition"], float(record["density_g_cm3"])): record
        for record in records
        if "composition" in record and "density_g_cm3" in record
    }


def run_trajectory(
    atoms: Atoms,
    output_dir: Path,
    seed: int,
) -> dict[str, int | float | bool | str | None]:
    """
    Run the melt and linear quench for one starting structure.

    Parameters
    ----------
    atoms
        Starting structure with an attached calculator.
    output_dir
        Directory for Janus trajectory and statistics files.
    seed
        Random seed for the initial velocities and thermostat.

    Returns
    -------
    dict
        Stability status and the last completed MD step.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    write(output_dir / "initial.extxyz", atoms)

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=INITIAL_TEMPERATURE_K, rng=rng)
    Stationary(atoms, preserve_temperature=True)

    melt = NVT_CSVR(
        struct=atoms,
        temp=MELT_TEMPERATURE_K,
        steps=MELT_STEPS,
        timestep=TIMESTEP_PS * FS_PER_PS,
        taut=THERMOSTAT_TIME_PS * FS_PER_PS,
        stats_every=STATS_EVERY,
        traj_every=TRAJ_EVERY,
        restart_every=TOTAL_STEPS + 1,
        file_prefix=output_dir / "melt",
        seed=seed,
    )
    melt_steps, failure = _run_stage(melt)
    if failure is not None:
        return {
            "stable": False,
            "completed_steps": melt_steps,
            "completed_time_ps": melt_steps * TIMESTEP_PS,
            "failure": failure,
        }

    quench = NVT_CSVR(
        struct=melt.struct,
        temp=MELT_TEMPERATURE_K,
        steps=QUENCH_STEPS,
        timestep=TIMESTEP_PS * FS_PER_PS,
        taut=THERMOSTAT_TIME_PS * FS_PER_PS,
        stats_every=STATS_EVERY,
        traj_every=TRAJ_EVERY,
        restart_every=TOTAL_STEPS + 1,
        file_prefix=output_dir / "quench",
        seed=seed,
    )
    quench.dyn.attach(_set_quench_temperature, interval=1, md=quench)
    quench_steps, failure = _run_stage(quench)
    completed_steps = MELT_STEPS + quench_steps
    return {
        "stable": failure is None and completed_steps == TOTAL_STEPS,
        "completed_steps": completed_steps,
        "completed_time_ps": completed_steps * TIMESTEP_PS,
        "failure": failure,
    }


def run_benchmark(model_name: str, model: Any) -> None:
    """
    Run all composition-density trajectories for one model.

    Parameters
    ----------
    model_name
        Registered model name.
    model
        Model wrapper used to construct the calculator.
    """
    model_dir = OUT_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    status_path = model_dir / "status.json"
    previous_results = _load_previous_results(status_path)
    results = []
    status_path.write_text("[]\n")
    calc = None

    for composition_index, composition in enumerate(COMPOSITIONS):
        seed = BASE_SEED + composition_index
        for density in DENSITIES:
            run_name = f"rho-{density:g}".replace(".", "p")
            output_dir = model_dir / composition / run_name
            result = None
            if _outputs_complete(output_dir):
                result = previous_results.get((composition, density), {}).copy()
                result.setdefault("stable", True)
                result.setdefault("completed_steps", TOTAL_STEPS)
                result.setdefault("completed_time_ps", TOTAL_STEPS * TIMESTEP_PS)
                result.setdefault("failure", None)
                result.setdefault("walltime_seconds", None)
                print(
                    f"Skipping {model_name} {composition} at {density:g} g cm^-3: "
                    "outputs found"
                )
            else:
                if calc is None:
                    calc = model.get_calculator(precision="low")
                    calc = model.add_d3_calculator(calc)
                atoms = build_structure(composition, density, seed)
                atoms.calc = calc
                start_time = perf_counter()
                try:
                    result = run_trajectory(atoms, output_dir, seed)
                except Exception as exc:
                    result = {
                        "stable": False,
                        "completed_steps": 0,
                        "completed_time_ps": 0.0,
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                result["walltime_seconds"] = perf_counter() - start_time
            result.update(
                {
                    "composition": composition,
                    "density_g_cm3": density,
                    "seed": seed,
                    "run_name": run_name,
                }
            )
            results.append(result)
            status_path.write_text(json.dumps(results, indent=2) + "\n")
            if not result["stable"]:
                warn(
                    f"{model_name} failed for {composition} at {density:g} g cm^-3: "
                    f"{result['failure']}",
                    stacklevel=2,
                )


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_melt_quench_stability(mlip: tuple[str, Any]) -> None:
    """
    Run the carbon melt-quench stability benchmark.

    Parameters
    ----------
    mlip
        Tuple of model name and model wrapper.
    """
    run_benchmark(*mlip)
