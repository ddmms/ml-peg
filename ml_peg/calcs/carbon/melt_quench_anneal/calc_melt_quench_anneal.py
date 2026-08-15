"""Run carbon melt-quench-anneal simulations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from janus_core.calculations.md import NVT_CSVR
import numpy as np
import pytest

from ml_peg.calcs.carbon.melt_quench_anneal.structure_utils import (
    RELAX_FMAX,
    RELAX_STEPS,
    clean_and_relax,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Side length of the simple-cubic starting structure, in unit cells.
N_CELL_SIZE = 6
N_ATOMS = N_CELL_SIZE**3
DENSITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)

# Atomic percentages of the non-carbon species in each target material, matching
# the reference C and C75H15O10 material configs. Carbon takes the remainder.
# Compositions are the outermost loop of `RUNS`, so new ones must be appended to
# keep the scheduler array indices of existing compositions stable.
COMPOSITION_PERCENTAGES = {
    "C": {},
    "CHO": {"H": 15.0, "O": 10.0},
    "CH": {"H": 15.0},
}

# Independent repeats per composition-density pair, each from its own random
# starting structure and initial velocities. Runs are numbered from 1.
N_RUNS = 5
BASE_SEED = 10
FS_PER_PS = 1000.0
INITIAL_TEMPERATURE_K = 100.0
MELT_TEMPERATURE_K = 8000.0
ANNEAL_TEMPERATURE_K = 3000.0
FINAL_TEMPERATURE_K = 300.0

# Melt and quench use a 0.5 fs timestep; the anneal stages use 1 fs, following
# the MACE-MP paper.
MELT_TIMESTEP_FS = 0.5
ANNEAL_TIMESTEP_FS = 1.0

MELT_TIME_PS = 5.0
# Quench at 1000 K ps^-1, as in the reference LAMMPS protocol.
QUENCH_TIME_PS = ceil((MELT_TEMPERATURE_K - FINAL_TEMPERATURE_K) / 1000)
EQUILIBRATE_TIME_PS = 4.0
ANNEAL_TIME_PS = 350.0
COOL_TIME_PS = 10.0

THERMOSTAT_TIME_PS = 0.1
STATS_EVERY = 100
TRAJ_EVERY = 1000
CHECK_EVERY = 10
MAX_TEMPERATURE_K = 10 * MELT_TEMPERATURE_K

# Stages longer than this are split into equal chunks, each writing its own final
# structure. A job killed at its scheduler walltime then loses at most one chunk
# rather than the whole stage, since completed chunks are skipped on resubmission.
MAX_CHUNK_TIME_PS = 25.0

# Stages whose final structure is cleaned and relaxed after the MD. Chunked
# stages are relaxed at their last chunk only.
RELAX_STAGE_NAMES = ("anneal", "cool")


def composition_counts(composition: str, n_atoms: int | None = None) -> dict[str, int]:
    """
    Get integer species counts for a target material.

    Non-carbon counts are rounded from their atomic percentages, as in the
    reference LAMMPS protocol, and carbon takes the remainder.

    Parameters
    ----------
    composition
        Key in ``COMPOSITION_PERCENTAGES``.
    n_atoms
        Total number of atoms in the cell. Default is `N_ATOMS`.

    Returns
    -------
    dict[str, int]
        Number of atoms of each species, summing to `n_atoms`.
    """
    if composition not in COMPOSITION_PERCENTAGES:
        raise ValueError(f"Unknown composition: {composition}")
    if n_atoms is None:
        n_atoms = N_ATOMS

    counts = {
        symbol: int(n_atoms * percent / 100 + 0.5)
        for symbol, percent in COMPOSITION_PERCENTAGES[composition].items()
    }
    carbon = n_atoms - sum(counts.values())
    if carbon <= 0:
        raise ValueError(
            f"{composition} leaves no carbon atoms in a cell of {n_atoms} atoms"
        )
    return {"C": carbon} | counts


COMPOSITIONS = {
    composition: composition_counts(composition)
    for composition in COMPOSITION_PERCENTAGES
}


@dataclass(frozen=True)
class Run:
    """
    One independent trajectory.

    Attributes
    ----------
    composition : str
        Key in ``COMPOSITIONS``.
    density : float
        Target mass density in g cm^-3.
    number : int
        Repeat number, from 1 to `N_RUNS`, naming the output directory.
    seed : int
        Random seed for the starting structure, initial velocities and
        thermostat.
    """

    composition: str
    density: float
    number: int
    seed: int


# Flat list of every trajectory, ordered composition-major so that a scheduler
# array index maps to a fixed run. Only appending a composition keeps existing
# indices stable: densities and run numbers are inner loops, so adding either
# renumbers every run of every later composition. Output directories are keyed
# by name rather than index, so they are unaffected either way.
#
# Run n uses the same seed in every composition, so runs are paired across
# compositions: C run 1 and CHO run 1 start from the same random draws for their
# site displacements and initial velocities, differing only in which species sit
# on which site and in the cell size set by the target density.
RUNS = tuple(
    Run(composition, density, number, BASE_SEED + number - 1)
    for composition in COMPOSITIONS
    for density in DENSITIES
    for number in range(1, N_RUNS + 1)
)


def cell_dir(model_name: str) -> Path:
    """
    Get the output directory for one model at the current cell size.

    Parameters
    ----------
    model_name
        Registered model name.

    Returns
    -------
    Path
        Directory holding every trajectory run at this cell size.
    """
    cell = "x".join([str(N_CELL_SIZE)] * 3)
    return OUT_PATH / model_name / cell


def trajectory_dir(model_name: str, run: Run) -> Path:
    """
    Get the output directory for one trajectory.

    Parameters
    ----------
    model_name
        Registered model name.
    run
        Trajectory to locate.

    Returns
    -------
    Path
        Directory for the trajectory's Janus and relaxation outputs.
    """
    density_name = f"rho-{run.density:g}".replace(".", "p")
    return cell_dir(model_name) / run.composition / density_name / f"run-{run.number}"


def stage_file_prefix(output_dir: Path, stage_name: str) -> Path:
    """
    Get the output file prefix for one stage, within its own subdirectory.

    Parameters
    ----------
    output_dir
        Trajectory output directory.
    stage_name
        Name of the stage.

    Returns
    -------
    Path
        Prefix shared by every file the stage writes.
    """
    return output_dir / stage_name / stage_name


def stage_file(output_dir: Path, stage_name: str, suffix: str) -> Path:
    """
    Get the path to one of a stage's output files.

    Parameters
    ----------
    output_dir
        Trajectory output directory.
    stage_name
        Name of the stage.
    suffix
        File suffix, such as ``final.extxyz``.

    Returns
    -------
    Path
        Path to the stage output file.
    """
    prefix = stage_file_prefix(output_dir, stage_name)
    return prefix.with_name(f"{stage_name}-{suffix}")


@dataclass(frozen=True)
class Stage:
    """
    One constant-temperature or linearly ramped MD stage.

    Attributes
    ----------
    name : str
        Stage name, used as the Janus output file prefix.
    start_temperature : float
        Target temperature at the start of the stage.
    end_temperature : float
        Target temperature at the end of the stage. Equal to
        `start_temperature` for a constant-temperature hold.
    time_ps : float
        Stage duration in ps.
    timestep_fs : float
        MD timestep in fs.
    """

    name: str
    start_temperature: float
    end_temperature: float
    time_ps: float
    timestep_fs: float

    @property
    def steps(self) -> int:
        """
        Number of MD steps in the stage.

        Returns
        -------
        int
            Steps required to cover `time_ps` at `timestep_fs`.
        """
        return ceil(self.time_ps * FS_PER_PS / self.timestep_fs)

    @property
    def is_ramp(self) -> bool:
        """
        Whether the stage ramps its target temperature.

        Returns
        -------
        bool
            Whether start and end temperatures differ.
        """
        return self.start_temperature != self.end_temperature


def chunk_stage(
    stage: Stage, max_time_ps: float = MAX_CHUNK_TIME_PS
) -> tuple[Stage, ...]:
    """
    Split a long stage into equal chunks.

    Each chunk writes its own final structure, so a killed job resumes at the
    last completed chunk. A ramped stage has its temperature range divided
    between chunks, leaving the overall ramp rate unchanged.

    Parameters
    ----------
    stage
        Stage to split.
    max_time_ps
        Longest chunk duration in ps. Default is `MAX_CHUNK_TIME_PS`.

    Returns
    -------
    tuple[Stage, ...]
        Chunks making up the stage, or the stage itself if short enough.
    """
    if stage.time_ps <= max_time_ps:
        return (stage,)

    n_chunks = ceil(stage.time_ps / max_time_ps)
    temperature_range = stage.end_temperature - stage.start_temperature
    return tuple(
        Stage(
            f"{stage.name}-{index:02d}",
            stage.start_temperature + temperature_range * index / n_chunks,
            stage.start_temperature + temperature_range * (index + 1) / n_chunks,
            stage.time_ps / n_chunks,
            stage.timestep_fs,
        )
        for index in range(n_chunks)
    )


BASE_STAGES = (
    Stage(
        "melt",
        MELT_TEMPERATURE_K,
        MELT_TEMPERATURE_K,
        MELT_TIME_PS,
        MELT_TIMESTEP_FS,
    ),
    Stage(
        "quench",
        MELT_TEMPERATURE_K,
        FINAL_TEMPERATURE_K,
        QUENCH_TIME_PS,
        MELT_TIMESTEP_FS,
    ),
    Stage(
        "equilibrate",
        FINAL_TEMPERATURE_K,
        FINAL_TEMPERATURE_K,
        EQUILIBRATE_TIME_PS,
        MELT_TIMESTEP_FS,
    ),
    Stage(
        "anneal",
        ANNEAL_TEMPERATURE_K,
        ANNEAL_TEMPERATURE_K,
        ANNEAL_TIME_PS,
        ANNEAL_TIMESTEP_FS,
    ),
    Stage(
        "cool",
        ANNEAL_TEMPERATURE_K,
        FINAL_TEMPERATURE_K,
        COOL_TIME_PS,
        ANNEAL_TIMESTEP_FS,
    ),
)

STAGES = tuple(chain.from_iterable(chunk_stage(stage) for stage in BASE_STAGES))

TOTAL_STEPS = sum(stage.steps for stage in STAGES)
TOTAL_TIME_PS = sum(stage.time_ps for stage in STAGES)

EXPECTED_OUTPUT_FILES = ("initial.extxyz",) + tuple(
    f"{stage.name}/{stage.name}-{suffix}"
    for stage in STAGES
    for suffix in ("final.extxyz", "stats.dat", "traj.extxyz")
)

# Chunked stages are relaxed at their last chunk, which holds the structure at
# the end of the stage.
RELAX_STAGES = tuple(
    chunk_stage(stage)[-1].name
    for stage in BASE_STAGES
    if stage.name in RELAX_STAGE_NAMES
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
        Periodic N_CELL_SIZE x N_CELL_SIZE x N_CELL_SIZE simple-cubic structure.
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
    spacing = cell_length / N_CELL_SIZE

    grid = np.indices((N_CELL_SIZE,) * 3).reshape(3, -1).T
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


def _set_ramp_temperature(md: NVT_CSVR, stage: Stage) -> None:
    """
    Update the CSVR target temperature along a linear ramp.

    Parameters
    ----------
    md
        Janus CSVR dynamics object.
    stage
        Stage being run, defining the ramp end points and length.
    """
    progress = min(md.dyn.nsteps / stage.steps, 1.0)
    temperature = stage.start_temperature + progress * (
        stage.end_temperature - stage.start_temperature
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


def _stage_complete(output_dir: Path, stage: Stage) -> bool:
    """
    Check whether one stage has already been run to completion.

    Parameters
    ----------
    output_dir
        Composition-density output directory.
    stage
        Stage to check.

    Returns
    -------
    bool
        Whether the stage wrote a non-empty final structure.
    """
    final_path = stage_file(output_dir, stage.name, "final.extxyz")
    return final_path.is_file() and final_path.stat().st_size > 0


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


def _load_previous_result(status_path: Path) -> dict:
    """
    Load the existing status record for one trajectory.

    Parameters
    ----------
    status_path
        Per-trajectory status file.

    Returns
    -------
    dict
        Existing record, or an empty dict if it is missing or unreadable.
    """
    if not status_path.exists():
        return {}
    try:
        record = json.loads(status_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return record if isinstance(record, dict) else {}


def run_trajectory(
    atoms: Atoms,
    output_dir: Path,
    seed: int,
) -> dict[str, int | float | bool | str | None]:
    """
    Run the melt, quench, equilibration, anneal and cool stages.

    Stages already written to `output_dir` are skipped, resuming from the last
    completed stage's final structure.

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

    calc = atoms.calc
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=INITIAL_TEMPERATURE_K, rng=rng)
    Stationary(atoms, preserve_temperature=True)

    struct = atoms
    completed_steps = 0
    completed_time_ps = 0.0

    for stage_index, stage in enumerate(STAGES):
        if _stage_complete(output_dir, stage):
            struct = read(stage_file(output_dir, stage.name, "final.extxyz"))
            struct.calc = calc
            completed_steps += stage.steps
            completed_time_ps += stage.time_ps
            print(f"Skipping {stage.name} stage in {output_dir}: outputs found")
            continue

        (output_dir / stage.name).mkdir(parents=True, exist_ok=True)
        md = NVT_CSVR(
            struct=struct,
            temp=stage.start_temperature,
            steps=stage.steps,
            timestep=stage.timestep_fs,
            taut=THERMOSTAT_TIME_PS * FS_PER_PS,
            stats_every=STATS_EVERY,
            traj_every=TRAJ_EVERY,
            restart_every=TOTAL_STEPS + 1,
            file_prefix=stage_file_prefix(output_dir, stage.name),
            # Offset per stage so consecutive stages, and the chunks of a split
            # stage, do not each replay the same thermostat noise sequence.
            seed=seed + stage_index,
        )
        if stage.is_ramp:
            md.dyn.attach(_set_ramp_temperature, interval=1, md=md, stage=stage)

        stage_steps, failure = _run_stage(md)
        completed_steps += stage_steps
        completed_time_ps += stage_steps * stage.timestep_fs / FS_PER_PS
        if failure is not None:
            return {
                "stable": False,
                "completed_steps": completed_steps,
                "completed_time_ps": completed_time_ps,
                "failure": failure,
                "failed_stage": stage.name,
            }
        struct = md.struct

    return {
        "stable": completed_steps == TOTAL_STEPS,
        "completed_steps": completed_steps,
        "completed_time_ps": completed_time_ps,
        "failure": None,
        "failed_stage": None,
    }


def run_benchmark(model_name: str, model: Any, runs: Sequence[Run] = RUNS) -> None:
    """
    Run trajectories for one model.

    Each trajectory writes its own ``status.json``, so runs dispatched
    concurrently as separate jobs never write to the same file.

    Parameters
    ----------
    model_name
        Registered model name.
    model
        Model wrapper used to construct the calculator.
    runs
        Trajectories to run. Default is `RUNS`.
    """
    calc = None

    for run in runs:
        output_dir = trajectory_dir(model_name, run)
        label = (
            f"{model_name} {run.composition} at {run.density:g} g cm^-3 "
            f"run {run.number}"
        )
        status_path = output_dir / "status.json"

        if _outputs_complete(output_dir):
            result = _load_previous_result(status_path)
            result.setdefault("stable", True)
            result.setdefault("completed_steps", TOTAL_STEPS)
            result.setdefault("completed_time_ps", TOTAL_TIME_PS)
            result.setdefault("failure", None)
            result.setdefault("failed_stage", None)
            result.setdefault("walltime_seconds", None)
            print(f"Skipping {label}: outputs found")
        else:
            if calc is None:
                calc = model.get_calculator(precision="high")
                calc = model.add_d3_calculator(calc)
            atoms = build_structure(run.composition, run.density, run.seed)
            atoms.calc = calc
            start_time = perf_counter()
            try:
                result = run_trajectory(atoms, output_dir, run.seed)
            except Exception as exc:
                result = {
                    "stable": False,
                    "completed_steps": 0,
                    "completed_time_ps": 0.0,
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failed_stage": None,
                }
            result["walltime_seconds"] = perf_counter() - start_time

        result.update(
            {
                "model": model_name,
                "composition": run.composition,
                "density_g_cm3": run.density,
                "run": run.number,
                "run_id": RUNS.index(run),
                "seed": run.seed,
                "n_cell_size": N_CELL_SIZE,
                "n_atoms": N_ATOMS,
                "anneal_temperature_K": ANNEAL_TEMPERATURE_K,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(result, indent=2) + "\n")
        if not result["stable"]:
            warn(f"{label} failed: {result['failure']}", stacklevel=2)


def run_clean_and_relax(
    model_name: str, model: Any, runs: Sequence[Run] = RUNS
) -> None:
    """
    Clean and relax the annealed and cooled structures for one model.

    Parameters
    ----------
    model_name
        Registered model name.
    model
        Model wrapper used to construct the calculator.
    runs
        Trajectories to clean and relax. Default is `RUNS`.
    """
    calc = None

    for run in runs:
        output_dir = trajectory_dir(model_name, run)
        label = (
            f"{model_name} {run.composition} at {run.density:g} g cm^-3 "
            f"run {run.number}"
        )
        for stage_name in RELAX_STAGES:
            structure_path = stage_file(output_dir, stage_name, "final.extxyz")
            file_prefix = stage_file_prefix(output_dir, stage_name)
            relaxed_path = stage_file(output_dir, stage_name, "relaxed.extxyz")
            info_path = stage_file(output_dir, stage_name, "relax.json")

            if not structure_path.is_file():
                warn(
                    f"Skipping {label}: {structure_path} not found",
                    stacklevel=2,
                )
                continue
            if relaxed_path.is_file() and relaxed_path.stat().st_size > 0:
                print(f"Skipping {relaxed_path}: output found")
                continue

            if calc is None:
                calc = model.get_calculator(precision="high")
                calc = model.add_d3_calculator(calc)

            start_time = perf_counter()
            info = clean_and_relax(structure_path, calc, file_prefix)
            info.update(
                {
                    "model": model_name,
                    "composition": run.composition,
                    "density_g_cm3": run.density,
                    "run": run.number,
                    "run_id": RUNS.index(run),
                    "seed": run.seed,
                    "n_cell_size": N_CELL_SIZE,
                    "stage": stage_name,
                    "walltime_seconds": perf_counter() - start_time,
                }
            )
            info_path.write_text(json.dumps(info, indent=2) + "\n")
            if not info["converged"]:
                warn(
                    f"{label} {stage_name} relaxation did not reach "
                    f"fmax={RELAX_FMAX} in {RELAX_STEPS} steps",
                    stacklevel=2,
                )


def select_runs(
    run_id: int, composition: str = "", density: float = 0.0
) -> tuple[Run, ...]:
    """
    Select the trajectories to run.

    Restricting to one composition and one index per job lets a scheduler array
    run a single MD per GPU.

    Parameters
    ----------
    run_id
        Index into the selected trajectories, or -1 to select all of them. The
        index counts within `composition` when one is given.
    composition
        Key in ``COMPOSITIONS`` to restrict to. Default is every composition.
    density
        Density in ``DENSITIES`` to restrict to. Default, 0, is every density.

    Returns
    -------
    tuple[Run, ...]
        Selected trajectories.
    """
    runs = RUNS
    if composition:
        assert composition in COMPOSITIONS, (
            f"Unknown composition: {composition}. "
            f"Please use one of {', '.join(COMPOSITIONS)}"
        )
        runs = tuple(run for run in runs if run.composition == composition)

    if density > 0:
        assert density in DENSITIES, (
            f"Unknown density: {density:g}. "
            f"Please use one of {', '.join(f'{d:g}' for d in DENSITIES)}"
        )
        runs = tuple(run for run in runs if run.density == density)

    assert run_id in range(-1, len(runs)), (
        f"run_id out of range. Please use -1 for all runs, or 0 to {len(runs) - 1}"
    )
    return runs if run_id < 0 else (runs[run_id],)


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_melt_quench_anneal(
    mlip: tuple[str, Any], run_id: int, composition: str, density: float
) -> None:
    """
    Run the carbon melt-quench-anneal MD benchmark.

    Parameters
    ----------
    mlip
        Tuple of model name and model wrapper.
    run_id
        Index of the trajectory to run, or -1 for all of them.
    composition
        Composition to restrict to, or empty for every composition.
    density
        Density to restrict to, or 0 for every density.
    """
    run_benchmark(*mlip, runs=select_runs(run_id, composition, density))


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_melt_quench_anneal_relax(
    mlip: tuple[str, Any], run_id: int, composition: str, density: float
) -> None:
    """
    Clean and relax the annealed and cooled structures.

    Requires `test_melt_quench_anneal` to have been run first.

    Parameters
    ----------
    mlip
        Tuple of model name and model wrapper.
    run_id
        Index of the trajectory to relax, or -1 for all of them.
    composition
        Composition to restrict to, or empty for every composition.
    density
        Density to restrict to, or 0 for every density.
    """
    run_clean_and_relax(*mlip, runs=select_runs(run_id, composition, density))
