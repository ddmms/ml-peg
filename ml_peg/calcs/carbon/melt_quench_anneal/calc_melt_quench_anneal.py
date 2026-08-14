"""Run carbon melt-quench-anneal simulations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import LBFGS
from janus_core.calculations.md import NVT_CSVR
import numpy as np
import pytest

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Side length of the simple-cubic starting structure, in unit cells.
N_CELL_SIZE = 12
N_ATOMS = N_CELL_SIZE**3
DENSITIES = (1.0,)

# Atomic percentages of the non-carbon species in each target material, matching
# the reference C and C75H15O10 material configs. Carbon takes the remainder.
COMPOSITION_PERCENTAGES = {
    "C": {},
    # "CHO": {"H": 15.0, "O": 10.0},
}

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

# Post-MD clean and relax. Isolated fragments smaller than MIN_CLUSTER_SIZE are
# dropped before relaxing, so gas-phase molecules ejected during the melt do not
# contribute to the final amorphous structure.
RELAX_STAGE_NAMES = ("anneal", "cool")
MIN_CLUSTER_SIZE = 20
BOND_SCALE = 1.10
RELAX_FMAX = 0.01
RELAX_STEPS = 5000


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


def trajectory_dir(model_name: str, composition: str, density: float) -> Path:
    """
    Get the output directory for one trajectory.

    Parameters
    ----------
    model_name
        Registered model name.
    composition
        Key in ``COMPOSITIONS``.
    density
        Target mass density in g cm^-3.

    Returns
    -------
    Path
        Directory for the trajectory's Janus and relaxation outputs.
    """
    run_name = f"rho-{density:g}".replace(".", "p")
    return cell_dir(model_name) / composition / run_name


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


STAGES = (
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

TOTAL_STEPS = sum(stage.steps for stage in STAGES)
TOTAL_TIME_PS = sum(stage.time_ps for stage in STAGES)

EXPECTED_OUTPUT_FILES = ("initial.extxyz",) + tuple(
    f"{stage.name}/{stage.name}-{suffix}"
    for stage in STAGES
    for suffix in ("final.extxyz", "stats.dat", "traj.extxyz")
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

    for stage in STAGES:
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
            seed=seed,
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
    output_root = cell_dir(model_name)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    previous_results = _load_previous_results(status_path)
    results = []
    status_path.write_text("[]\n")
    calc = None

    for composition_index, composition in enumerate(COMPOSITIONS):
        seed = BASE_SEED + composition_index
        for density in DENSITIES:
            output_dir = trajectory_dir(model_name, composition, density)
            result = None
            if _outputs_complete(output_dir):
                result = previous_results.get((composition, density), {}).copy()
                result.setdefault("stable", True)
                result.setdefault("completed_steps", TOTAL_STEPS)
                result.setdefault("completed_time_ps", TOTAL_TIME_PS)
                result.setdefault("failure", None)
                result.setdefault("failed_stage", None)
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
                        "failed_stage": None,
                    }
                result["walltime_seconds"] = perf_counter() - start_time
            result.update(
                {
                    "composition": composition,
                    "density_g_cm3": density,
                    "seed": seed,
                    "run_name": output_dir.name,
                    "n_cell_size": N_CELL_SIZE,
                    "n_atoms": N_ATOMS,
                    "anneal_temperature_K": ANNEAL_TEMPERATURE_K,
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


def _bond_graph(atoms: Atoms, bond_scale: float = BOND_SCALE) -> list[list[int]]:
    """
    Build the bonded-neighbour adjacency list of a structure.

    Parameters
    ----------
    atoms
        Structure to analyse.
    bond_scale
        Multiplier applied to the natural covalent cutoffs.

    Returns
    -------
    list[list[int]]
        Neighbour indices for each atom.
    """
    cutoffs = natural_cutoffs(atoms, mult=bond_scale)
    neighbor_list = NeighborList(
        cutoffs, skin=0.0, self_interaction=False, bothways=True
    )
    neighbor_list.update(atoms)
    return [
        neighbor_list.get_neighbors(index)[0].tolist() for index in range(len(atoms))
    ]


def _components(graph: list[list[int]]) -> list[list[int]]:
    """
    Find the connected components of a bond graph.

    Parameters
    ----------
    graph
        Neighbour indices for each atom.

    Returns
    -------
    list[list[int]]
        Atom indices making up each connected component.
    """
    seen = np.zeros(len(graph), dtype=bool)
    components = []
    for start in range(len(graph)):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in graph[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def remove_small_clusters(
    atoms: Atoms,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    bond_scale: float = BOND_SCALE,
) -> tuple[Atoms, dict[str, Any]]:
    """
    Remove isolated fragments smaller than a minimum size.

    Parameters
    ----------
    atoms
        Structure to clean.
    min_cluster_size
        Smallest connected component retained.
    bond_scale
        Multiplier applied to the natural covalent cutoffs.

    Returns
    -------
    tuple[ase.Atoms, dict[str, Any]]
        Cleaned structure and a summary of what was removed.
    """
    components = _components(_bond_graph(atoms, bond_scale=bond_scale))
    sizes = np.array([len(component) for component in components], dtype=int)
    small = sizes < min_cluster_size
    removed = sorted(
        {
            index
            for component_index in np.where(small)[0]
            for index in components[component_index]
        }
    )

    mask = np.ones(len(atoms), dtype=bool)
    if removed:
        mask[removed] = False
    cleaned = atoms[mask]

    info = {
        "n_atoms_in": len(atoms),
        "n_atoms_out": len(cleaned),
        "n_clusters": len(components),
        "n_small_clusters": int(small.sum()),
        "small_cluster_sizes": sorted(sizes[small].tolist()),
        "n_removed_atoms": len(removed),
        "min_cluster_size": min_cluster_size,
        "bond_scale": bond_scale,
    }
    return cleaned, info


def clean_and_relax(
    structure_path: Path,
    calc: Any,
    file_prefix: Path,
) -> dict[str, Any]:
    """
    Remove small clusters from an MD structure and relax what remains.

    Parameters
    ----------
    structure_path
        Final structure written by an MD stage.
    calc
        Calculator used for the relaxation.
    file_prefix
        Prefix for the cleaned, relaxed, log and trajectory outputs.

    Returns
    -------
    dict[str, Any]
        Cluster removal summary and relaxation outcome.
    """
    atoms = read(structure_path)
    cleaned, info = remove_small_clusters(atoms)
    # Velocities from the MD stage are meaningless after relaxation.
    cleaned.arrays.pop("momenta", None)
    write(file_prefix.with_name(f"{file_prefix.name}-cleaned.extxyz"), cleaned)

    # A low-density cell can fragment entirely into small clusters, leaving
    # nothing to relax.
    if not len(cleaned):
        info.update(
            {"converged": False, "relax_steps": 0, "empty_after_cleaning": True}
        )
        return info

    cleaned.calc = calc
    optimizer = LBFGS(
        cleaned,
        logfile=str(file_prefix.with_name(f"{file_prefix.name}-relax.log")),
        trajectory=str(file_prefix.with_name(f"{file_prefix.name}-relax.traj")),
    )
    converged = optimizer.run(fmax=RELAX_FMAX, steps=RELAX_STEPS)
    write(file_prefix.with_name(f"{file_prefix.name}-relaxed.extxyz"), cleaned)

    info.update(
        {
            "converged": bool(converged),
            "relax_steps": int(optimizer.get_number_of_steps()),
            "max_relax_steps": RELAX_STEPS,
            "fmax": RELAX_FMAX,
            "max_force": float(np.linalg.norm(cleaned.get_forces(), axis=1).max()),
            "energy": float(cleaned.get_potential_energy()),
        }
    )
    return info


def run_clean_and_relax(model_name: str, model: Any) -> None:
    """
    Clean and relax the annealed and cooled structures for one model.

    Parameters
    ----------
    model_name
        Registered model name.
    model
        Model wrapper used to construct the calculator.
    """
    results = []
    calc = None

    for composition in COMPOSITIONS:
        for density in DENSITIES:
            output_dir = trajectory_dir(model_name, composition, density)
            for stage_name in RELAX_STAGE_NAMES:
                structure_path = stage_file(output_dir, stage_name, "final.extxyz")
                file_prefix = stage_file_prefix(output_dir, stage_name)
                relaxed_path = stage_file(output_dir, stage_name, "relaxed.extxyz")
                info_path = stage_file(output_dir, stage_name, "relax.json")

                if not structure_path.is_file():
                    warn(
                        f"Skipping {model_name} {composition} at {density:g} g cm^-3: "
                        f"{structure_path} not found",
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
                        "composition": composition,
                        "density_g_cm3": density,
                        "n_cell_size": N_CELL_SIZE,
                        "stage": stage_name,
                        "walltime_seconds": perf_counter() - start_time,
                    }
                )
                info_path.write_text(json.dumps(info, indent=2) + "\n")
                results.append(info)
                if not info["converged"]:
                    warn(
                        f"{model_name} {composition} at {density:g} g cm^-3 "
                        f"{stage_name} relaxation did not reach "
                        f"fmax={RELAX_FMAX} in {RELAX_STEPS} steps",
                        stacklevel=2,
                    )


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_melt_quench_anneal(mlip: tuple[str, Any]) -> None:
    """
    Run the carbon melt-quench-anneal MD benchmark.

    Parameters
    ----------
    mlip
        Tuple of model name and model wrapper.
    """
    run_benchmark(*mlip)


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_melt_quench_anneal_relax(mlip: tuple[str, Any]) -> None:
    """
    Clean and relax the annealed and cooled structures.

    Requires `test_melt_quench_anneal` to have been run first.

    Parameters
    ----------
    mlip
        Tuple of model name and model wrapper.
    """
    run_clean_and_relax(*mlip)
