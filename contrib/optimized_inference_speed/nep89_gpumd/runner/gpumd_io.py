"""GPUMD input/output helpers for the ML-PEG inference-timing contract.

File I/O goes through calorine (``calorine.gpumd`` for model.xyz, run.in,
thermo.out and dump_xyz frames; ``calorine.nep`` for the model). What calorine
does not cover, GPUMD's stdout, is parsed here. Everything runs without a GPU.
It targets GPUMD v5.9.1; see README.md for the source references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re

from ase import Atoms
from calorine.gpumd import read_runfile, read_thermo, read_xyz, write_runfile, write_xyz
from calorine.nep import read_model
import numpy as np

# GPUMD's internal time unit in fs (src/utilities/common.cuh). Only used by the
# tests to mirror GPUMD's own conversion; calorine uses ASE's exact value.
GPUMD_TIME_UNIT_CONVERSION = 10.18051

RUN_ID_RE = re.compile(r"^(?P<material>.+)-n(?P<n_atoms>\d+)-r(?P<repeat>\d+)$")


def parse_run_id(run_id: str) -> tuple[str, int, int]:
    """
    Split a run ID of the form ``<material>-n<n_atoms>-r<repeat>``.

    Parameters
    ----------
    run_id
        Run identifier passed on the command line.

    Returns
    -------
    tuple[str, int, int]
        Material name, number of atoms and repeat index.
    """
    match = RUN_ID_RE.match(run_id)
    if match is None:
        raise ValueError(f"Malformed run ID: {run_id!r}")
    return match["material"], int(match["n_atoms"]), int(match["repeat"])


def find_size(spec: dict, material: str, n_atoms: int) -> tuple[dict, dict]:
    """
    Find the system and size entries of the spec matching a run.

    Parameters
    ----------
    spec
        Parsed ``spec.json``.
    material
        Material name from the run ID.
    n_atoms
        Number of atoms from the run ID.

    Returns
    -------
    tuple[dict, dict]
        The system entry and the size entry.
    """
    for system in spec["systems"]:
        if system["material"] != material:
            continue
        for size in system["sizes"]:
            if size["n_atoms"] == n_atoms:
                return system, size
    raise KeyError(f"No size with n_atoms={n_atoms} for material {material!r}")


def nep_elements(nep_path: str | Path) -> list[str]:
    """
    Read the species supported by a NEP model.

    Parameters
    ----------
    nep_path
        Path to the NEP model file.

    Returns
    -------
    list[str]
        Chemical symbols supported by the model.
    """
    return list(read_model(str(nep_path)).types)


def write_model_xyz(atoms: Atoms, path: str | Path) -> None:
    """
    Write a GPUMD ``model.xyz`` with masses and velocities via calorine.

    Only numbers, positions, cell, PBC, masses and momenta are passed on. GPUMD
    splits the comment line on whitespace, so free-text ``info`` entries from the
    input file must not reach it. calorine converts velocities to Å/fs.

    Parameters
    ----------
    atoms
        Structure carrying cell, PBC, masses and momenta.
    path
        Output file path.
    """
    if atoms.cell.rank != 3:
        raise ValueError("GPUMD requires a full 3D cell")
    if not atoms.has("momenta"):
        raise ValueError("Structure has no momenta")
    clean = Atoms(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
        masses=atoms.get_masses(),
        momenta=atoms.get_momenta(),
    )
    write_xyz(str(path), clean)
    # calorine omits velocities when they are all below 1e-6 (ASE units); GPUMD
    # would then draw random 300 K velocities, which the contract forbids.
    with open(path) as fh:
        fh.readline()
        header = fh.readline()
    if ":vel:R:3" not in header:
        raise ValueError(
            "model.xyz has no velocities (momenta are all ~0); GPUMD would "
            "re-thermalize the system"
        )


def thermo_interval(log_interval: int, warmup: int, measured: int) -> int:
    """
    Choose a ``dump_thermo`` interval that lands a row on both run ends.

    Parameters
    ----------
    log_interval
        Requested energy logging interval.
    warmup
        Number of warm-up steps.
    measured
        Number of measured steps.

    Returns
    -------
    int
        Largest interval dividing all three numbers.
    """
    return math.gcd(log_interval, warmup, measured)


def run_parameters(
    timestep_fs: float, warmup: int, measured: int, interval: int
) -> list[tuple[str, object]]:
    """
    Keyword/value pairs of the timed NVE ``run.in``.

    GPUMD resets the ensemble and all output keywords after every ``run``, so
    both blocks repeat them. No ``velocity`` keyword: velocities come from
    ``model.xyz`` and are used verbatim.

    Parameters
    ----------
    timestep_fs
        Integration time step in fs.
    warmup
        Number of warm-up steps.
    measured
        Number of measured steps.
    interval
        ``dump_thermo`` interval.

    Returns
    -------
    list[tuple[str, object]]
        Parameters for :func:`calorine.gpumd.write_runfile`.
    """
    return [
        ("potential", "nep.txt"),
        ("time_step", timestep_fs),
        ("ensemble", "nve"),
        ("dump_thermo", interval),
        ("run", warmup),
        ("ensemble", "nve"),
        ("dump_thermo", interval),
        ("run", measured),
    ]


def single_point_parameters() -> list[tuple[str, object]]:
    """
    Keyword/value pairs of the untimed step-0 single point.

    ``run 0`` is rejected by GPUMD and dumps first write at step 1. With a zero
    time step neither positions nor velocities change, so the step-1 dump holds
    the energy and forces at the input positions.

    Returns
    -------
    list[tuple[str, object]]
        Parameters for :func:`calorine.gpumd.write_runfile`.
    """
    return [
        ("potential", "nep.txt"),
        ("time_step", 0),
        ("ensemble", "nve"),
        ("dump_xyz", (1, "step0.xyz", "precision", "double", "force", "potential")),
        ("run", 1),
    ]


def write_run_in(path: str | Path, parameters: list[tuple[str, object]]) -> None:
    """
    Write ``run.in`` with calorine and check that it reads back unchanged.

    Parameters
    ----------
    path
        Output file path.
    parameters
        Keyword/value pairs.
    """
    write_runfile(Path(path), parameters)
    if read_runfile(str(path)) != parameters:
        raise ValueError(f"run.in round trip mismatch in {path}")


def thermo_energies(
    path: str | Path, warmup_rows: int, measured_rows: int
) -> tuple[float, float]:
    """
    Total energies at the end of the warm-up and of the measured run.

    calorine lays the per-run blocks of ``thermo.out`` end to end, so the
    warm-up block is the first ``warmup_rows`` rows.

    Parameters
    ----------
    path
        Path to ``thermo.out``.
    warmup_rows
        Number of rows written during the warm-up.
    measured_rows
        Number of rows written during the measured run.

    Returns
    -------
    tuple[float, float]
        KE + PE (eV) of the last warm-up row and of the last measured row.
    """
    df = read_thermo(str(path))
    if len(df) != warmup_rows + measured_rows:
        raise ValueError(
            f"thermo.out has {len(df)} rows, expected {warmup_rows} + {measured_rows}"
        )
    total = (df["kinetic_energy"] + df["potential_energy"]).to_numpy()
    return float(total[warmup_rows - 1]), float(total[-1])


_RUN_TIME_RE = re.compile(r"^Time used for this run = (\S+) second\.")
_RUN_SPEED_RE = re.compile(r"^Speed of this run = (\S+) atom\*step/second\.")
_N_GPUS_RE = re.compile(r"^number of GPUs = (\d+)")
_DEVICE_RE = re.compile(r"^\s+Device name:\s+(.+?)\s*$")
_CC_RE = re.compile(r"^\s+Compute capability:\s+(\S+)")
_SETTING_RE = re.compile(
    r"^\s+(radial cutoff|angular cutoff|MN_radial|enlarged MN_radial"
    r"|enlarged MN_angular) = ([0-9.eE+-]+?)( A)?\.\s*$"
)


@dataclass
class StdoutTracker:
    """Collect timing events and settings from streamed GPUMD stdout."""

    run_seconds: list[float] = field(default_factory=list)
    run_end_seen: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    n_gpus: int | None = None
    devices: list[str] = field(default_factory=list)
    compute_capabilities: list[str] = field(default_factory=list)
    settings: dict[str, float] = field(default_factory=dict)

    def feed(self, line: str, wall_time: float) -> None:
        """
        Process one stdout line.

        Parameters
        ----------
        line
            A line of GPUMD stdout.
        wall_time
            ``time.time()`` when the line was read.
        """
        if match := _RUN_TIME_RE.match(line):
            self.run_seconds.append(float(match[1]))
            self.run_end_seen.append(wall_time)
        elif match := _RUN_SPEED_RE.match(line):
            self.speeds.append(float(match[1]))
        elif match := _N_GPUS_RE.match(line):
            self.n_gpus = int(match[1])
        elif match := _DEVICE_RE.match(line):
            self.devices.append(match[1])
        elif match := _CC_RE.match(line):
            self.compute_capabilities.append(match[1])
        elif match := _SETTING_RE.match(line):
            key = match[1].replace(" ", "_")
            self.settings.setdefault(key, float(match[2]))


@dataclass
class Timestamps:
    """Derived phase boundaries for the contract."""

    setup_end: float
    warmup_end: float
    measure_end: float
    run1_end_seen: float
    run1_offset_s: float


def derive_timestamps(tracker: StdoutTracker) -> Timestamps:
    """
    Derive the phase boundaries from the stdout clock and GPUMD's run timers.

    ``measure_end`` is the wall clock when the second run's "Time used" line was
    read; the earlier boundaries subtract GPUMD's own per-run timers so that
    stdout latency never enters the measured window.

    Parameters
    ----------
    tracker
        Tracker that has seen both run blocks.

    Returns
    -------
    Timestamps
        Phase boundaries and the run-1 consistency offset.
    """
    if len(tracker.run_seconds) != 2:
        raise ValueError(
            f"Expected 2 completed runs, saw {len(tracker.run_seconds)} in stdout"
        )
    measure_end = tracker.run_end_seen[1]
    warmup_end = measure_end - tracker.run_seconds[1]
    setup_end = warmup_end - tracker.run_seconds[0]
    run1_end_seen = tracker.run_end_seen[0]
    return Timestamps(
        setup_end=setup_end,
        warmup_end=warmup_end,
        measure_end=measure_end,
        run1_end_seen=run1_end_seen,
        run1_offset_s=warmup_end - run1_end_seen,
    )


_OOM_PATTERNS = ("out of memory", "cudaErrorMemoryAllocation", "bad_alloc")


def detect_oom(*texts: str) -> bool:
    """
    Check GPUMD output for an out-of-memory failure.

    Parameters
    ----------
    *texts
        Captured stdout and/or stderr.

    Returns
    -------
    bool
        Whether any text reports an allocation failure.
    """
    return any(pattern in text for text in texts for pattern in _OOM_PATTERNS)


def read_step0_xyz(path: str | Path, reference: Atoms) -> tuple[float, np.ndarray]:
    """
    Read energy and forces from the single-point ``dump_xyz`` frame.

    The frame is checked against the input structure (species order and
    minimum-image positions) so that atom reordering or a moved frame is caught.

    Parameters
    ----------
    path
        Path to ``step0.xyz`` written by GPUMD.
    reference
        The input structure.

    Returns
    -------
    tuple[float, np.ndarray]
        Potential energy in eV and forces in eV/Å.
    """
    frame = read_xyz(str(path))
    if len(frame) != len(reference):
        raise ValueError(f"step0.xyz has {len(frame)} atoms, expected {len(reference)}")
    if frame.get_chemical_symbols() != reference.get_chemical_symbols():
        raise ValueError("step0.xyz species order differs from the input")
    delta = frame.positions - reference.positions
    frac = np.linalg.solve(reference.cell.array.T, delta.T).T
    frac -= np.round(frac) * reference.pbc
    max_shift = np.abs(frac @ reference.cell.array).max()
    # model.xyz positions are written with 8 decimals (ASE extxyz '%16.8f').
    if max_shift > 1e-6:
        raise ValueError(f"step0.xyz positions moved by up to {max_shift:.2e} Å")
    return float(frame.get_potential_energy()), frame.get_forces()
