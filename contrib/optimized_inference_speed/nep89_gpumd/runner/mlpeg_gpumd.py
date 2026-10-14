"""ML-PEG inference-timing contract v1.0 runner for GPUMD + NEP."""

from __future__ import annotations

import time

# Contract: `start` is the moment the runscript is entered, before any heavy imports.
START = time.time()

import argparse  # noqa: E402
from collections import deque  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402

from ase import Atoms  # noqa: E402
import ase.io  # noqa: E402
import gpumd_io  # noqa: E402

CONTRACT_VERSION = "1.0"
MODEL_NAME = "nep89"
# Must match the `mlpeg.precisions` label in container.def (checked by the tests).
DECLARED_PRECISIONS = ("mixed",)

NEP_PATH = Path("/opt/model/nep.txt")
GPUMD_BIN = Path("/opt/gpumd/bin/gpumd")
BUILD_INFO = Path("/opt/build-info.json")
SELF_TEST_STRUCTURE = Path("/opt/selftest/amorphous-silica-structure.xyz")
STDBUF = shutil.which("stdbuf")
RUN1_TOLERANCE_S = 1.0
TMP_DIR = Path("/tmp")
OUT_SCRATCH_NAME = ".gpumd-scratch"
# model.xyz is ~190 B/atom; the margin also covers step0.xyz for validation.
SCRATCH_BYTES_PER_ATOM = 500


def build_parser() -> argparse.ArgumentParser:
    """
    Build the strict contract CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting exactly the four contract arguments.
    """
    parser = argparse.ArgumentParser(
        prog="mlpeg-gpumd",
        allow_abbrev=False,
        description="ML-PEG inference-timing runner (GPUMD + NEP89).",
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--precision", required=True, choices=DECLARED_PRECISIONS)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def load_build_info() -> dict:
    """
    Read the build-time provenance written by container.def.

    Returns
    -------
    dict
        Provenance fields; ``"unknown"`` values outside the container.
    """
    if BUILD_INFO.exists():
        return json.loads(BUILD_INFO.read_text())
    return {
        "gpumd_version": "unknown",
        "gpumd_commit": "unknown",
        "gpumd_repo": "https://github.com/brucefan1983/GPUMD",
        "runner_repo": "unknown",
        "runner_commit": "unknown",
        "cuda_archs": "unknown",
    }


def sha256_file(path: Path) -> str:
    """
    Compute the sha256 of a file.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_result(run_id: str, precision: str, material: str, n_atoms: int) -> dict:
    """
    Build an ``inference.json`` skeleton with status ``error``.

    Parameters
    ----------
    run_id
        Run identifier.
    precision
        Requested precision mode.
    material
        Material name.
    n_atoms
        Number of atoms.

    Returns
    -------
    dict
        Result skeleton to be filled in.
    """
    info = load_build_info()
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "error",
        "error": None,
        "model": {
            "name": MODEL_NAME,
            "weights_sha256": sha256_file(NEP_PATH) if NEP_PATH.exists() else None,
            "precision": precision,
        },
        "implementation": {
            "engine": "GPUMD",
            "engine_version": info["gpumd_version"],
            "repo": info["runner_repo"],
            "commit": info["runner_commit"],
            "settings": {
                "gpumd_repo": info["gpumd_repo"],
                "gpumd_commit": info["gpumd_commit"],
                "cuda_archs": info["cuda_archs"],
                "nep_precision": "fp32 descriptors/ANN/per-atom forces; fp64 "
                "positions, velocities, global forces and integration",
                "neighbor_list": "cell list at rc + 1.0 A skin, rebuilt when an "
                "atom moved > 0.5 A; exact per-step cutoff filter",
            },
        },
        "system": {"material": material, "n_atoms": n_atoms},
        "timestamps_unix": {
            "start": START,
            "setup_end": None,
            "warmup_end": None,
            "measure_end": None,
        },
        "steps": {"warmup": None, "measured": None},
        "energy": {"total_start_eV": None, "total_end_eV": None},
        "self_reported": {},
    }


def write_result(out_dir: Path, result: dict) -> None:
    """
    Write ``inference.json``.

    Parameters
    ----------
    out_dir
        Output directory.
    result
        Result dictionary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inference.json").write_text(json.dumps(result, indent=2) + "\n")


def run_gpumd(workdir: Path, tracker: gpumd_io.StdoutTracker | None = None):
    """
    Run GPUMD in ``workdir``, streaming stdout line by line.

    Parameters
    ----------
    workdir
        Directory holding ``run.in``, ``model.xyz`` and ``nep.txt``.
    tracker
        Optional tracker fed with each stdout line and its read time.

    Returns
    -------
    tuple[int, str, str]
        Return code, full stdout and full stderr.
    """
    cmd = [str(GPUMD_BIN)]
    if STDBUF is not None:
        cmd = [STDBUF, "-oL", "-eL", *cmd]
    stdout_lines: list[str] = []
    with open(workdir / "stderr.log", "w+") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            now = time.time()
            stdout_lines.append(line)
            if tracker is not None:
                tracker.feed(line, now)
        returncode = proc.wait()
        err.seek(0)
        stderr = err.read()
    return returncode, "".join(stdout_lines), stderr


def tail(text: str, n: int = 50) -> str:
    """
    Return the last ``n`` lines of a text.

    Parameters
    ----------
    text
        Input text.
    n
        Number of lines.

    Returns
    -------
    str
        The tail.
    """
    return "\n".join(deque(text.splitlines(), maxlen=n))


def make_scratch(n_atoms: int, out_dir: Path) -> Path:
    """
    Create the scratch directory, on /tmp if it has room, else under ``out_dir``.

    Apptainer's ``--containall`` /tmp is a session tmpfs capped by the site's
    ``sessiondir max size`` (64 MB by default), too small for ``model.xyz`` beyond
    a few hundred thousand atoms. ``/mlpeg/out`` is the only other writable
    location the contract allows.

    Parameters
    ----------
    n_atoms
        Number of atoms (sizes the estimate).
    out_dir
        Contract output directory, used as fallback.

    Returns
    -------
    Path
        A fresh, empty scratch directory.
    """
    needed = SCRATCH_BYTES_PER_ATOM * n_atoms + (1 << 20)
    base = TMP_DIR
    if shutil.disk_usage(TMP_DIR).free < needed:
        base = out_dir / OUT_SCRATCH_NAME
        base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="mlpeg-gpumd-", dir=base))


def remove_scratch(scratch: Path, out_dir: Path) -> None:
    """
    Delete the scratch directory and, if empty, the fallback parent in ``out_dir``.

    Parameters
    ----------
    scratch
        Directory returned by :func:`make_scratch`.
    out_dir
        Contract output directory.
    """
    shutil.rmtree(scratch, ignore_errors=True)
    fallback = out_dir / OUT_SCRATCH_NAME
    if fallback.is_dir() and not any(fallback.iterdir()):
        fallback.rmdir()


def prepare_workdir(
    atoms: Atoms, run_params: list[tuple[str, object]], scratch: Path, name: str
) -> Path:
    """
    Create a GPUMD working directory with ``model.xyz``, ``nep.txt`` and ``run.in``.

    Parameters
    ----------
    atoms
        Input structure.
    run_params
        ``run.in`` keyword/value pairs.
    scratch
        Scratch directory from :func:`make_scratch`.
    name
        Subdirectory name.

    Returns
    -------
    Path
        The working directory.
    """
    workdir = scratch / name
    workdir.mkdir()
    gpumd_io.write_model_xyz(atoms, workdir / "model.xyz")
    (workdir / "nep.txt").symlink_to(NEP_PATH)
    gpumd_io.write_run_in(workdir / "run.in", run_params)
    return workdir


def single_point(atoms: Atoms, scratch: Path) -> tuple[float, object]:
    """
    Untimed GPUMD single point at the input positions.

    Parameters
    ----------
    atoms
        Input structure.
    scratch
        Scratch directory from :func:`make_scratch`.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Potential energy (eV) and forces (eV/Å).
    """
    workdir = prepare_workdir(
        atoms, gpumd_io.single_point_parameters(), scratch, "single-point"
    )
    returncode, stdout, stderr = run_gpumd(workdir)
    if returncode != 0:
        raise RuntimeError(
            f"GPUMD single point failed ({returncode}):\n{tail(stderr or stdout)}"
        )
    return gpumd_io.read_step0_xyz(workdir / "step0.xyz", atoms)


def run(spec_path: Path, run_id: str, precision: str, out_dir: Path) -> dict:
    """
    Execute one contract run and write ``inference.json``.

    Parameters
    ----------
    spec_path
        Path to ``spec.json``; structure paths are relative to its directory.
    run_id
        Run identifier ``<material>-n<n_atoms>-r<repeat>``.
    precision
        Requested precision mode (already validated).
    out_dir
        Output directory.

    Returns
    -------
    dict
        The written result.
    """
    material, n_atoms, _repeat = gpumd_io.parse_run_id(run_id)
    result = base_result(run_id, precision, material, n_atoms)
    try:
        _run(spec_path, material, n_atoms, out_dir, result)
    except Exception:  # noqa: BLE001 - every failure must still produce a result
        result["status"] = "error"
        result["error"] = tail(traceback.format_exc())
    write_result(out_dir, result)
    return result


def _run(
    spec_path: Path, material: str, n_atoms: int, out_dir: Path, result: dict
) -> None:
    """
    Fill ``result`` in place; see :func:`run`.

    Parameters
    ----------
    spec_path
        Path to ``spec.json``.
    material
        Material name.
    n_atoms
        Number of atoms.
    out_dir
        Output directory.
    result
        Result dictionary to fill.
    """
    spec = json.loads(spec_path.read_text())
    system, size = gpumd_io.find_size(spec, material, n_atoms)
    structure_path = spec_path.parent / size["file"]
    expected_sha = size.get("sha256", "")
    if len(expected_sha) == 64 and sha256_file(structure_path) != expected_sha:
        raise ValueError(f"sha256 mismatch for {structure_path}")
    atoms = ase.io.read(structure_path)
    if len(atoms) != n_atoms:
        raise ValueError(f"{structure_path} has {len(atoms)} atoms, expected {n_atoms}")

    supported = set(gpumd_io.nep_elements(NEP_PATH))
    missing = sorted(set(atoms.get_chemical_symbols()) - supported)
    if missing:
        result["status"] = "unsupported_elements"
        result["error"] = f"Elements not in nep.txt: {', '.join(missing)}"
        return

    md = spec["md"]
    timestep = (
        md["timestep_fs_if_hydrogen"]
        if system["contains_hydrogen"]
        else md["timestep_fs"]
    )
    warmup, measured = md["warmup_steps"], md["measured_steps"]
    interval = gpumd_io.thermo_interval(md["energy_log_interval"], warmup, measured)
    result["steps"] = {"warmup": warmup, "measured": measured}
    settings = result["implementation"]["settings"]
    settings.update({"timestep_fs": timestep, "dump_thermo_interval": interval})

    scratch = make_scratch(n_atoms, out_dir)
    result["self_reported"]["scratch_dir"] = str(scratch)
    try:
        _simulate(
            atoms, size, timestep, warmup, measured, interval, scratch, out_dir, result
        )
    finally:
        remove_scratch(scratch, out_dir)


def _simulate(
    atoms: Atoms,
    size: dict,
    timestep: float,
    warmup: int,
    measured: int,
    interval: int,
    scratch: Path,
    out_dir: Path,
    result: dict,
) -> None:
    """
    Run the timed MD and the optional validation single point; fill ``result``.

    Parameters
    ----------
    atoms
        Input structure.
    size
        Size entry of the spec.
    timestep
        Time step in fs.
    warmup
        Number of warm-up steps.
    measured
        Number of measured steps.
    interval
        ``dump_thermo`` interval.
    scratch
        Scratch directory.
    out_dir
        Output directory.
    result
        Result dictionary to fill.
    """
    n_atoms = len(atoms)
    settings = result["implementation"]["settings"]
    workdir = prepare_workdir(
        atoms,
        gpumd_io.run_parameters(timestep, warmup, measured, interval),
        scratch,
        "md",
    )
    tracker = gpumd_io.StdoutTracker()
    returncode, stdout, stderr = run_gpumd(workdir, tracker)
    reported = result["self_reported"]
    reported.update(
        {
            "gpumd_run_seconds": tracker.run_seconds,
            "gpumd_speed_atom_step_per_s": tracker.speeds,
            "devices": tracker.devices,
            "compute_capabilities": tracker.compute_capabilities,
        }
    )
    settings.update(tracker.settings)

    if returncode != 0:
        if gpumd_io.detect_oom(stdout, stderr):
            result["status"] = "oom"
            result["error"] = tail(stderr or stdout)
        else:
            result["status"] = "error"
            result["error"] = (
                f"gpumd exited with {returncode}:\n{tail(stderr or stdout)}"
            )
        return
    if tracker.n_gpus is not None and tracker.n_gpus != 1:
        raise RuntimeError(
            f"GPUMD saw {tracker.n_gpus} GPUs and would use multi-GPU NEP; "
            "expose exactly one GPU via CUDA_VISIBLE_DEVICES"
        )

    stamps = gpumd_io.derive_timestamps(tracker)
    result["timestamps_unix"].update(
        setup_end=stamps.setup_end,
        warmup_end=stamps.warmup_end,
        measure_end=stamps.measure_end,
    )
    reported["run1_end_seen_minus_warmup_end_s"] = -stamps.run1_offset_s
    if abs(stamps.run1_offset_s) > RUN1_TOLERANCE_S:
        reported["warning"] = (
            f"end of run 1 seen {-stamps.run1_offset_s:+.3f} s from derived "
            f"warmup_end (tolerance {RUN1_TOLERANCE_S} s)"
        )

    e_start, e_end = gpumd_io.thermo_energies(
        workdir / "thermo.out", warmup // interval, measured // interval
    )
    result["energy"] = {"total_start_eV": e_start, "total_end_eV": e_end}
    reported["nve_drift_meV_per_atom"] = 1000 * (e_end - e_start) / n_atoms

    if size.get("validate", False):
        energy, forces = single_point(atoms, scratch)
        frame = Atoms(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        frame.info["energy"] = energy
        frame.arrays["forces"] = forces
        out_dir.mkdir(parents=True, exist_ok=True)
        ase.io.write(out_dir / "validation.extxyz", frame, format="extxyz")

    result["status"] = "ok"


def self_test() -> int:
    """
    Stage a small amorphous SiO2 system and run the full contract path.

    Returns
    -------
    int
        Exit code: 0 if the run is ``ok`` and the NVE drift is below 1 meV/atom.
    """
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
    )
    import numpy as np

    root = Path(tempfile.mkdtemp(prefix="mlpeg-selftest-", dir="/tmp"))
    in_dir, out_dir = root / "in", root / "out"
    (in_dir / "structures").mkdir(parents=True)
    atoms = ase.io.read(SELF_TEST_STRUCTURE)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))
    Stationary(atoms)
    material = "sio2_amorphous"
    rel = f"structures/{material}_{len(atoms)}.extxyz"
    ase.io.write(in_dir / rel, atoms, format="extxyz")
    spec = {
        "md": {
            "timestep_fs": 1.0,
            "timestep_fs_if_hydrogen": 0.5,
            "warmup_steps": 100,
            "measured_steps": 1000,
            "energy_log_interval": 100,
        },
        "systems": [
            {
                "material": material,
                "elements": ["Si", "O"],
                "contains_hydrogen": False,
                "sizes": [
                    {
                        "n_atoms": len(atoms),
                        "file": rel,
                        "sha256": sha256_file(in_dir / rel),
                        "validate": True,
                    }
                ],
            }
        ],
    }
    (in_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    result = run(in_dir / "spec.json", f"{material}-n{len(atoms)}-r1", "mixed", out_dir)
    print(json.dumps(result, indent=2))
    ok = result["status"] == "ok"
    if ok:
        drift = result["self_reported"]["nve_drift_meV_per_atom"]
        frame = ase.io.read(out_dir / "validation.extxyz")
        print(f"\nNVE drift: {drift:+.4f} meV/atom over 1000 steps (limit 1)")
        print(f"step-0 energy: {frame.get_potential_energy():.6f} eV")
        print(f"timestamps consistent: {'warning' not in result['self_reported']}")
        print(f"outputs in: {out_dir}")
        ok = abs(drift) < 1.0
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """
    Contract entry point.

    Parameters
    ----------
    argv
        Command-line arguments (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return self_test()
    args = build_parser().parse_args(argv)
    try:
        gpumd_io.parse_run_id(args.run_id)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") > 0:
        print("warning: more than one GPU visible", file=sys.stderr)
    run(args.spec, args.run_id, args.precision, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
