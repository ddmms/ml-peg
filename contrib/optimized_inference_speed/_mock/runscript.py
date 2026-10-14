"""
Mock container implementing the ML-PEG optimised inference contract v1.0.

Uses only the standard library. It does no MD: it sleeps briefly, then writes a
valid ``inference.json`` (and a zero-force ``validation.extxyz`` when required).

Failures are planted with ``MLPEG_MOCK_FAIL``, a ``;``-separated list of
``<fault>:<run-id glob>`` entries, e.g. ``oom:sio2_quartz-n6144-*;error:*-r2``.

Faults
------
oom, unsupported_elements, skipped
    Exit 0 with that status.
error
    Exit 1 without writing ``inference.json``.
nojson
    Exit 0 without writing ``inference.json``.
hang
    Sleep for an hour, so ML-PEG times out.
drift
    Report a 10 meV/atom energy drift.
badforces
    Write non-zero forces to ``validation.extxyz``.
badhash
    Report the wrong weights checksum.
fewsteps
    Report only 10 measured steps.
badtime
    Report non-monotonic timestamps.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

MODEL_NAME = "mock"
PRECISIONS = ("fp64", "fp32", "mixed")
WEIGHTS = Path(__file__).with_name("weights.txt")


def parse_args() -> argparse.Namespace:
    """
    Parse exactly the four contract arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def active_faults(run_id: str) -> set[str]:
    """
    Get the faults planted for this run.

    Parameters
    ----------
    run_id
        Run ID.

    Returns
    -------
    set[str]
        Fault names.
    """
    faults = set()
    for entry in os.environ.get("MLPEG_MOCK_FAIL", "").split(";"):
        fault, _, pattern = entry.strip().partition(":")
        if fault and fnmatch(run_id, pattern or "*"):
            faults.add(fault)
    return faults


def write_validation(structure: Path, out: Path, force: float) -> None:
    """
    Write a single-frame extxyz with the input positions and constant forces.

    Parameters
    ----------
    structure
        Input extxyz file.
    out
        Output file.
    force
        Value of every force component, in eV/Å.
    """
    lines = structure.read_text().splitlines()
    n_atoms = int(lines[0])
    header = lines[1]
    lattice = re.search(r'Lattice="([^"]+)"', header).group(1)
    body = []
    for line in lines[2 : 2 + n_atoms]:
        species, x, y, z = line.split()[:4]
        body.append(f"{species} {x} {y} {z} {force} {force} {force}")
    out.write_text(
        f"{n_atoms}\n"
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy=0.0 pbc="T T T"\n' + "\n".join(body) + "\n"
    )


def main() -> int:
    """
    Run the mock.

    Returns
    -------
    int
        Exit code.
    """
    start = time.time()
    args = parse_args()
    faults = active_faults(args.run_id)
    if "error" in faults:
        print("mock: planted error", file=sys.stderr)
        return 1
    if "hang" in faults:
        time.sleep(3600)

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    match = re.fullmatch(r"(.+)-n(\d+)-r(\d+)", args.run_id)
    material, n_atoms = match.group(1), int(match.group(2))
    system = next(s for s in spec["systems"] if s["material"] == material)
    size = next(s for s in system["sizes"] if s["n_atoms"] == n_atoms)
    md = spec["md"]

    weights_sha256 = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if "badhash" in faults:
        weights_sha256 = "0" * 64

    out = Path(args.out)
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "status": "ok",
        "error": None,
        "model": {
            "name": MODEL_NAME,
            "weights_sha256": weights_sha256,
            "precision": args.precision,
        },
        "implementation": {
            "engine": "mock",
            "engine_version": "1.0",
            "repo": None,
            "commit": None,
            "settings": {},
        },
        "system": {"material": material, "n_atoms": n_atoms},
        "timestamps_unix": {"start": start},
        "steps": None,
        "energy": None,
        "self_reported": {"faults": sorted(faults)},
    }

    statuses = {"oom", "unsupported_elements", "skipped"} & faults
    if statuses:
        result["status"] = statuses.pop()
        result["error"] = f"mock: planted {result['status']}"
        result["timestamps_unix"]["setup_end"] = time.time()
    elif "nojson" not in faults:
        phase = float(os.environ.get("MLPEG_MOCK_PHASE_S", "0.05"))
        time.sleep(phase)
        setup_end = time.time()
        if size.get("validate"):
            force = 0.1 if "badforces" in faults else 0.0
            write_validation(
                spec_path.parent / size["file"], out / "validation.extxyz", force
            )
        time.sleep(phase)
        warmup_end = time.time()
        time.sleep(phase + n_atoms * 1e-7)
        measure_end = time.time()
        if "badtime" in faults:
            warmup_end, measure_end = measure_end, warmup_end
        drift = 0.01 * n_atoms if "drift" in faults else 0.0
        result["timestamps_unix"].update(
            setup_end=setup_end, warmup_end=warmup_end, measure_end=measure_end
        )
        result["steps"] = {
            "warmup": md["warmup_steps"],
            "measured": 10 if "fewsteps" in faults else md["measured_steps"],
        }
        result["energy"] = {"total_start_eV": -1.0, "total_end_eV": -1.0 + drift}

    if "nojson" not in faults:
        (out / "inference.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
