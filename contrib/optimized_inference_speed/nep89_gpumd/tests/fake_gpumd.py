"""Stand-in for the ``gpumd`` binary, replaying captured v5.9.1 output.

Behaviour is selected with ``FAKE_GPUMD_MODE``:

- ``ok``: replay the stdout fixture with sleeps matching its run timers and
  write the ``thermo.out`` fixture. A ``time_step 0`` run.in writes ``step0.xyz``.
- ``oom``: CUDA out-of-memory error on stderr, exit 1.
- ``error``: GPUMD input error on stderr, exit 1.
- ``multigpu``: like ``ok`` but reports two GPUs.
- ``slow_gap``: like ``ok`` with a 1.5 s pause between the two runs.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import time

from calorine.gpumd import read_xyz

DATA = Path(__file__).parent / "data"
MODE = os.environ.get("FAKE_GPUMD_MODE", "ok")
STEP0_ENERGY = -123.456


def emit(line: str) -> None:
    """
    Print one stdout line and flush, like GPUMD under ``stdbuf -oL``.

    Parameters
    ----------
    line
        Line without trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def fake_forces(index: int) -> tuple[float, float, float]:
    """
    Deterministic per-atom forces for the step-0 dump.

    Parameters
    ----------
    index
        Atom index.

    Returns
    -------
    tuple[float, float, float]
        Force components.
    """
    return (0.001 * index, -0.002 * index, 0.5)


def write_step0() -> None:
    """Write ``step0.xyz`` in GPUMD's dump_xyz format from ``model.xyz``."""
    model = read_xyz("model.xyz")
    n_atoms = len(model)
    cell = model.cell.array
    lattice = " ".join(f"{x:.16e}" for x in cell.ravel())
    out = [
        str(n_atoms),
        f'Time=0.00000000 pbc="T T T" Lattice="{lattice}" '
        f"energy={STEP0_ENERGY:.16e} "
        'virial="0 0 0 0 0 0 0 0 0" stress="0 0 0 0 0 0 0 0 0" '
        "Properties=species:S:1:pos:R:3:forces:R:3:energy_atom:R:1",
    ]
    # GPUMD wraps positions into the box; mimic that.
    wrapped = model.get_positions(wrap=True)
    for i, (symbol, pos) in enumerate(zip(model.symbols, wrapped, strict=True)):
        fx, fy, fz = fake_forces(i)
        out.append(
            f"{symbol} {pos[0]:.16e} {pos[1]:.16e} {pos[2]:.16e} "
            f"{fx:.16e} {fy:.16e} {fz:.16e} {STEP0_ENERGY / n_atoms:.16e}"
        )
    Path("step0.xyz").write_text("\n".join(out) + "\n")


def main() -> int:
    """
    Run the fake.

    Returns
    -------
    int
        Exit code.
    """
    run_in = Path("run.in").read_text()
    if "time_step 0" in run_in:
        emit("Run 1 steps.")
        write_step0()
        return 0
    if MODE == "oom":
        emit("Started running GPUMD.")
        sys.stderr.write((DATA / "stderr_oom_v5.9.1.txt").read_text())
        return 1
    if MODE == "error":
        sys.stderr.write((DATA / "stderr_input_error_v5.9.1.txt").read_text())
        return 1

    runs_done = 0
    for line in (DATA / "gpumd_stdout_v5.9.1.txt").read_text().splitlines():
        if MODE == "multigpu" and line.startswith("number of GPUs"):
            line = "number of GPUs = 2"
        if line.startswith("Time used for this run"):
            # Spend the time GPUMD reports, so the stdout clock is consistent.
            time.sleep(float(line.split("=")[1].split()[0]))
            runs_done += 1
        if MODE == "slow_gap" and runs_done == 1 and line.startswith("Run 1000"):
            time.sleep(1.5)
        emit(line)
    shutil.copy(DATA / "thermo_v5.9.1.out", "thermo.out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
