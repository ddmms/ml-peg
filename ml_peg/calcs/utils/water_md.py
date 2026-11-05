"""Checkpoint-aware execution for long water benchmark simulations."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase.io import iread, read, write
from janus_core.calculations.md import NVT


def prepare_restart(atoms, directory: Path, equil_steps: int, total_steps: int):
    """
    Restore a checkpoint with its calculator and trim outputs to that step.

    Automatic Janus restart can replace the attached calculator. Load the structure
    explicitly instead, and keep constraints and masses from the benchmark input.
    Trajectories after the checkpoint are discarded to avoid duplicate timesteps.

    Parameters
    ----------
    atoms : ase.Atoms
        Initial structure with calculator, masses, constraints, charge and spin.
    directory : pathlib.Path
        Directory containing the checkpoint, statistics and trajectory files.
    equil_steps : int
        Number of equilibration steps before production frames are written.
    total_steps : int
        Target number of steps, including equilibration and production.

    Returns
    -------
    ase.Atoms
        Restored structure with the original calculator and input settings, or the
        initial structure if no valid checkpoint is available.
    int
        Checkpoint step offset, or zero for a fresh run.
    """
    checkpoints = sorted(
        directory.glob("md-res-*.extxyz"),
        key=lambda p: int(p.stem.rsplit("-", 1)[1]),
        reverse=True,
    )
    stats = directory / "md-stats.dat"
    trajectory = directory / "md-traj.extxyz"
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.rsplit("-", 1)[1])
        if step >= total_steps or not stats.exists():
            continue
        try:
            restored = read(checkpoint)
            if (
                len(restored) != len(atoms)
                or not (restored.numbers == atoms.numbers).all()
            ):
                raise ValueError("Checkpoint composition differs from benchmark input")
            lines = stats.read_text().splitlines(keepends=True)
            kept = []
            for line in lines:
                fields = line.split()
                if not fields or not fields[0].isdigit():
                    kept.append(line)
                elif int(fields[0]) <= step:
                    kept.append(line)
            if not kept or int(kept[-1].split()[0]) != step:
                raise ValueError("No statistics record at checkpoint step")
            temporary = trajectory.with_suffix(".restart.extxyz")
            temporary.write_text("")
            last_step = None
            if step >= equil_steps:
                for frame in iread(trajectory, index=":"):
                    frame_step = int(frame.info["step"])
                    if frame_step > step:
                        break
                    write(temporary, frame, format="extxyz", append=True)
                    last_step = frame_step
                    if frame_step == step:
                        break
                if last_step != step:
                    raise ValueError("Trajectory does not reach checkpoint step")
            temporary.replace(trajectory)
            stats.write_text("".join(kept))
            restored.set_constraint(atoms.constraints)
            restored.set_masses(atoms.get_masses())
            restored.pbc = atoms.pbc
            restored.info.update(
                charge=int(atoms.info["charge"]), spin=int(atoms.info["spin"])
            )
            restored.calc = atoms.calc
            return restored, step
        except (OSError, ValueError, KeyError, IndexError) as exc:
            warn(f"Cannot resume {checkpoint}: {exc}", stacklevel=2)
    return atoms, 0


def run_water_md(
    atoms,
    directory: Path,
    *,
    equil_steps: int,
    run_steps: int,
    timestep: float,
    temperature: float,
    friction: float,
    traj_every: int,
) -> None:
    """
    Run or resume MD, leaving a failure marker until successful completion.

    Parameters
    ----------
    atoms : ase.Atoms
        Initial structure with calculator, masses, constraints, charge and spin.
    directory : pathlib.Path
        Existing directory in which to write trajectories, statistics and checkpoints.
    equil_steps : int
        Number of equilibration steps before production sampling.
    run_steps : int
        Number of production steps after equilibration.
    timestep : float
        MD integration timestep in femtoseconds.
    temperature : float
        Target thermostat temperature in kelvin.
    friction : float
        Langevin friction coefficient in inverse femtoseconds, as accepted by Janus.
    traj_every : int
        Number of MD steps between recorded production frames.
    """
    marker = directory / "md-failed"
    marker.touch()
    try:
        total = equil_steps + run_steps
        atoms, offset = prepare_restart(atoms, directory, equil_steps, total)
        md = NVT(
            struct=atoms,
            temp=temperature,
            steps=total,
            equil_steps=max(0, equil_steps - offset),
            timestep=timestep,
            friction=friction,
            stats_every=100,
            traj_every=traj_every,
            traj_start=max(0, equil_steps - offset),
            file_prefix=directory / "md",
            restart=bool(offset),
            restart_auto=False,
            restart_stem=directory / "md-res",
            restart_every=1000,
            rotate_restart=True,
            restarts_to_keep=3,
            write_kwargs={"columns": ["symbols", "positions", "momenta", "masses"]},
        )
        md.run()
        marker.unlink(missing_ok=True)
    except Exception as exc:
        warn(f"Error running MD in {directory}: {exc}", stacklevel=2)
