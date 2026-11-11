"""
Run NPT MD of binary liquid mixtures and their pure components.

The 49 boxes (30 mixture compositions of 10 binaries and 19 pure liquids) were
pre-equilibrated with OpenFF Sage at the experimental density. Each box is run
for 0.2 ns of equilibration followed by 1 ns of production at the box's
``exp_temperature`` and 1 atm. Scalars are logged every 100 fs to a CSV for the
enthalpy of mixing and density analysis; full frames are saved every 10 ps for
restarts and structural checks.
"""

from __future__ import annotations

import csv
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

from ase import Atoms, units
from ase.io import Trajectory, read
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import pandas as pd
import pytest
from scipy import constants

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.data.data import exists
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"
EQUILIBRATION_PS = 200.0
PRODUCTION_PS = 1000.0
DATA_ZIP = "binary_mixtures.zip"
DATA_KEY = f"inputs/molecular_dynamics/binary_mixtures/{DATA_ZIP}"
# Zip of one model's outputs/<model>/ directory, uploaded once all boxes are run
RUNS_KEY = "outputs/molecular_dynamics/binary_mixtures/{model}.zip"
PS = constants.pico * units.s
TIMESTEP = 1 * units.fs
TDAMP = 50 * units.fs  # thermostat time constant
PDAMP = 500 * units.fs  # barostat time constant
PRECISION = "low"  # ASE calculator precision
LOG_INTERVAL = 100  # steps between scalar CSV rows
TRAJ_INTERVAL = 10000  # steps between saved frames (restart points)
PRESSURE_EV_PER_A3 = constants.atm * units.Pascal
# Converts from ASE units (amu/Å³) by multiplication
AMU_PER_A3_TO_G_PER_CM3 = (constants.centi * units.m) ** 3 / (
    constants.milli * units.kg
)


class LogRow(NamedTuple):
    """
    One row of the scalar log; the field names are the CSV header.

    Values other than ``step`` are stored already formatted for the CSV. The
    unit suffixes break the naming convention, but the names must match the
    Sage reference logs and the analysis.
    """

    step: int
    time_ps: str
    epot_eV: str  # noqa: N815
    ekin_eV: str  # noqa: N815
    temperature_K: str  # noqa: N815
    volume_A3: str  # noqa: N815
    density_g_cm3: str


def get_density_g_cm3(atoms: Atoms) -> float:
    """
    Get the density of the system in g/cm^3.

    Parameters
    ----------
    atoms
        Periodic system.

    Returns
    -------
    float
        Density in g/cm^3.
    """
    return (atoms.get_masses().sum() / atoms.get_volume()) * AMU_PER_A3_TO_G_PER_CM3


@cache
def runs_on_s3(model_name: str) -> bool:
    """
    Check whether a model's pre-computed runs can be downloaded from S3.

    Parameters
    ----------
    model_name
        Model name.

    Returns
    -------
    bool
        Whether the zip of the model's outputs is on S3.
    """
    return exists(RUNS_KEY.format(model=model_name))


def restart_state(traj_path: Path, csv_path: Path) -> tuple[Atoms | None, int]:
    """
    Find the last saved frame and trim the scalar log to match it.

    A job killed while writing a frame leaves the previous frames readable, as
    ASE updates the frame count only after the frame's data. An unreadable
    trajectory is therefore corrupt, and the error is raised rather than
    discarding the run.

    Parameters
    ----------
    traj_path
        Trajectory with one frame every ``TRAJ_INTERVAL`` steps.
    csv_path
        Scalar log.

    Returns
    -------
    tuple[Atoms | None, int]
        Last frame (``None`` for a fresh start) and its step number.
    """
    # A 0-byte trajectory is left by a job that stopped before the first frame,
    # e.g. while loading the model
    if not csv_path.exists() or not traj_path.exists() or not traj_path.stat().st_size:
        return None, 0
    with Trajectory(traj_path) as traj:
        n_frames = len(traj)
        if n_frames == 0:  # killed before the first frame was written
            return None, 0
        atoms = traj[-1]
    step = (n_frames - 1) * TRAJ_INTERVAL
    # Keep only the rows up to the last frame: the steps after it are rerun and
    # logged again
    with open(csv_path) as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["step"]) <= step]
    # Rewrite the log with just those rows ("w" empties the file first)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LogRow._fields)
        writer.writeheader()
        writer.writerows(rows)
    return atoms, step


def run_npt(atoms: Atoms, calc, out_stem: Path, run_ps: float) -> None:
    """
    Run (or continue) NPT MD with the isotropic MTK barostat.

    Parameters
    ----------
    atoms
        Starting structure, with ``exp_temperature`` in ``atoms.info``.
    calc
        ASE calculator.
    out_stem
        Output path without suffix; ``.csv`` and ``.traj`` are written.
    run_ps
        Total run length in ps.
    """
    traj_path, csv_path = out_stem.with_suffix(".traj"), out_stem.with_suffix(".csv")
    temperature = float(atoms.info["exp_temperature"])
    info = dict(atoms.info)
    restart, nsteps = restart_state(traj_path, csv_path)
    if restart is None:
        traj_path.unlink(missing_ok=True)
        with open(csv_path, "w", newline="") as handle:
            csv.writer(handle).writerow(LogRow._fields)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        Stationary(atoms)
    else:
        atoms = restart
        atoms.info.update(info)

    atoms.calc = calc
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    dyn = IsotropicMTKNPT(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=temperature,
        pressure_au=PRESSURE_EV_PER_A3,
        tdamp=TDAMP,
        pdamp=PDAMP,
    )
    dyn.nsteps = nsteps

    def is_restart_step() -> bool:
        """
        Check whether this is the already-saved first step of a restart.

        Returns
        -------
        bool
            ``True`` if this step was logged before the restart.
        """
        return restart is not None and dyn.nsteps == nsteps

    def log_scalars() -> None:
        """Append one row of scalars to the CSV."""
        if is_restart_step():
            return
        writer.writerow(
            LogRow(
                step=dyn.nsteps,
                time_ps=f"{dyn.nsteps * TIMESTEP / PS:.4f}",
                epot_eV=f"{atoms.get_potential_energy():.6f}",
                ekin_eV=f"{atoms.get_kinetic_energy():.6f}",
                temperature_K=f"{atoms.get_temperature():.3f}",
                volume_A3=f"{atoms.get_volume():.4f}",
                density_g_cm3=f"{get_density_g_cm3(atoms):.6f}",
            )
        )

    def save_frame() -> None:
        """Append the current frame to the trajectory, after the logged rows."""
        if not is_restart_step():
            # A restart resumes from the last frame, so every row up to it must be
            # on disk first, or a killed job leaves a gap in the log
            handle.flush()
            traj.write(atoms)

    total_steps = round(run_ps * PS / TIMESTEP)
    with (
        open(csv_path, "a", newline="") as handle,
        Trajectory(traj_path, "a", atoms) as traj,
    ):
        writer = csv.writer(handle)
        dyn.attach(log_scalars, interval=LOG_INTERVAL)
        dyn.attach(save_frame, interval=TRAJ_INTERVAL)
        dyn.run(steps=total_steps - nsteps)


@pytest.mark.multi_day
@pytest.mark.parametrize("mlip", MODELS.items())
def test_binary_mixtures(
    mlip: tuple[str, Any], box_id: int, run_ps: float | None, recompute: bool
) -> None:
    """
    Run one binary mixtures box.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    box_id
        Index of the box to simulate, in ``boxes.csv`` sorted by ``box_id``.
    run_ps
        Total NPT length (equilibration plus production) in ps. ``None`` runs
        the benchmark protocol, ``EQUILIBRATION_PS + PRODUCTION_PS``.
    recompute
        Whether to run the box even if the model's runs are already on S3.
    """
    model_name, model = mlip
    out_dir = OUT_PATH / model_name
    # Only defer to S3 for models with no local outputs: once any box has been run
    # here (e.g. with --recompute), keep running and restarting its boxes
    if not recompute and not any(out_dir.glob("*.csv")) and runs_on_s3(model_name):
        pytest.skip(
            f"Pre-computed runs are on S3 at {RUNS_KEY.format(model=model_name)}, "
            "and are downloaded by the analysis; pass --recompute to rerun"
        )

    if run_ps is None:
        run_ps = EQUILIBRATION_PS + PRODUCTION_PS
    data_path = (
        download_s3_data(
            filename=DATA_ZIP,
            key=DATA_KEY,
        )
        / "binary_mixtures"
    )
    # Index boxes.csv rather than globbing, so stale boxes left in the cache by an
    # older zip extracted to the same directory are never picked up
    box_ids = sorted(pd.read_csv(data_path / "boxes.csv").box_id)
    assert box_id in range(len(box_ids)), (
        f"box_id must be between 0 and {len(box_ids) - 1}"
    )
    box_path = data_path / "boxes" / f"{box_ids[box_id]}.xyz"

    calc = model.get_calculator(precision=PRECISION)
    calc = model.add_d3_calculator(calc)

    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(box_path)
    assert atoms.pbc.all(), f"{box_path.name} is not periodic"

    run_npt(atoms, calc, out_dir / box_path.stem, run_ps)
