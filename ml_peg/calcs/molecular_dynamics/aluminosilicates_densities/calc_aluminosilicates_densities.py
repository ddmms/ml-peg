"""
Melt-quench of albite, anorthite, and sanidine for the ML-PEG benchmark.

Protocol:
- Velocities initialized at T_HOLD (2500K) - no heating ramp.
- NVT hold at 2500K for 5 ps (volume fixed, stabilizes random packed structure).
- NPT hold at 2500K for 20 ps to ensure full melting.
- NPT staircase cooling 2500 -> 300K at 5 K/ps (50K steps, 10 ps/rung).
- NPT hold at 300K for 20 ps (production run for density averaging).
- Barostat: NPTBerendsen (ASE native isotropic - cubic cell only).
- Thermostat: NPTBerendsen (taut=taup=25 fs).
- Bulk modulus estimate: 20 GPa (melt/glass compromise for aluminosilicates).

System size: ~350 atoms per composition.
Density is averaged over the 20 ps production run at 300K.

Models included (15 total):
  .venv:          mace-mp-0a, mace-mp-0b3, mace-mpa-0, mace-omat-0,
                  mace-matpes-r2scan, mace-mh-1-omat, mace-omol, mace-mh-1-omol,
                  mace-polar-1-s, mace-polar-1-m, mace-polar-1-l,
                  orb-v3-consv-inf-omat, pet-mad
  .venv_mattersim: mattersim-5M
  .venv_uma:      uma-s-1p1-omat

Models excluded:
  - MACE-OFF23(L):      Na/Al/Si/K/Ca not in supported elements
  - orb-v3-consv-omol:  molecular model, requires charge/spin in atoms.info
  - GRACE-2L-OAM:       CPU-only (~510h/run)
  - uma-m-1p1-omat:     ~124h/run, exceeds practical limit
  - uma-s-1p2-omat:     not available in current ml-peg models.yml
  - uma-*-omol:         molecular model

Models not completed:
  - mace-omol, mace-polar-1-l: only a few tens of MD steps in several hours
       (float32 and float64), too slow to run the full protocol.

Precision notes:
  All MACE models run in float32 (PRECISION_OVERRIDES) for GPU performance.
  ORB runs in float64 (ml-peg default via OrbCalc, no override needed).
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import warnings

from ase import units
from ase.io import read, write
from ase.io import write as ase_write
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import numpy as np
import pytest

from ml_peg.models.get_models import load_models

ALL_MODELS = [
    "mace-mp-0a",
    "mace-mp-0b3",
    "mace-mpa-0",
    "mace-omat-0",
    "mace-matpes-r2scan",
    "mace-mh-1-omat",
    "mace-omol",
    "mace-mh-1-omol",
    "mace-polar-1-s",
    "mace-polar-1-m",
    "mace-polar-1-l",
    "orb-v3-consv-inf-omat",
    "pet-mad",
    "mattersim-5M",
    "uma-s-1p1-omat",
]

MODELS = load_models(ALL_MODELS)

# All MACE models use float32 for GPU performance (recommended for MD).
# ORB and UMA handle precision internally.
PRECISION_OVERRIDES = {
    "mace-mp-0a": "low",
    "mace-mp-0b3": "low",
    "mace-mpa-0": "low",
    "mace-omat-0": "low",
    "mace-matpes-r2scan": "low",
    "mace-mh-1-omat": "low",
    "mace-mh-1-omol": "low",
    "mace-polar-1-l": "low",
    "mace-omol": "low",
    "mace-polar-1-s": "low",
    "mace-polar-1-m": "low",
    "mattersim-5M": "low",
}


def get_calc(model_name, model):
    """
    Return the calculator with appropriate precision for the model.

    Parameters
    ----------
    model_name : str
        The name/identifier of the MLIP model.
    model : object
        The model wrapper object providing the `get_calculator` method.

    Returns
    -------
    Calculator
        The initialized ASE calculator with the appropriate precision settings.
    """
    precision = PRECISION_OVERRIDES.get(model_name, "high")
    return model.get_calculator(precision=precision)


DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(os.environ.get("MLPEG_OUT_PATH", Path(__file__).parent / "outputs"))

COMPOSITIONS = ["albite", "anorthite", "sanidine"]
N_REPLICAS = 3

# ---------------------------------------------------------------------------
# MD protocol parameters
# ---------------------------------------------------------------------------
T_START = 300.0
T_HOLD = 2500.0
T_STEP = 50.0
RAMP_RATE = 5.0  # K/ps
TIMESTEP_PS = 0.001  # 1 fs

TAUT = 25 * units.fs  # thermostat time constant
TAUP = 25 * units.fs  # barostat time constant
PRESSURE_AU = 0.0

B_ESTIMATE = 20 * units.GPa
COMPRESSIBILITY_AU = 1 / B_ESTIMATE

NVT_HOT_PS = 5.0
HOLD_HOT_PS = 20.0
HOLD_COLD_PS = 20.0
NVT_HOT_STEPS = int(NVT_HOT_PS / TIMESTEP_PS)
HOLD_HOT_STEPS = int(HOLD_HOT_PS / TIMESTEP_PS)
HOLD_COLD_STEPS = int(HOLD_COLD_PS / TIMESTEP_PS)
STEPS_PER_RUNG = int((T_STEP / RAMP_RATE) / TIMESTEP_PS)


def save_checkpoint(atoms, path: Path, stages_done: int) -> None:
    """
    Save the state of the run after a finished stage.

    Only the last checkpoint is kept. A non-finite state is never saved, so an
    exploded run stops here instead of continuing for hours.

    Parameters
    ----------
    atoms : Atoms
        Structure (positions, cell and momenta) at the end of the stage.
    path : Path
        Checkpoint file.
    stages_done : int
        Number of stages completed so far.

    Returns
    -------
    None
        This function does not return a value.
    """
    positions_ok = np.isfinite(atoms.get_positions()).all()
    if not (positions_ok and np.isfinite(atoms.get_volume())):
        raise RuntimeError(f"Non-finite positions or volume after stage {stages_done}")
    atoms.info["stages_done"] = stages_done
    tmp = path.with_name(path.stem + "_tmp.extxyz")
    ase_write(tmp, atoms)
    tmp.replace(path)  # a kill during writing cannot corrupt the checkpoint


def run_melt_quench(atoms, model_name, model, log_prefix: Path, seed: int = 0):
    """
    Run the full melt-quench protocol on a single structure.

    A checkpoint is saved after every stage. If a checkpoint exists, the run
    resumes from the last finished stage.

    Parameters
    ----------
    atoms : Atoms
        The input ASE Atoms structure.
    model_name : str
        The name/identifier of the MLIP model.
    model : object
        The model wrapper object.
    log_prefix : Path
        Path prefix for log and output files.
    seed : int
        Seed of the random generator used for the initial velocities.

    Returns
    -------
    Atoms
        The relaxed structure with density metadata (`density_final_gcm3`,
        `density_std_gcm3`, `precision`) stored in `atoms.info`.
    """
    # stages_done counts the finished stages:
    # 1 = NVT, 2 = NPT hold at 2500 K, 3-46 = the 44 cooling steps
    ckpt = Path(str(log_prefix) + "_checkpoint.extxyz")
    stages_done = 0
    if ckpt.exists():
        atoms = read(ckpt)  # positions, cell and momenta of the last finished stage
        stages_done = int(atoms.info["stages_done"])
        print(
            f"Resuming {log_prefix.name} after {stages_done} finished stages",
            flush=True,
        )

    # Neutral, closed-shell system: total charge 0, spin multiplicity 1.
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = get_calc(model_name, model)

    dt = TIMESTEP_PS * 1000 * units.fs

    # --- Stage 1: NVT at 2500 K, 5 ps ---
    if stages_done == 0:
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=T_HOLD, rng=np.random.default_rng(seed)
        )
        traj_nvt_path = str(log_prefix) + "_00_hold_hot.extxyz"
        nvt = NVTBerendsen(
            atoms,
            dt,
            temperature_K=T_HOLD,
            taut=TAUT,
            logfile=str(log_prefix) + "_00_nvt_hot.log",
            loginterval=1000,
        )
        ase_write(traj_nvt_path, atoms)

        # Progress check: warn if the NVT stage is progressing very slowly
        step_counter = {"steps": 0}
        stop_watchdog = threading.Event()

        def watchdog_timer(max_minutes=200, target_steps=1000):
            """
            Warn if the simulation appears to be progressing very slowly.

            Parameters
            ----------
            max_minutes : int, default=200
                Time in minutes after which progress is checked.
            target_steps : int, default=1000
                Number of steps expected to be completed within that time.

            Returns
            -------
            None
                This function runs in a background thread and returns nothing.
            """
            start_time = time.time()
            while not stop_watchdog.is_set():
                time.sleep(30)
                elapsed_min = (time.time() - start_time) / 60
                if elapsed_min > max_minutes and step_counter["steps"] < target_steps:
                    ps_done = step_counter["steps"] * TIMESTEP_PS
                    print(
                        f"[WARNING] Slow run: only {step_counter['steps']} steps "
                        f"({ps_done:.3f} ps) in {elapsed_min:.1f} minutes. "
                        f"Check {log_prefix}_00_nvt_hot.log.",
                        flush=True,
                    )
                    return

        watchdog_thread = threading.Thread(target=watchdog_timer, daemon=True)
        watchdog_thread.start()

        nvt.attach(
            lambda: step_counter.update({"steps": step_counter["steps"] + 10}),
            interval=10,
        )
        nvt.attach(lambda: ase_write(traj_nvt_path, atoms, append=True), interval=1000)

        try:
            nvt.run(NVT_HOT_STEPS)
        finally:
            stop_watchdog.set()
        save_checkpoint(atoms, ckpt, 1)

    # --- Stage 2: NPT hold at 2500 K, 20 ps ---
    if stages_done <= 1:
        traj_hot_path = str(log_prefix) + "_01_hold_hot.extxyz"
        Path(traj_hot_path).unlink(missing_ok=True)  # frames of an interrupted try
        dyn = NPTBerendsen(
            atoms,
            dt,
            temperature_K=T_HOLD,
            pressure_au=PRESSURE_AU,
            taut=TAUT,
            taup=TAUP,
            compressibility_au=COMPRESSIBILITY_AU,
            logfile=str(log_prefix) + "_01_hold_hot.log",
            loginterval=1000,
        )
        dyn.attach(lambda: ase_write(traj_hot_path, atoms, append=True), interval=1000)
        dyn.run(HOLD_HOT_STEPS)
        save_checkpoint(atoms, ckpt, 2)

    # --- Stages 3-46: NPT staircase cooling 2500 -> 300 K at 5 K/ps ---
    temps = range(int(T_HOLD) - int(T_STEP), int(T_START) - 1, -int(T_STEP))
    for rung, temp in enumerate(temps):
        if stages_done > 2 + rung:  # this step was already finished
            continue
        dyn = NPTBerendsen(
            atoms,
            dt,
            temperature_K=temp,
            pressure_au=PRESSURE_AU,
            taut=TAUT,
            taup=TAUP,
            compressibility_au=COMPRESSIBILITY_AU,
            logfile=str(log_prefix) + f"_02_cool_{int(temp)}K.log",
            loginterval=1000,
        )
        dyn.run(STEPS_PER_RUNG)
        save_checkpoint(atoms, ckpt, 3 + rung)

    # --- NPT at 300 K for 20 ps (production run) ---
    traj_cold_path = str(log_prefix) + "_03_hold_cold.extxyz"
    Path(traj_cold_path).unlink(missing_ok=True)
    densities = []

    def write_and_record() -> None:
        """
        Write current frame to trajectory and calculate density.

        Returns
        -------
        None
            Appends trajectory and density values in-place.
        """
        ase_write(traj_cold_path, atoms, append=True)
        vol_a3 = atoms.get_volume()
        mass_g = sum(atoms.get_masses()) * 1.66054e-24
        rho = mass_g / (vol_a3 * 1e-24)
        densities.append(rho)
        print(f"  density: {rho:.4f} g/cm3 | vol: {vol_a3:.2f} A3", flush=True)

    dyn = NPTBerendsen(
        atoms,
        dt,
        temperature_K=T_START,
        pressure_au=PRESSURE_AU,
        taut=TAUT,
        taup=TAUP,
        compressibility_au=COMPRESSIBILITY_AU,
        logfile=str(log_prefix) + "_03_hold_cold.log",
        loginterval=1000,
    )
    dyn.attach(write_and_record, interval=1000)
    dyn.run(HOLD_COLD_STEPS)

    # --- Final density: mean and std over production run ---
    density_mean = np.mean(densities)
    density_std = np.std(densities)
    if not np.isfinite(density_mean):
        raise RuntimeError("Non-finite density at the end of the production run")
    atoms.info.pop("stages_done", None)
    atoms.info["density_final_gcm3"] = density_mean
    atoms.info["density_std_gcm3"] = density_std
    atoms.info["precision"] = PRECISION_OVERRIDES.get(model_name, "high")
    print(
        f"Final density ({model_name}, {log_prefix.parent.name}): "
        f"{density_mean:.4f} +/- {density_std:.4f} g/cm3",
        flush=True,
    )
    return atoms


@pytest.mark.multi_day
@pytest.mark.parametrize("mlip", MODELS.items(), ids=list(MODELS.keys()))
@pytest.mark.parametrize("composition", COMPOSITIONS)
@pytest.mark.parametrize("replica", range(N_REPLICAS))
def test_aluminosilicates_densities(mlip, composition, replica) -> None:
    """
    Run melt-quench for one model/composition/replica combination.

    Skipped if a valid output file already exists. A failed run is written as a
    structure with a NaN density, and is retried at the next launch.

    Parameters
    ----------
    mlip : tuple
        A tuple of (model_name, model) specifying the MLIP model.
    composition : str
        The aluminosilicate name.
    replica : int
        The replica/seed index for the simulation.

    Returns
    -------
    None
        This test function does not return a value.
    """
    model_name, model = mlip
    write_dir = OUT_PATH / model_name / composition
    write_dir.mkdir(parents=True, exist_ok=True)

    struct_path = write_dir / f"replica{replica}_quenched.xyz"
    start_path = DATA_PATH / f"{composition}_start_seed{replica}.xyz"
    if struct_path.exists() and not read(struct_path).info.get("failed", False):
        return  # already completed, skip

    try:
        atoms = run_melt_quench(
            read(start_path),
            model_name,
            model,
            write_dir / f"replica{replica}",
            seed=replica,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Melt-quench failed for {model_name}, {composition}, "
            f"replica {replica}: {exc}",
            stacklevel=2,
        )
        atoms = read(start_path)  # clean structure as failure marker
        atoms.info["density_final_gcm3"] = float("nan")
        atoms.info["failed"] = True
        atoms.info["failure"] = str(exc).replace("\n", " ")[:200]
        write(struct_path, atoms)
        return  # the checkpoint is kept, so the next launch resumes

    write(struct_path, atoms)
    checkpoint = write_dir / f"replica{replica}_checkpoint.extxyz"
    checkpoint.unlink(missing_ok=True)
