"""
Reference MD protocol for the WiSE 21 m LiTFSI/H2O benchmark.

This script reproduces, with janus-core, the same MD protocol used to
generate the production trajectories analysed by the consolidated
litfsi_h2o_21m benchmark. Production trajectories were run externally
with LAMMPS + MACE (symmetrix/Kokkos) on Adastra (MI250X); the driving
scripts are ``in.pipeline_cpu.lmp`` (Min -> NVT -> NPT) and
``in.nvt_continue.lmp`` (NVT continuation) on the Adastra workdir.

Integrators are matched to LAMMPS where janus-core exposes a choice:

  * NVT: Nosé-Hoover chain (``NVT_NH``) -- equivalent to LAMMPS ``fix nvt``.
  * NPT: Martyna-Tobias-Klein chain (``NPT_MTK``) -- the same formulation
    used by LAMMPS ``fix npt``. (``NPT`` in janus is Melchionna and would
    *not* match.)

The protocol is intentionally marked ``pytest.mark.very_slow``: at ~1500
atoms and 250+ ps of cumulative MD, an ASE/Janus run takes a day of GPU
time per registered model. Run manually with

    python calc_md_reference.py <model_name>

or call :func:`run_reference_md` from your own script.

Protocol summary
----------------
System     : 64 LiTFSI + 170 H2O  (1534 atoms, 21 m)
Ensembles  : Min (FIRE) -> NVT 50 ps (NH chain) ->
             NPT 200 ps (MTK chain, iso) ->
             optional NVT continuation 50 ps (NH chain)
Temperature: 298.15 K
Pressure   : 1.01325 bar (1 atm)
Timestep   : 0.5 fs
TDAMP/PDAMP: 50 fs / 500 fs (100*dt / 1000*dt, Nosé-Hoover damping)
NH chains  : length 3 for both thermostat and barostat (LAMMPS default)
Dump cadence: every 0.1 ps (= 200 steps)
References : Gilbert et al., JCED 62, 2056 (2017);
             Watanabe et al., JPCB 125, 7477 (2021).
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# --- Physical parameters (mirror Adastra LAMMPS in.pipeline_cpu.lmp) ---------

TEMPERATURE_K = 298.15
PRESSURE_BAR = 1.01325             # 1 atm
TIMESTEP_FS = 0.5

# 1 bar = 1e-4 GPa (Janus NPT accepts pressure in GPa)
PRESSURE_GPA = PRESSURE_BAR * 1e-4

# Nosé-Hoover damping times (match LAMMPS `fix npt`/`fix nvt` TDAMP/PDAMP)
THERMOSTAT_TIME_FS = 50.0          # 100 * timestep
BAROSTAT_TIME_FS = 500.0           # 1000 * timestep

# Step counts (at 0.5 fs/step)
NVT_STEPS = 100_000                # 50 ps  (warm-up / equilibration)
NPT_STEPS = 400_000                # 200 ps (density + configurational sampling)
NVT_CONT_STEPS = 100_000           # 50 ps  (optional continuation)

# IO cadence (match LAMMPS THERMO_EVERY / DUMP_EVERY)
STATS_EVERY = 100                  # 0.05 ps
TRAJ_EVERY = 200                   # 0.1 ps  (501 frames in 50 ps)

# Minimization (match LAMMPS `min_style fire; minimize 1.0e-6 0.2 2000 20000`)
MIN_FMAX = 0.2                     # eV/Å
MIN_STEPS = 2000

SEED = 42

INITIAL_STRUCTURE = DATA_PATH / "p64_w170_initial.xyz"


def load_initial_structure() -> Atoms:
    """
    Load the 64 LiTFSI + 170 H2O cubic simulation box.

    Returns
    -------
    Atoms
        ASE Atoms at the experimental density (L = 27.4938 Å, 1534 atoms).

    Raises
    ------
    FileNotFoundError
        If the reference structure file is not present. The file is not
        shipped with the repository; provide an equivalent p64_w170 box
        (LAMMPS data file converted to .xyz) or point ``INITIAL_STRUCTURE``
        at your own starting configuration.
    """
    if not INITIAL_STRUCTURE.exists():
        raise FileNotFoundError(
            f"Reference initial structure not found at {INITIAL_STRUCTURE}. "
            "Provide an equivalent p64_w170 LAMMPS box (converted to extxyz) "
            "or point INITIAL_STRUCTURE at your own starting configuration."
        )
    return read(INITIAL_STRUCTURE)


def run_reference_md(
    model_name: str,
    model: Any,
    *,
    run_continuation: bool = False,
) -> None:
    """
    Run the Min -> NVT -> NPT reference protocol for one registered model.

    Parameters
    ----------
    model_name
        Registry name of the MLIP model.
    model
        Model object returned by :func:`load_models`; must expose
        ``default_dtype`` and ``get_calculator``.
    run_continuation
        If True, append a further 50 ps of NVT after the NPT block
        (analogue of ``in.nvt_continue.lmp``). Default False.
    """
    # Lazy imports: janus_core pulls in torch, which may be absent in
    # environments that only run the analysis stack.
    from janus_core.calculations.geom_opt import GeomOpt
    from janus_core.calculations.md import NPT_MTK, NVT_NH

    model.default_dtype = "float64"
    calc = model.get_calculator()

    struct = load_initial_structure()
    struct.calc = copy(calc)

    model_out = OUT_PATH / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    # -- Minimization (FIRE, loose tolerance) ---------------------------------
    GeomOpt(
        struct=struct,
        fmax=MIN_FMAX,
        optimizer="FIRE",
        steps=MIN_STEPS,
        write_traj=False,
        file_prefix=model_out / "minimize",
    ).run()

    # -- NVT equilibration, 50 ps ---------------------------------------------
    NVT_NH(
        struct=struct,
        temp=TEMPERATURE_K,
        timestep=TIMESTEP_FS,
        steps=NVT_STEPS,
        thermostat_time=THERMOSTAT_TIME_FS,
        stats_every=STATS_EVERY,
        traj_every=TRAJ_EVERY,
        seed=SEED,
        file_prefix=model_out / "nvt_equil",
    ).run()

    # -- NPT production, 200 ps (isotropic MTK chain; density + structure) ----
    # NPT_MTK uses the Martyna-Tobias-Klein chain integrator -- the same
    # formulation as LAMMPS `fix npt iso`. Chain length 3 is the LAMMPS
    # default for both the thermostat and barostat sub-chains.
    NPT_MTK(
        struct=struct,
        temp=TEMPERATURE_K,
        pressure=PRESSURE_GPA,
        timestep=TIMESTEP_FS,
        steps=NPT_STEPS,
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        thermostat_chain=3,
        barostat_chain=3,
        stats_every=STATS_EVERY,
        traj_every=TRAJ_EVERY,
        seed=SEED,
        file_prefix=model_out / "npt_prod",
    ).run()

    # -- Optional NVT continuation, 50 ps -------------------------------------
    if run_continuation:
        NVT_NH(
            struct=struct,
            temp=TEMPERATURE_K,
            timestep=TIMESTEP_FS,
            steps=NVT_CONT_STEPS,
            thermostat_time=THERMOSTAT_TIME_FS,
            stats_every=STATS_EVERY,
            traj_every=TRAJ_EVERY,
            seed=SEED,
            file_prefix=model_out / "nvt_continue",
        ).run()


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_md_reference(mlip: tuple[str, Any]) -> None:
    """
    Smoke test for the reference Janus MD protocol.

    Parameters
    ----------
    mlip
        ``(model_name, model)`` pair from the ml-peg registry.
    """
    model_name, model = mlip
    run_reference_md(model_name, model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the WiSE reference MD protocol for one registered model."
    )
    parser.add_argument("model", help=f"one of: {sorted(MODELS)}")
    parser.add_argument(
        "--continuation",
        action="store_true",
        help="append the optional 50 ps NVT continuation after NPT.",
    )
    args = parser.parse_args()

    if args.model not in MODELS:
        parser.error(f"unknown model '{args.model}'. Registered: {sorted(MODELS)}")

    run_reference_md(args.model, MODELS[args.model], run_continuation=args.continuation)
