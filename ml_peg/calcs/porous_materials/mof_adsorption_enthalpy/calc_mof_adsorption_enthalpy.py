"""Run calculations for MOF adsorption enthalpy benchmark."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import copy
import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.build import molecule
from ase.data import vdw_radii
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.md import NVT
import numpy as np
import pytest
import torch

from ml_peg.calcs.porous_materials.mof_adsorption_enthalpy.widom_utils import (
    get_data_dir,
    load_reference,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

pytest.importorskip("flames", reason="Please install the `flames` dependency group")

from flames.adsorbate import Adsorbate  # noqa: E402
from flames.utilities import calculate_unit_cells  # noqa: E402
from flames.widom import Widom  # noqa: E402

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Relaxation of the empty framework
FMAX = 0.01
RELAX_STEPS = 1000

# Widom insertion
N_INSERTIONS = 20_000
CUTOFF = 6.0
SEED = 42

# NVT sampling of the empty framework for the flexible framework estimate
TIMESTEP = 1.0
EQUIL_STEPS = 5_000
N_SNAPSHOTS = 5
SNAPSHOT_EVERY = 1_000
N_INSERTIONS_SNAPSHOT = N_INSERTIONS // N_SNAPSHOTS


def set_neutral(struct: Atoms) -> None:
    """
    Set charge and spin multiplicity for a neutral singlet.

    Parameters
    ----------
    struct
        Structure to update.
    """
    struct.info["charge"] = 0
    struct.info["spin"] = 1


def relax_co2(calc: Any) -> Atoms:
    """
    Relax an isolated CO2 molecule.

    Parameters
    ----------
    calc
        Calculator to use.

    Returns
    -------
    Atoms
        Relaxed CO2.
    """
    co2 = molecule("CO2")
    co2.center(vacuum=10.0)
    set_neutral(co2)
    co2.calc = copy(calc)
    GeomOpt(struct=co2, fmax=FMAX, steps=RELAX_STEPS, filter_class=None).run()
    co2.calc = None
    co2.set_cell(None)
    co2.center()
    return co2


def run_widom(
    framework: Atoms,
    co2: Atoms,
    calc: Any,
    temperature: float,
    n_insertions: int,
    out_dir: Path,
) -> np.ndarray:
    """
    Run Widom insertion of CO2 and return the interaction energies.

    Parameters
    ----------
    framework
        Empty framework.
    co2
        Relaxed CO2 molecule.
    calc
        Calculator for the framework and guest.
    temperature
        Temperature in K.
    n_insertions
        Number of trial insertions.
    out_dir
        Directory for FLAMES output.

    Returns
    -------
    np.ndarray
        Interaction energy of each trial insertion, in eV.
    """
    co2 = co2.copy()
    set_neutral(co2)
    # Build the supercell here, as FLAMES does not keep atoms.info
    framework = framework.repeat(calculate_unit_cells(framework.cell, cutoff=CUTOFF))
    set_neutral(framework)
    widom = Widom(
        framework_atoms=framework,
        adsorbate_atoms=Adsorbate(name="CO2", structure=co2),
        temperature=temperature,
        model=calc,
        vdw_radii=vdw_radii,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_snapshots=False,
        output_to_file=False,
        output_folder=str(out_dir),
        random_seed=SEED,
        cutoff_radius=CUTOFF,
        automatic_supercell=False,
    )
    with open(out_dir / "widom.log", "w", encoding="utf8") as log, redirect_stdout(log):
        widom.run(n_insertions)
    # FLAMES initialises the energy list with a single zero
    return np.asarray(widom.int_energy_list[1:])


def run_framework(
    name: str, struct: Atoms, calc: Any, temperature: float, out_dir: Path
) -> None:
    """
    Relax a framework, then run Widom insertion on rigid and NVT structures.

    Parameters
    ----------
    name
        Framework name.
    struct
        Framework structure.
    calc
        Calculator to use.
    temperature
        Temperature in K.
    out_dir
        Directory to write outputs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    set_neutral(struct)
    struct.calc = copy(calc)

    opt = GeomOpt(struct=struct, fmax=FMAX, steps=RELAX_STEPS)
    opt.run()
    relaxed = opt.struct
    info = {
        "name": name,
        "n_atoms": len(relaxed),
        "volume": relaxed.get_volume(),
        "mass": float(relaxed.get_masses().sum()),
        "relax_steps": opt.dyn.nsteps,
        "relax_converged": opt.dyn.nsteps < RELAX_STEPS,
    }
    write(out_dir / "relaxed.extxyz", relaxed)

    co2 = relax_co2(calc)
    energies = run_widom(
        relaxed, co2, calc, temperature, N_INSERTIONS, out_dir / "flames_rigid"
    )
    np.save(out_dir / "widom_rigid.npy", energies)

    md_struct = relaxed.copy()
    set_neutral(md_struct)
    md_struct.calc = copy(calc)
    NVT(
        struct=md_struct,
        temp=temperature,
        steps=EQUIL_STEPS + N_SNAPSHOTS * SNAPSHOT_EVERY,
        timestep=TIMESTEP,
        traj_every=SNAPSHOT_EVERY,
        traj_start=EQUIL_STEPS + SNAPSHOT_EVERY,
        file_prefix=out_dir / "nvt",
        restart_every=EQUIL_STEPS + N_SNAPSHOTS * SNAPSHOT_EVERY + 1,
        seed=SEED,
    ).run()
    snapshots = read(out_dir / "nvt-traj.extxyz", index=":")

    md_energies = []
    for i, snapshot in enumerate(snapshots[:N_SNAPSHOTS]):
        set_neutral(snapshot)
        md_energies.append(
            run_widom(
                snapshot,
                co2,
                calc,
                temperature,
                N_INSERTIONS_SNAPSHOT,
                out_dir / f"flames_nvt_{i}",
            )
        )
    np.save(out_dir / "widom_nvt.npy", np.concatenate(md_energies))
    info["n_snapshots"] = len(md_energies)

    with open(out_dir / "info.json", "w", encoding="utf8") as file:
        json.dump(info, file, indent=2)


@pytest.mark.very_slow
@pytest.mark.framework("umof")
@pytest.mark.parametrize("mlip", MODELS.items())
def test_mof_adsorption_enthalpy(mlip: tuple[str, Any]) -> None:
    """
    Run Widom insertion of CO2 in each framework.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    calc = model.add_d3_calculator(calc)

    data_dir = get_data_dir()
    reference = load_reference(data_dir)

    for name, entry in reference["frameworks"].items():
        out_dir = OUT_PATH / model_name / entry["structure"].removesuffix(".cif")
        if (out_dir / "info.json").exists():
            continue
        struct = read(data_dir / "structures" / entry["structure"])
        try:
            run_framework(name, struct, calc, reference["temperature"], out_dir)
        except Exception as exc:
            warn(f"Error running {name} with {model_name}: {exc}", stacklevel=2)
