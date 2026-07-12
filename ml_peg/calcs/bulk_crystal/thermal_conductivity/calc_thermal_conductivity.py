"""
Run calculations for thermal conductivity tests.

Code is adapted from https://github.com/MPA2suite/k_SRME/blob/6ff4c867/k_srme/conductivity.py
by Balázs Póta, Paramvir Ahlawat, Gábor Csányi, Michele Simoncelli
and https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/phonons/thermal_conductivity.py
by Janosh Riebesell.
See https://arxiv.org/abs/2408.00755 for details.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import traceback
from typing import Any, Literal
import warnings

import ase
from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter, Filter, FrechetCellFilter
from ase.io import read, write
from ase.optimize.optimize import Optimizer
import h5py
import pandas as pd
import pytest
from tqdm import tqdm

from ml_peg.calcs.bulk_crystal.thermal_conductivity import thermal_conductivity as tc
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

if len(sys.argv) >= 3:
    try:
        PARALLEL_TASK_ID = int(sys.argv[1])
        PARALLEL_TASK_NUM = int(sys.argv[2])
    except ValueError:
        print(
            "Could not parse parallel task arguments, running in serial.",
            file=sys.stderr,
        )
        PARALLEL_TASK_ID = 0
        PARALLEL_TASK_NUM = 1
else:
    PARALLEL_TASK_ID = 0
    PARALLEL_TASK_NUM = 1


ase_optimizer: Literal[
    "GPMin",
    "GOQN",
    "BFGSLineSearch",
    "QuasiNewton",
    "SciPyFminBFGS",
    "BFGS",
    "LBFGSLineSearch",
    "SciPyFminCG",
    "FIRE2",
    "FIRE",
    "LBFGS",
] = "LBFGS"
ase_filter: Literal["frechet", "exp"] = "frechet"
max_steps = 300
fmax = 1e-4  # Run until the forces are smaller than this in eV/A
enforce_relax_symm = True  # Enforce symmetry during relaxation

# Symmetry parameters
symprec = 1e-5  # symmetry precision for enforcing relaxation and conductivity calcs

# Conductivity to be calculated if symmetry group changed during relaxation
conductivity_broken_symm = False
save_forces = True  # Save force sets to file
temperatures: list[float] = [300]
displacement_distance = 0.03
ignore_imaginary_freqs = True

default_dtype = "float64"

STRUCTURE_PATH = DATA_PATH / "phononDB-PBE-structures.extxyz"

FAST_ONLY = True


@pytest.mark.parametrize("mlip", MODELS.items())
def test_thermal_conductivity(mlip: tuple[str, Any]) -> None:
    """
    Run thermal conductivity test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    model.default_dtype = default_dtype
    calculator = model.get_calculator()

    # Download dataset
    # thermal_conductivity_dir = (
    #     download_s3_data(
    #         key="inputs/bulk_crystal/thermal_conductivity/thermal_conductivity.zip",
    #         filename="thermal_conductivity.zip",
    #     )
    #     / "thermal_conductivity"
    # )

    atoms_list = read(STRUCTURE_PATH, format="extxyz", index=":")
    atoms_list = atoms_list[PARALLEL_TASK_ID - 0 :: PARALLEL_TASK_NUM]

    kappa_dicts = {}
    fast_kappa_dicts = {}

    for i, atoms_input in enumerate(tqdm(atoms_list, desc="Atoms loop")):
        structure_id = atoms_input.info.get(tc.TCKeys.mat_id, f"structure_{i}")

        out_dir = OUT_PATH / model_name / structure_id
        out_dir.mkdir(parents=True, exist_ok=True)

        results_dict, fast_results_dict = calc_thermal_conductivity_per_structure(
            atoms_input, calculator, out_dir
        )

        results_dict[tc.TCKeys.mat_id] = structure_id
        fast_results_dict[tc.TCKeys.mat_id] = structure_id

        if not FAST_ONLY:
            results_path = out_dir / "kappa.hdf5"
            with h5py.File(results_path, "w") as f:
                tc.dict_to_hdf5(results_dict, f)
            kappa_dicts[structure_id] = results_dict

        fast_results_path = out_dir / "fast_kappa.hdf5"
        with h5py.File(fast_results_path, "w") as f:
            tc.dict_to_hdf5(fast_results_dict, f)
        fast_kappa_dicts[structure_id] = fast_results_dict

    if PARALLEL_TASK_NUM == 1:
        if not FAST_ONLY:
            df = pd.DataFrame(kappa_dicts).T
            df.index.name = tc.TCKeys.mat_id
            df.reset_index().to_json(OUT_PATH / model_name / "kappas.json.gz")
            with h5py.File(OUT_PATH / model_name / "kappas.hdf5", "w") as f:
                tc.dict_to_hdf5(kappa_dicts, f)

        fast_df = pd.DataFrame(fast_kappa_dicts).T
        fast_df.index.name = tc.TCKeys.mat_id
        fast_df.reset_index().to_json(OUT_PATH / model_name / "fast_kappas.json.gz")
        with h5py.File(OUT_PATH / model_name / "fast_kappas.hdf5", "w") as f:
            tc.dict_to_hdf5(fast_kappa_dicts, f)


def calc_thermal_conductivity_per_structure(
    atoms_input: ase.Atoms, calculator: ase.Calculator, out_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Calculate thermal conductivity results for a single structure.

    Parameters
    ----------
    atoms_input : ase.Atoms
        Input atomic structure to evaluate.
    calculator : ase.Calculator
        Calculator used for the structure relaxation and property evaluation.
    out_dir : Path
        Directory where intermediate and final results are written.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple containing the main thermal conductivity result dictionary and
        the fast calculation result dictionary.
    """
    atoms = atoms_input.copy()

    atoms.calc = calculator

    # Relax structure before calculating thermal conductivity
    relax_path = out_dir / "relaxed.extxyz"

    formula = atoms.info.get("name", atoms.get_chemical_formula())

    mat_id = atoms.info[tc.TCKeys.mat_id]
    init_info = deepcopy(atoms.info)
    info_dict: dict[str, Any] = {
        str(tc.TCKeys.mat_id): mat_id,
        str(tc.TCKeys.formula): formula,
    }
    err_dict: dict[str, list[str]] = {"errors": [], "error_traceback": []}

    # Select filter class
    if ase_filter in {"frechet", "exp"}:
        filter_cls: type[Filter] = {
            "frechet": FrechetCellFilter,
            "exp": ExpCellFilter,
        }[ase_filter]
    else:
        # Default to FrechetCellFilter if not specified (for MACE compatibility)
        filter_cls = FrechetCellFilter

    # Select optimizer class
    optimizer_dict = {
        "GPMin": ase.optimize.GPMin,
        "GOQN": ase.optimize.GoodOldQuasiNewton,
        "BFGSLineSearch": ase.optimize.BFGSLineSearch,
        "QuasiNewton": ase.optimize.BFGSLineSearch,
        "SciPyFminBFGS": ase.optimize.sciopt.SciPyFminBFGS,
        "BFGS": ase.optimize.BFGS,
        "LBFGSLineSearch": ase.optimize.LBFGSLineSearch,
        "SciPyFminCG": ase.optimize.sciopt.SciPyFminCG,
        "FIRE2": ase.optimize.FIRE2,
        "FIRE": ase.optimize.FIRE,
        "LBFGS": ase.optimize.LBFGS,
    }
    optim_cls: type[Optimizer] = optimizer_dict[ase_optimizer]

    # Initialize variables that might be needed in error handling
    relax_dict: dict[str, Any] = {
        "max_stress": None,
        "reached_max_steps": False,
        "broken_symmetry": False,
    }
    # initial space group for symmetry breaking detection
    init_spg_num = tc.get_spacegroup_number_from_atoms(atoms, symprec=symprec)

    # Relaxation
    try:
        results_dict = {}
        atoms.calc = calculator
        if max_steps > 0:
            if enforce_relax_symm:
                atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
                filtered_atoms = filter_cls(atoms, mask=[True] * 3 + [False] * 3)
            else:
                filtered_atoms = filter_cls(atoms)

            optimizer = optim_cls(filtered_atoms, logfile=out_dir / "relax.log")
            optimizer.run(fmax=fmax, steps=max_steps)

            step_count = getattr(optimizer, "nsteps", None)  # Get optimizer step count
            if step_count is None:  # fallback to extract from state_dict if available
                state = getattr(optimizer, "state_dict", dict)()
                step_count = state.get("step", 0)

            reached_max_steps = step_count >= max_steps
            if reached_max_steps:
                print(f"Material {mat_id=} reached {max_steps=} during relaxation")

            # maximum residual stress component in for xx,yy,zz and xy,yz,xz
            # components separately result is a array of 2 elements
            max_stress = abs(atoms.get_stress()).reshape((2, 3), order="C").max(axis=1)

            atoms.calc = None
            atoms.constraints = None
            atoms.info = init_info | atoms.info

            # Check if symmetry was broken during relaxation
            relaxed_spg_num = tc.get_spacegroup_number_from_atoms(
                atoms, symprec=symprec
            )
            broken_symmetry = init_spg_num != relaxed_spg_num

            relax_dict = {
                "max_stress": max_stress,
                "reached_max_steps": reached_max_steps,
                "broken_symmetry": broken_symmetry,
                tc.TCKeys.final_spg_num: relaxed_spg_num,
                tc.TCKeys.init_spg_num: init_spg_num,
            }

    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        warnings.warn(f"Failed to relax {formula=}, {mat_id=}: {exc!r}", stacklevel=2)
        traceback.print_exc()
        ph3 = None
        err_dict["errors"] += [f"RelaxError: {exc!r}"]
        err_dict["error_traceback"] += [traceback.format_exc()]
        results_dict = info_dict | relax_dict | err_dict
        return results_dict, results_dict

    write(relax_path, atoms, format="extxyz")

    try:
        ph3 = tc.init_phono3py(
            atoms,
            fc2_supercell=atoms.info["fc2_supercell"],
            fc3_supercell=atoms.info["fc3_supercell"],
            q_point_mesh=atoms.info["q_point_mesh"],
            displacement_distance=displacement_distance,
            symprec=symprec,
        )

        ph3, fc2_set, freqs = tc.get_fc2_and_freqs(
            ph3, calculator=calculator, pbar_kwargs={"disable": True}
        )

        has_imag_ph_modes = tc.check_imaginary_freqs(freqs)
        freqs_dict = {
            tc.TCKeys.has_imag_ph_modes: has_imag_ph_modes,
            tc.TCKeys.ph_freqs: freqs,
        }

        # Determine if conductivity calculation should proceed
        if ignore_imaginary_freqs:
            # NequIP/Allegro mode: ignore imaginary frequencies
            ltc_condition = True
        else:
            # MACE mode: check both imaginary freqs and broken symmetry
            broken_symmetry = relax_dict.get("broken_symmetry", False)
            ltc_condition = not has_imag_ph_modes and (
                not broken_symmetry or conductivity_broken_symm
            )

        if ltc_condition:
            tc.calculate_fc3_set(
                ph3, calculator=calculator, pbar_kwargs={"leave": False}
            )
            ph3.produce_fc3(symmetrize_fc3r=True)
        else:
            reason = []
            if has_imag_ph_modes:
                reason.append("imaginary frequencies")
            if relax_dict.get("broken_symmetry") and not conductivity_broken_symm:
                reason.append("broken symmetry")
            warnings.warn(
                f"{' and '.join(reason).capitalize()} detected for {mat_id}, "
                f"skipping FC3 and LTC calculation!",
                stacklevel=2,
            )

        if not ltc_condition:
            results_dict = info_dict | relax_dict | freqs_dict | err_dict
            return results_dict, results_dict

    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        warnings.warn(f"Failed to calculate force sets {mat_id}: {exc!r}", stacklevel=2)
        traceback.print_exc()
        err_dict["errors"] += [f"ForceConstantError: {exc!r}"]
        err_dict["error_traceback"] += [traceback.format_exc()]
        results_dict = info_dict | relax_dict | err_dict
        return results_dict, results_dict

    # Calculation of conductivity
    try:
        with tc.tqdm_gridpoints(desc="Conducitivity calc"):
            ph3.mesh_numbers = atoms.info["fast_q_point_mesh"]
            ph3, fast_kappa_dict, _cond = tc.calculate_conductivity(
                ph3, temperatures=temperatures, log_level=2
            )
        fast_results_dict = (
            info_dict | relax_dict | freqs_dict | fast_kappa_dict | err_dict
        )

        if not FAST_ONLY:
            with tc.tqdm_gridpoints(desc="Conducitivity calc"):
                ph3.mesh_numbers = atoms.info["q_point_mesh"]
                ph3, kappa_dict, _cond = tc.calculate_conductivity(
                    ph3, temperatures=temperatures, log_level=2
                )
            results_dict = info_dict | relax_dict | freqs_dict | kappa_dict | err_dict

    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        warnings.warn(
            f"Failed to calculate conductivity {mat_id}: {exc!r}", stacklevel=2
        )
        traceback.print_exc()
        err_dict["errors"] += [f"ConductivityError: {exc!r}"]
        err_dict["error_traceback"] += [traceback.format_exc()]
        results_dict = info_dict | relax_dict | freqs_dict | err_dict
        return results_dict, results_dict

    return results_dict, fast_results_dict
