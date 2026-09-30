"""
Run the shared MOF phonon calculations.

One finite-displacement phonon calculation per framework per model produces
everything the two scored quantities need: the heat capacity benchmark reads
``<mof>_phonon_summary.json``, the INS benchmark reads ``<mof>_ins.json``, and
both read the imaginary-mode fraction recorded in the summary. Running this
module once is therefore enough for both analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from warnings import warn

from ase.filters import FrechetCellFilter
import ase.io
from ase.optimize import LBFGS
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    get_fc2_and_freqs,
    init_phonopy_from_ref,
)
from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
    INS_BACKENDS,
    OUTPUT_PATH,
    THERMAL_CUTOFF_FREQUENCY,
    copy_reference_data,
    diagonal_supercell_matrix,
    get_data_dir,
    get_structure_names,
    heat_capacity_per_gram,
    imaginary_mode_percentage,
    resolve_ins_backend,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = OUTPUT_PATH

# Relaxation. A very tight force convergence is used deliberately: residual
# forces in the starting structure show up as spurious imaginary modes in the
# finite-difference phonons, which would otherwise contaminate the
# imaginary-mode metric. Whichever of FMAX or RELAX_STEPS is reached first
# ends the relaxation, so RELAX_STEPS is the effective limit in practice.
# Cell and coordinates are relaxed together with no symmetry constraint, so
# the imaginary-mode count measures unconstrained dynamical stability.
# phonopy then determines the space group from the relaxed structure, which
# means the supercell and displacement set may differ between models for the
# same framework.
FMAX = 1e-8
RELAX_STEPS = 1000

# Finite displacements.
DISPLACEMENT = 0.01
# Minimum supercell lattice vector. MOF unit cells are already large, so most
# frameworks need no expansion at all to reach this.
SUPERCELL_MIN_LENGTH = 20.0

# Sampling. The mesh is shared by the thermal properties, the imaginary-mode
# count and the INS spectrum, so all three are converged on the same grid.
Q_MESH = (11, 11, 11)
REFERENCE_TEMPERATURE = 300.0
T_MIN, T_MAX, T_STEP = 0.0, 600.0, 10.0

# INS energy grid. The reference spectra are digitised over a range of
# windows; this grid spans all of them and is resampled per comparison.
INS_ENERGY_MIN_MEV = 0.0
INS_ENERGY_MAX_MEV = 520.0
INS_ENERGY_STEP_MEV = 0.5
INS_TEMPERATURE = 10.0
INS_BACKEND = "auto"


@pytest.fixture(scope="session")
def mof_data() -> Path:
    """
    Download the benchmark data and stage the reference files.

    Returns
    -------
    Path
        Directory containing the structures and reference data.
    """
    data_dir = get_data_dir()
    copy_reference_data(data_dir)
    return data_dir


def ins_energy_bins() -> np.ndarray:
    """
    Get the INS energy bin edges.

    Returns
    -------
    np.ndarray
        Bin edges in meV.
    """
    return np.arange(
        INS_ENERGY_MIN_MEV,
        INS_ENERGY_MAX_MEV + INS_ENERGY_STEP_MEV,
        INS_ENERGY_STEP_MEV,
    )


def _structure_complete(mof: str, out_dir: Path) -> bool:
    """
    Check whether a framework's outputs already exist.

    Parameters
    ----------
    mof
        Framework name.
    out_dir
        Model output directory.

    Returns
    -------
    bool
        Whether every expected output file is present.
    """
    return all(
        (out_dir / f"{mof}{suffix}").exists()
        for suffix in ("_phonon_summary.json", "_ins.json", ".xyz")
    )


def _relax(mof: str, calc: Any, data_dir: Path, out_dir: Path) -> Any:
    """
    Relax a framework tightly and write the relaxed structure.

    Unconstrained full cell relaxation with LBFGS, to ``FMAX`` or
    ``RELAX_STEPS``, whichever is reached first. No symmetry constraint is
    applied, so a framework is free to relax away from its input space group
    if the potential drives it there.

    Parameters
    ----------
    mof
        Framework name.
    calc
        ASE calculator for the model under test.
    data_dir
        Directory containing the benchmark structures.
    out_dir
        Model output directory.

    Returns
    -------
    Any
        Relaxed ASE atoms.
    """
    atoms = ase.io.read(data_dir / "structures" / f"{mof}.cif")
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    atoms.calc = calc
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=FMAX, steps=RELAX_STEPS)

    relaxed = atoms.copy()
    relaxed.calc = None
    ase.io.write(out_dir / f"{mof}.xyz", relaxed)
    return atoms


def calc_mof(mof: str, calc: Any, data_dir: Path, out_dir: Path, backend: str) -> None:
    """
    Run the full phonon workflow for one framework and write its outputs.

    Parameters
    ----------
    mof
        Framework name.
    calc
        ASE calculator for the model under test.
    data_dir
        Directory containing the benchmark structures.
    out_dir
        Model output directory.
    backend
        Name of the INS backend to use.
    """
    atoms = _relax(mof, calc, data_dir, out_dir)

    supercell = diagonal_supercell_matrix(atoms, min_length=SUPERCELL_MIN_LENGTH)
    phonons = init_phonopy_from_ref(
        atoms=atoms,
        fc2_supercell=supercell,
        primitive_matrix="auto",
        displacement_distance=DISPLACEMENT,
        is_plusminus=True,
    )
    phonons, _, _ = get_fc2_and_freqs(phonons, calc, symmetrize_fc2=True)

    # One mesh serves both quantities. The imaginary-mode percentage is read
    # back out of the thermal-property integration, so it must run after it.
    phonons.run_mesh(list(Q_MESH))
    phonons.run_thermal_properties(
        t_min=T_MIN,
        t_max=T_MAX,
        t_step=T_STEP,
        cutoff_frequency=THERMAL_CUTOFF_FREQUENCY,
    )
    heat_capacity = heat_capacity_per_gram(phonons, REFERENCE_TEMPERATURE)
    percent_imaginary = imaginary_mode_percentage(phonons)

    summary = {
        "mof": mof,
        "n_atoms": len(atoms),
        "n_atoms_primitive": len(phonons.primitive),
        "supercell_matrix": supercell,
        "heat_capacity_J_per_g_K": heat_capacity,
        "heat_capacity_temperature_K": REFERENCE_TEMPERATURE,
        "percent_imaginary_modes": percent_imaginary,
        "q_mesh": list(Q_MESH),
    }
    with open(out_dir / f"{mof}_phonon_summary.json", "w", encoding="utf8") as handle:
        json.dump(summary, handle, indent=4)

    bins = ins_energy_bins()
    # The force-constant array for a MOF supercell is hundreds of MB, and the
    # INS backends only need it transiently, so it is written to scratch.
    with TemporaryDirectory(prefix=f"{mof}_phonopy_") as scratch:
        intensity = INS_BACKENDS[backend](
            phonons,
            Path(scratch),
            bins,
            INS_TEMPERATURE,
            Q_MESH,
        )
    spectrum = {
        "mof": mof,
        "backend": backend,
        "temperature_K": INS_TEMPERATURE,
        "energy_meV": (0.5 * (bins[:-1] + bins[1:])).tolist(),
        "intensity": np.asarray(intensity, dtype=float).tolist(),
    }
    with open(out_dir / f"{mof}_ins.json", "w", encoding="utf8") as handle:
        json.dump(spectrum, handle)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_mof_phonons(mlip: tuple[str, Any], mof_data: Path) -> None:
    """
    Run the shared MOF phonon calculations for one model.

    Parameters
    ----------
    mlip
        Name of model to use and model to get calculator.
    mof_data
        Directory containing the benchmark structures and reference data.
    """
    model_name, model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        mof
        for mof in get_structure_names(mof_data)
        if not _structure_complete(mof, out_dir)
    ]
    if not pending:
        return

    backend = resolve_ins_backend(INS_BACKEND)
    calc = model.get_calculator(precision="high")

    for mof in tqdm(pending, desc=f"{model_name} MOF phonons", unit="mof"):
        try:
            calc_mof(mof, calc, mof_data, out_dir, backend)
        except Exception as exc:
            warn(f"{model_name}: MOF {mof} failed: {exc}", stacklevel=2)
