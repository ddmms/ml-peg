"""
Run the MOF bulk modulus calculations.

Each framework is relaxed, then its cell is scaled isotropically over a range
of volumes with the internal coordinates re-relaxed at each fixed cell. The
resulting energy-volume curve is written out for the analysis step to fit with
a Birch-Murnaghan equation of state, which is the form the experimental
references were themselves fitted with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase.filters import FrechetCellFilter
import ase.io
from ase.optimize import LBFGS
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
    N_VOLUMES,
    VOLUME_RANGE,
    copy_reference_data,
    get_data_dir,
    load_bulk_modulus_reference,
    volume_scales,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Initial full relaxation, matching the phonon benchmark: LBFGS with the cell
# free and no symmetry constraint.
FMAX = 1e-8
RELAX_STEPS = 1000

# The volume scan grid itself lives in mof_phonon_utils so that it can be
# imported without the model registry; see VOLUME_RANGE / N_VOLUMES there.
# Internal relaxation at fixed cell. Looser than the initial relaxation: the
# equation-of-state fit needs a smooth curve, not machine-precision forces.
INNER_FMAX = 1e-6
INNER_STEPS = 500


def _structure_complete(mof: str, out_dir: Path) -> bool:
    """
    Check whether a framework's energy-volume curve already exists.

    Parameters
    ----------
    mof
        Framework name.
    out_dir
        Model output directory.

    Returns
    -------
    bool
        Whether the expected output file is present.
    """
    return (out_dir / f"{mof}_eos.json").exists()


def calc_mof_eos(mof: str, calc: Any, data_dir: Path, out_dir: Path) -> None:
    """
    Compute one framework's energy-volume curve and write it out.

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
    """
    atoms = ase.io.read(data_dir / "structures" / f"{mof}.cif")
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    atoms.calc = calc
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=FMAX, steps=RELAX_STEPS)

    reference_cell = atoms.cell.array.copy()
    reference_volume = float(atoms.get_volume())

    volumes, energies = [], []
    inner_steps, inner_fmax, inner_converged = [], [], []
    for scale in volume_scales():
        strained = atoms.copy()
        # Isotropic scaling: linear factor is the cube root of the volume factor.
        strained.set_cell(reference_cell * scale ** (1.0 / 3.0), scale_atoms=True)
        strained.calc = calc
        opt = LBFGS(strained, logfile=None)
        converged = bool(opt.run(fmax=INNER_FMAX, steps=INNER_STEPS))
        # Record how hard each fixed-cell relaxation was: a point that exhausts
        # INNER_STEPS without reaching INNER_FMAX still contributes to the fit,
        # so its residual force has to be visible rather than silent.
        residual = float(np.linalg.norm(strained.get_forces(), axis=1).max())
        inner_steps.append(int(opt.get_number_of_steps()))
        inner_fmax.append(residual)
        inner_converged.append(converged)
        volumes.append(float(strained.get_volume()))
        energies.append(float(strained.get_potential_energy()))

    if not all(inner_converged):
        unconverged = [
            f"{s:.2f} (fmax {f:.2e} eV/Ang)"
            for s, f, ok in zip(
                volume_scales(), inner_fmax, inner_converged, strict=True
            )
            if not ok
        ]
        print(
            f"{mof}: {len(unconverged)} of {N_VOLUMES} fixed-cell relaxations hit "
            f"the {INNER_STEPS}-step cap before reaching {INNER_FMAX:.0e} eV/Ang "
            f"at volume scales {', '.join(unconverged)}. The equation-of-state "
            "fit uses them regardless; treat its bulk modulus with caution."
        )

    record = {
        "mof": mof,
        "n_atoms": len(atoms),
        "relaxed_volume_ang3": reference_volume,
        "volumes_ang3": volumes,
        "energies_eV": energies,
        "volume_range": VOLUME_RANGE,
        "n_volumes": N_VOLUMES,
        "inner_fmax_target_eV_ang": INNER_FMAX,
        "inner_steps_limit": INNER_STEPS,
        "inner_steps_taken": inner_steps,
        "inner_fmax_reached_eV_ang": inner_fmax,
        "inner_converged": inner_converged,
    }
    with open(out_dir / f"{mof}_eos.json", "w", encoding="utf8") as handle:
        json.dump(record, handle, indent=4)


@pytest.fixture(scope="session")
def bulk_modulus_data() -> Path:
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


@pytest.mark.parametrize("mlip", MODELS.items())
def test_mof_bulk_modulus(mlip: tuple[str, Any], bulk_modulus_data: Path) -> None:
    """
    Run the MOF bulk modulus calculations for one model.

    Only frameworks carrying a usable experimental reference are run, since
    the energy-volume scan is expensive and unscored frameworks would add
    nothing.

    Parameters
    ----------
    mlip
        Name of model to use and model to get calculator.
    bulk_modulus_data
        Directory containing the benchmark structures and reference data.
    """
    model_name, model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = load_bulk_modulus_reference()
    frameworks = sorted(
        name for name, entry in reference.items() if not entry.get("excluded")
    )
    pending = [mof for mof in frameworks if not _structure_complete(mof, out_dir)]
    if not pending:
        return

    calc = model.get_calculator(precision="high")

    for mof in tqdm(pending, desc=f"{model_name} MOF bulk modulus", unit="mof"):
        try:
            calc_mof_eos(mof, calc, bulk_modulus_data, out_dir)
        except Exception as exc:
            warn(f"{model_name}: MOF {mof} failed: {exc}", stacklevel=2)
