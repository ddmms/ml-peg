"""Run phonon dispersion and thermal calculations for diamond."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
from typing import Any
from warnings import warn

from ase.constraints import FixSymmetry
import ase.io
from ase.optimize import FIRE
import pytest

from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    get_fc2_and_freqs,
    init_phonopy_from_ref,
)
from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import compute_thermal_properties
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

OUT_PATH = Path(__file__).parent / "outputs"
DFT_REF_PATH = OUT_PATH / "DFT"

# The CASTEP/RSCAN reference, pre-converted to the shared phonon benchmark
# formats, plus the q-path metadata used to compute model band structures.
REF_FILES = (
    "diamond_band_structure.npz",
    "diamond_thermal.json",
    "diamond.xyz",
    "diamond_qpath_metadata.pkl",
)

# Relaxation settings, matching the general phonon benchmark.
FMAX = 0.005
RELAX_STEPS = 1000

THERMAL_MESH = [20, 20, 20]
DISPLACEMENT = 0.01
# 4×4×4 of the 8-atom conventional cell = 512-atom supercell.
SUPERCELL = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
PRIMITIVE_MATRIX = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]

MODELS = load_models(current_models)


@pytest.fixture(scope="session")
def diamond_data() -> Path:
    """
    Download diamond benchmark inputs and return the data directory.

    Returns
    -------
    Path
        Directory containing the pre-converted CASTEP/RSCAN reference data.
    """
    extracted = Path(
        download_github_data(filename="diamond_data/data.zip", github_uri=GITHUB_BASE)
    )
    return extracted / "data"


def test_diamond_phonons_ref(diamond_data: Path) -> None:
    """
    Copy the pre-converted DFT reference data to ``outputs/DFT/``.

    Parameters
    ----------
    diamond_data
        Directory containing the downloaded reference data.
    """
    DFT_REF_PATH.mkdir(parents=True, exist_ok=True)
    for name in REF_FILES:
        shutil.copy2(diamond_data / name, DFT_REF_PATH / name)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_diamond_phonons(mlip: tuple[str, Any], diamond_data: Path) -> None:
    """
    Compute phonon band structure and thermal properties for diamond with one MLIP.

    MLIP: 512-atom conventional supercell for both band and thermal calculations.
    DFT ref: 128-atom primitive supercell (CASTEP/RSCAN).

    Parameters
    ----------
    mlip
        Tuple of (model_name, model) as provided by pytest parametrize.
    diamond_data
        Directory containing the downloaded reference data.
    """
    model_name, model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    band_path = out_dir / "diamond_band_structure.npz"
    thermal_path = out_dir / "diamond_thermal.json"
    struct_path = out_dir / "diamond.xyz"

    if band_path.exists() and thermal_path.exists() and struct_path.exists():
        return

    calc = model.get_calculator(precision="high")
    with open(diamond_data / "diamond_qpath_metadata.pkl", "rb") as handle:
        qpath = pickle.load(handle)

    # Relax with fixed symmetry, as in the general phonon benchmark.
    atoms = ase.io.read(diamond_data / "diamond.xyz")
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))
    try:
        FIRE(atoms, logfile=None).run(fmax=FMAX, steps=RELAX_STEPS)
    except Exception as exc:
        warn(f"{model_name}: diamond relaxation failed: {exc}", stacklevel=2)
        return
    atoms.set_constraint()
    atoms.calc = None
    ase.io.write(struct_path, atoms)

    # The band structure and the Grüneisen/thermal step use the same
    # equilibrium force constants, so compute them once.
    try:
        phonons = init_phonopy_from_ref(
            atoms=atoms,
            fc2_supercell=SUPERCELL,
            primitive_matrix=PRIMITIVE_MATRIX,
            displacement_distance=DISPLACEMENT,
            is_plusminus=True,
        )
        phonons, _, _ = get_fc2_and_freqs(
            phonons=phonons,
            calculator=calc,
            symmetrize_fc2=True,
        )
    except Exception as exc:
        warn(f"{model_name}: diamond force constants failed: {exc}", stacklevel=2)
        return

    if not band_path.exists():
        try:
            phonons.run_band_structure(
                paths=qpath["qpoints"],
                labels=qpath["labels"],
                path_connections=qpath["connections"],
            )
            with open(band_path, "wb") as handle:
                pickle.dump(phonons.get_band_structure_dict(), handle)
        except Exception as exc:
            warn(f"{model_name}: diamond band structure failed: {exc}", stacklevel=2)

    if not thermal_path.exists():
        try:
            thermal = compute_thermal_properties(
                phonons=phonons,
                atoms=atoms,
                calculator=calc,
                q_mesh=THERMAL_MESH,
                symmetrize_fc2=True,
                temperature=300.0,
            )
            thermal_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "mean_gamma": thermal["mean_gamma"],
                        "debye_temperature_K": thermal["debye_temperature_K"],
                        "kappa_W_per_mK": thermal["kappa_W_per_mK"],
                        "temperature_K": 300.0,
                    },
                    indent=2,
                ),
                encoding="utf8",
            )
        except Exception as exc:
            warn(f"{model_name}: diamond thermal failed: {exc}", stacklevel=2)
