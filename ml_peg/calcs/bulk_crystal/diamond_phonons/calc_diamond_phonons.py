"""Run phonon dispersion calculations for diamond."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
from phonopy import load as load_phonopy
import pytest

from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    get_fc2_and_freqs,
    init_phonopy_from_ref,
    phonopy_to_ase_atoms,
)
from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import compute_thermal_properties
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names, load_models

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

EXTRACTED_ROOT = Path(
    download_github_data(
        filename="diamond_data/data.zip",
        github_uri=GITHUB_BASE,
    )
)

DATA_PATH = EXTRACTED_ROOT / "data"
DIAMOND_YAML = DATA_PATH / "diamond.yaml"
DFT_BAND = DATA_PATH / "dft_band.npz"
OUT_PATH = Path(__file__).parent / "outputs"

Q_MESH = 6
THERMAL_MESH = [20, 20, 20]
# 4×4×4 of the 8-atom conventional cell = 512-atom supercell.
GRUNEISEN_SUPERCELL = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]

MODEL_NAMES = get_model_names(current_models)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_diamond_phonons(model_name: str) -> None:
    """
    Compute phonon band structure and thermal properties for diamond using one MLIP.

    MLIP: 512-atom conventional supercell for both band and thermal calculations.
    DFT ref: 128-atom primitive supercell (CASTEP/RSCAN).

    Parameters
    ----------
    model_name
        Name of the ML-PEG model to evaluate.
    """
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    band_path = out_dir / "diamond_band_structure.npz"
    thermal_path = out_dir / "diamond_thermal.json"

    need_band = not band_path.exists()
    need_thermal = not thermal_path.exists()

    if not need_band and not need_thermal:
        return

    model = load_models(model_name)[model_name]
    calc = model.get_calculator()
    phonons_ref = load_phonopy(str(DIAMOND_YAML))

    if need_band:
        atoms = phonopy_to_ase_atoms(phonons_ref)

        if "primitive_matrix" in atoms.info:
            primitive_matrix = atoms.info["primitive_matrix"]
        else:
            unitcell = phonons_ref.unitcell
            primitive_matrix = np.linalg.inv(np.array(unitcell.cell)) @ np.array(
                phonons_ref.primitive.cell
            )

        phonons = init_phonopy_from_ref(
            atoms=atoms,
            displacement_dataset=phonons_ref.dataset,
            primitive_matrix=primitive_matrix,
            symprec=1e-5,
        )
        phonons, _, _ = get_fc2_and_freqs(
            phonons=phonons,
            calculator=calc,
            q_mesh=np.array([Q_MESH] * 3),
            symmetrize_fc2=True,
        )

        dft_qpoints = np.load(DFT_BAND)["qpoints"]
        phonons.run_band_structure(paths=[dft_qpoints])
        band_structure = phonons.get_band_structure_dict()
        with open(band_path, "wb") as f:
            pickle.dump(band_structure, f)

    if need_thermal:
        atoms_conv = phonopy_to_ase_atoms(phonons_ref)
        phonons_grun = init_phonopy_from_ref(
            atoms=atoms_conv,
            fc2_supercell=GRUNEISEN_SUPERCELL,
            displacement_distance=0.01,
        )
        phonons_grun, _, _ = get_fc2_and_freqs(
            phonons=phonons_grun,
            calculator=calc,
            symmetrize_fc2=True,
        )
        thermal = compute_thermal_properties(
            phonons=phonons_grun,
            atoms=atoms_conv,
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
