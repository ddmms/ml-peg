"""
Predict isotropic NMR shieldings for the experimental shift benchmark.

Structures and GIPAW-PBE reference shieldings are the 49 crystals of the ShiftML3
data record (Kellner et al., J. Phys. Chem. Lett. 16, 8714 (2025); Materials Cloud
Archive, DOI 10.24435/materialscloud:kt-kd, CC-BY-4.0): 13 structures for 1H and the
36-structure 13C/15N set of Ramos, Mueller & Beran, Faraday Discuss. 255, 119
(2025).

For each model, the ensemble-mean and per-member isotropic shieldings are written
per atom, one extxyz file per structure. Comparison with experiment happens in the
analysis step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import write
import numpy as np
import pytest

from ml_peg.calcs.utils.nmr import read_structure, structure_dirs
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_nmr_models, load_nmr_predictor

SETS = ("H", "C", "N")  # structure folders, named by the nucleus they are scored on
OUT_PATH = Path(__file__).parent / "outputs"
MODELS = get_nmr_models(current_models)


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_nmr_isotropic_expt(mlip: tuple[str, dict[str, Any]]) -> None:
    """
    Predict isotropic shieldings for all benchmark structures with one model.

    Parameters
    ----------
    mlip
        Name of the NMR model and its entry in ``nmr_models.yml``.
    """
    model_name, config = mlip
    predictor = load_nmr_predictor(config, precision="high")
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Download data
    data_path = (
        download_s3_data(
            filename="nmr_isotropic_expt.zip",
            key="inputs/nmr_organic_materials/nmr_isotropic_expt/nmr_isotropic_expt.zip",
        )
        / "nmr_isotropic_expt"
    )
    for set_name, struct_dir in structure_dirs(data_path / "structures", SETS):
        _, atoms = read_structure(struct_dir)
        try:
            members = np.asarray(predictor.get_cs_iso_ensemble(atoms), dtype=float)
        except Exception as exc:
            warn(f"{model_name} failed on {struct_dir.name}: {exc}", stacklevel=2)
            members = np.full((len(atoms), 1), np.nan)

        atoms.arrays["sigma_iso_members"] = members
        atoms.arrays["sigma_iso_pred"] = members.mean(axis=1)
        atoms.info["set"] = set_name
        atoms.info["structure"] = struct_dir.name
        write(write_dir / f"{set_name}__{struct_dir.name}.extxyz", atoms)
