"""
Predict 13C NMR shielding tensors for the experimental tensor benchmark.

Structures and GIPAW-PBE reference tensors are the 13C structures of the ShiftML3
data record (Kellner et al., J. Phys. Chem. Lett. 16, 8714 (2025); Materials Cloud
Archive, DOI 10.24435/materialscloud:kt-kd, CC-BY-4.0), relaxed with PBE-D3(BJ) by
Ramos, Mueller & Beran, Faraday Discuss. 255, 119 (2025). Experimental principal
components for 20 of them are from Hartman et al., J. Chem. Phys. 143, 102809
(2015).

For each model, the ensemble-mean and per-member shielding tensors (3 x 3,
row-major, flattened) are written per atom, one extxyz file per structure.
Comparison with experiment happens in the analysis step.
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

SETS = ("C",)
OUT_PATH = Path(__file__).parent / "outputs"
MODELS = get_nmr_models(current_models)


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_nmr_tensor_expt(mlip: tuple[str, dict[str, Any]]) -> None:
    """
    Predict shielding tensors for all 13C benchmark structures with one model.

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
            filename="nmr_tensor_expt.zip",
            key="inputs/nmr_organic_materials/nmr_tensor_expt/nmr_tensor_expt.zip",
        )
        / "nmr_tensor_expt"
    )
    for set_name, struct_dir in structure_dirs(data_path / "structures", SETS):
        _, atoms = read_structure(struct_dir)
        try:
            # (atoms, 3, 3, members)
            members = np.asarray(
                predictor.get_cs_tensor_ensemble(atoms, return_symmetric=True),
                dtype=float,
            )
        except Exception as exc:
            warn(f"{model_name} failed on {struct_dir.name}: {exc}", stacklevel=2)
            members = np.full((len(atoms), 3, 3, 1), np.nan)

        atoms.arrays["sigma_tensor_members"] = members.reshape(len(atoms), -1)
        atoms.arrays["sigma_tensor_pred"] = members.mean(axis=-1).reshape(-1, 9)
        atoms.info["n_members"] = members.shape[-1]
        atoms.info["set"] = set_name
        atoms.info["structure"] = struct_dir.name
        write(write_dir / f"{set_name}__{struct_dir.name}.extxyz", atoms)
