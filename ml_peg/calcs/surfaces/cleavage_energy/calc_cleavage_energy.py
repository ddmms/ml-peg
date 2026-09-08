"""Run calculations for cleavage energy benchmark."""

from __future__ import annotations

from copy import copy
import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


def _set_neutral_singlet_info(struct: Atoms) -> None:
    """
    Set default charge and spin multiplicity for neutral surface structures.

    Parameters
    ----------
    struct
        ASE Atoms object to annotate.
    """
    struct.info.setdefault("charge", 0)
    struct.info.setdefault("spin", 1)


def _miller_indices(value: Any) -> list[int]:
    """
    Convert Miller-index metadata to a JSON-safe list of integers.

    Parameters
    ----------
    value
        Miller-index metadata from an ASE Atoms info dictionary.

    Returns
    -------
    list[int]
        Miller indices.
    """
    if isinstance(value, str):
        value = value.strip("()[]").replace(",", " ")
        parts = value.split()
        if len(parts) > 1:
            return [int(part.replace("m", "-")) for part in parts]

        indices = []
        token = parts[0]
        i = 0
        while i < len(token):
            sign = 1
            if token[i] in {"m", "-"}:
                sign = -1
                i += 1
            elif token[i] == "+":
                i += 1

            if i >= len(token) or not token[i].isdigit():
                msg = f"Could not parse Miller indices from {value!r}"
                raise ValueError(msg)

            indices.append(sign * int(token[i]))
            i += 1
        return indices
    return [int(index) for index in value]


@pytest.mark.parametrize("mlip", MODELS.items())
def test_cleavage_energy(mlip: tuple[str, Any]) -> None:
    """
    Run cleavage energy benchmark calculations.

    For each surface configuration, single-point energies are computed for the slab
    and the lattice-matched bulk. Structures are retained as extxyz files, while
    scalar results are also saved as a compact JSON file per model for analysis.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_dir = (
        download_s3_data(
            key="inputs/surfaces/cleavage_energy/cleavage_energy.zip",
            filename="cleavage_energy.zip",
        )
        / "cleavage_energy"
    )

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    results = []
    idx = 0
    for mpid_dir in tqdm(sorted(d for d in data_dir.iterdir() if d.is_dir())):
        for xyz_file in sorted(mpid_dir.glob("*.xyz")):
            structs = read(xyz_file, index=":")
            slab, bulk = structs[0], structs[1]
            _set_neutral_singlet_info(slab)
            _set_neutral_singlet_info(bulk)

            slab.calc = copy(calc)
            try:
                slab_energy = float(slab.get_potential_energy())
            except Exception as exc:
                warn(
                    f"Error calculating slab energy for {xyz_file}: {exc}",
                    stacklevel=2,
                )
                slab_energy = np.nan

            bulk.calc = copy(calc)
            try:
                bulk_energy = float(bulk.get_potential_energy())
            except Exception as exc:
                warn(
                    f"Error calculating bulk energy for {xyz_file}: {exc}",
                    stacklevel=2,
                )
                bulk_energy = np.nan

            slab.info.update(
                {
                    "slab_energy": slab_energy,
                    "bulk_energy": bulk_energy,
                    "area_slab": float(slab.info["area_slab"]),
                    "thickness_ratio": float(slab.info["thickness_ratio"]),
                    "ref_cleavage_energy": float(slab.info["ref_cleavage_energy"]),
                    "mpid": slab.info["mpid"],
                    "miller": _miller_indices(slab.info["miller"]),
                    "term": int(slab.info["term"]),
                }
            )
            write(write_dir / f"{idx}.xyz", slab, format="extxyz")
            results.append(
                {
                    "id": idx,
                    "structure_file": f"{idx}.xyz",
                    "mpid": slab.info["mpid"],
                    "miller": slab.info["miller"],
                    "term": int(slab.info["term"]),
                    "slab_energy": slab_energy,
                    "bulk_energy": bulk_energy,
                    "area_slab": float(slab.info["area_slab"]),
                    "thickness_ratio": float(slab.info["thickness_ratio"]),
                    "ref_cleavage_energy": float(slab.info["ref_cleavage_energy"]),
                }
            )
            idx += 1

    (write_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf8"
    )
