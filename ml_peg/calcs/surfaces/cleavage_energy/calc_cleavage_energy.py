"""Run calculations for cleavage energy benchmark."""

from __future__ import annotations

from copy import copy
import json
from pathlib import Path
from typing import Any

from ase.io import read, write
from tqdm import tqdm
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_cleavage_energy(mlip: tuple[str, Any]) -> None:
    """
    Run cleavage energy benchmark calculations.

    For each surface configuration, single-point energies are computed for
    the slab and the lattice-matched bulk. Results are saved as a single
    JSON file per model containing only energies and system identifiers,
    avoiding redundant storage of atomic coordinates.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_dir = (
        download_s3_data(
            key="inputs/surfaces/cleavage_energy/cleavage_energy.zip",
            filename="cleavage_energy.zip",
        )
        / "cleavage_energy"
    )

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for mpid_dir in tqdm(sorted(d for d in data_dir.iterdir() if d.is_dir())):
        for xyz_file in sorted(mpid_dir.glob("*.xyz")):
            structs = read(xyz_file, index=":")
            slab, bulk = structs[0], structs[1]

            slab.calc = copy(calc)
            slab_energy = float(slab.get_potential_energy())

            bulk.calc = copy(calc)
            bulk_energy = float(bulk.get_potential_energy())

            unique_id = slab.info["unique_id"]
            results[unique_id] = {
                "slab_energy": slab_energy,
                "bulk_energy": bulk_energy,
                "area_slab": float(slab.info["area_slab"]),
                "thickness_ratio": float(slab.info["thickness_ratio"]),
                "ref_cleavage_energy": float(slab.info["ref_cleavage_energy"]),
                "mpid": slab.info["mpid"],
                "miller": slab.info["miller"],
                "term": int(slab.info["term"]),
            }

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUT_PATH / f"{model_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f)
