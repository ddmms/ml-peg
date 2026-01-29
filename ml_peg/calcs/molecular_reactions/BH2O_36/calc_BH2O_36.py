"""
Calculate the BH2O-36 benchmark for hydrolysis reaction barriers.

Journal of Chemical Theory and Computation 2023 19 (11), 3159-3171
DOI: 10.1021/acs.jctc.3c00176
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


def get_systems(info_path, xyz_dir):
    """
    Get names and atoms objects.

    Parameters
    ----------
    info_path
        Path to the system labels.
    xyz_dir
        Path to the directory containing structures.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping the system labels to dictionaries
        containing system xyz paths and charges.
    """
    replacements = {
        "amide11": "amide_1_1",
        "amide12": "amide_1_2",
        "amide21": "amide_2_1",
        "amide22": "amide_2_2",
        "basicepoxide1": "basic_epoxide_1",
        "basicepoxide2": "basic_epoxide_2",
    }
    systems = {}

    with open(info_path) as f:
        data = json.load(f)
        for key in data.keys():
            if "vacuum" not in key:
                continue
            items = key.strip().split("_")
            if items[0] not in systems.keys():
                systems[items[0]] = {}
            systems[items[0]][items[1]] = {}
            systems[items[0]][items[1]]["energy"] = data[key]

            xyz_prefix = items[0]
            if items[0] in replacements:
                xyz_prefix = replacements[items[0]]
            xyz_prefix += f"_{items[1]}_"
            xyz_path = list(xyz_dir.glob(xyz_prefix + "*"))[0]
            systems[items[0]][items[1]]["xyz_path"] = xyz_path
            systems[items[0]][items[1]]["charge"] = int(
                str(xyz_path).replace(".xyz", "").split("_")[-1]
            )

    return systems


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bh2o_36(mlip: tuple[str, Any]) -> None:
    """
    Run BH2O-36 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="BH2O-36.zip",
            key="inputs/molecular_reactions/BH2O-36/BH2O-36.zip",
        )
        / "BH2O-36"
    )
    # Read in data and attach calculator
    systems = get_systems(data_path / "mp2_super.json", data_path / "molecules/for_sp")

    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for identifier, system in tqdm(systems.items()):
        atoms_rct = read(system["rct"]["xyz_path"])
        atoms_rct.info["charge"] = int(system["rct"]["charge"])
        atoms_rct.info["spin"] = 1
        atoms_rct.info["ref_energy"] = system["rct"]["energy"]
        atoms_rct.calc = calc
        atoms_rct.info["pred_energy"] = atoms_rct.get_potential_energy()

        atoms_pro = read(system["pro"]["xyz_path"])
        atoms_pro.info["charge"] = int(system["pro"]["charge"])
        atoms_pro.info["spin"] = 1
        atoms_pro.info["ref_energy"] = system["pro"]["energy"]
        atoms_pro.calc = calc
        atoms_pro.info["pred_energy"] = atoms_pro.get_potential_energy()

        atoms_ts = read(system["ts"]["xyz_path"])
        atoms_ts.info["charge"] = int(system["ts"]["charge"])
        atoms_ts.info["spin"] = 1
        atoms_ts.info["ref_energy"] = system["ts"]["energy"]
        atoms_ts.calc = calc
        atoms_ts.info["pred_energy"] = atoms_ts.get_potential_energy()

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{identifier}_rct.xyz", atoms_rct)
        write(write_dir / f"{identifier}_pro.xyz", atoms_pro)
        write(write_dir / f"{identifier}_ts.xyz", atoms_ts)
