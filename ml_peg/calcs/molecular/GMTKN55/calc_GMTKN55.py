"""Run GMTKN55 single point calculations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write
import numpy as np
from pandas import read_csv
import pytest
from tqdm import tqdm
import yaml

from ml_peg.calcs.utils.utils import get_benchmark_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Raw download URL (for direct downloading)
BENCHMARK_DATA_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/joehart2001/mlipx/main/benchmark_data/"
)


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_gmtkn55(mlip: tuple[str, Any]) -> None:
    """
    Run single point calculations for GMTKN55 dataset.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    print(f"\nEvaluating with model: {model_name}")
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # Download GMTKN55.yaml and subsets.csv
    data_dir = get_benchmark_data("GMTKN55.zip") / "GMTKN55"

    with open(data_dir / "GMTKN55.yaml") as file:
        structure_dict = yaml.safe_load(file)

    with open(data_dir / "subsets.csv") as subsets_file:
        subsets_info = read_csv(subsets_file, delimiter=",")

    for subset_name, subset in tqdm(structure_dict.items()):
        subset_name = subset_name.lower()

        # Get category and weight from csv file
        subsets_info["subset"] = subsets_info["subset"].str.lower()
        row = subsets_info.loc[subsets_info["subset"] == subset_name]
        category = row["category"].values[0]
        excluded = row["excluded"].values[0]

        for system_name, system in subset.items():
            # system 1,2,3...
            system_structs = []
            ref_value = system["Energy"]
            weight = system["Weight"]

            for species_name, species in system["Species"].items():
                # e.g. species name ALA_xac, ALA_xag (for Amino20x4)

                atoms = Atoms(
                    species["Elements"],
                    positions=np.array(species["Positions"]),
                )
                atoms.info["head"] = "mp_pbe"
                atoms.info["subset_name"] = subset_name
                atoms.info["system_name"] = system_name
                atoms.info["species_name"] = species_name
                atoms.info["category"] = category
                atoms.info["weight"] = weight
                atoms.info["excluded"] = excluded
                atoms.info["ref_value"] = ref_value
                atoms.info["count"] = species["Count"]
                atoms.info["uhf"] = species["UHF"]
                atoms.info["Charge"] = species["Charge"]
                atoms.cell = None
                atoms.pbc = False

                atoms.calc = deepcopy(calc)
                atoms.get_potential_energy()

                system_structs.append(atoms)

            # Write out system paris
            write_dir = OUT_PATH / model_name / subset_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{system_name}.xyz", system_structs)
