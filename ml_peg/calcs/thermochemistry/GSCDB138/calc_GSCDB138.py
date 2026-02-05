"""
Calculate the thermochemistry datasets in GSCDB138 database.

https://arxiv.org/html/2508.13468v1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pandas as pd
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Thermochemistry datasets.
DATASETS = [
    "AE11",
    "AE18",
    "AL2X6",
    "ALK8",
    "AlkAtom19ALKBDE10",
    "AlkIsod14",
    "BDE99MR",
    "BDE99nonMR",
    "BH76RC",
    "BSR36",
    "CR20",
    "DARC",
    "DC13",
    "DIPCS9",
    "EA50",
    "FH51",
    "G21EA",
    "G21IP",
    "G2RC24",
    "HAT707MR",
    "HAT707nonMR",
    "HEAVYSB11",
    "HNBrBDE18",
    "IP23",
    "IP30",
    "MB08-165",
    "MB16-43",
    "MX34",
    "NBPRC",
    "P34",
    "PA26",
    "PlatonicID6",
    "PlatonicIG6",
    "PlatonicTAE6",
    "RC21",
    "RSE43",
    "SIE4x4",
    "SN13",
    "TAE_W4-17MR",
    "TAE_W4-17nonMR",
    "WCPT6",
    "YBDE18",
]


def process_atoms(atoms):
    """
    Prepare atoms with charge and spin info.

    Parameters
    ----------
    atoms
        ASE Atoms.

    Returns
    -------
    ASE.Atoms
        Same Atoms object with integer charge and spin mult.
    """
    if "charge" in atoms.info:
        atoms.info["charge"] = int(atoms.info["charge"])

    if "multiplicity" in atoms.info:
        atoms.info["spin"] = int(atoms.info["multiplicity"])
    return atoms


@pytest.mark.parametrize("mlip", MODELS.items())
def test_gscdb138(mlip: tuple[str, Any]) -> None:
    """
    Run GSCDB138 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="GSCDB138.zip",
            key="inputs/GSCDB138/GSCDB138.zip",
        )
        / "GSCDB138"
    )

    xyz_dir = data_path / "xyz_files"
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(exist_ok=True, parents=True)
    calc = model.get_calculator()
    # Add D3 calculator for this test.
    calc = model.add_d3_calculator(calc)

    for dataset in DATASETS:
        # Load dataset information.
        df_refs = pd.read_excel(data_path / "Info/DatasetEval.xlsx", header=0)
        df_refs = df_refs.loc[df_refs["Dataset"] == dataset]

        # Convert reference energies from Hartree to eV.
        df_refs["Reference"] *= units.Hartree

        # Calculate relative energy for each entry.
        for _, row in tqdm(df_refs.iterrows(), dataset):
            atoms_list = []
            identifier = row["Reaction"]
            reactions = row["Stoichiometry"].split(",")  # Parse stoichiometry string.
            e_rel_ref = row["Reference"]
            num_species = len(reactions) // 2  # Each species has coefficient and name.

            e_rel_model = 0
            # Sum up contributions from all species in the reaction.
            for i in range(num_species):
                stoi = float(reactions[2 * i])
                specy = reactions[2 * i + 1]
                atoms = process_atoms(read(xyz_dir / f"{specy}.xyz"))
                # Exclude Radon.
                if 86 in atoms.numbers:
                    continue
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                e_rel_model += stoi * energy
                atoms.info["model_energy"] = energy
                atoms.calc = None
                atoms_list.append(atoms)
            if len(atoms_list) == 0:
                continue
            atoms_list[0].info["model_rel_energy"] = e_rel_model
            atoms_list[0].info["ref_rel_energy"] = e_rel_ref
            write(write_dir / f"{identifier}.xyz", atoms_list)
