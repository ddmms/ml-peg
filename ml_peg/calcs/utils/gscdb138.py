"""Shared functionality for GSCDB138 tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read, write
import numpy as np
import pandas as pd
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data


def process_atoms(atoms: Atoms) -> Atoms:
    """
    Prepare atoms with charge and spin info.

    Parameters
    ----------
    atoms
        ASE Atoms.

    Returns
    -------
    ASE.Atoms
        Same Atoms object with integer charge and spin multiplicity.
    """
    # Remove the barycnter of the positions to atoms for translation invariance.
    atoms.positions -= np.mean(atoms.positions, axis=0)

    if "charge" in atoms.info:
        atoms.info["charge"] = int(np.rint(atoms.info["charge"]))
    else:
        atoms.info["charge"] = 0
    if "multiplicity" in atoms.info:
        atoms.info["spin"] = int(np.rint(atoms.info["multiplicity"]))
    else:
        atoms.info["spin"] = 1
    if "multipole_field" in atoms.info:
        try:
            field = atoms.info["multipole_field"]
            field_dict = {}

            for item in field.split(","):
                if item.strip():
                    key, value = item.split(":")
                    field_dict[key.strip().strip("'")] = float(value.strip())

            x = units.Hartree / units.Bohr * field_dict.get("X", 0.0)
            y = units.Hartree / units.Bohr * field_dict.get("Y", 0.0)
            z = units.Hartree / units.Bohr * field_dict.get("Z", 0.0)

            atoms.info["external_field"] = np.array([x, y, z])  # in V/Å
        except Exception:
            atoms.info["external_field"] = np.array([0.0, 0.0, 0.0])
    return atoms


def run_gscdb138(
    mlip: tuple[str, Any],
    datasets: list[str],
    out_path: Path,
    exclude_elements: tuple[int] = (),
) -> None:
    """
    Run GSCDB138 test for specific model and datasets.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    datasets
        Datasets to run benchmark for.
    out_path
        Directory to write out data to.
    exclude_elements
        Elements to exclude from calculations. Default is all elements.
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
    write_dir = out_path / model_name
    write_dir.mkdir(exist_ok=True, parents=True)
    calc = model.get_calculator()
    # Add D3 calculator for this test.
    calc = model.add_d3_calculator(calc)

    for dataset in datasets:
        # Load dataset information.
        df_refs = pd.read_excel(data_path / "Info/DatasetEval.xlsx", header=0)
        df_refs = df_refs.loc[df_refs["Dataset"] == dataset]

        # Convert reference energies from Hartree to eV.
        df_refs["Reference"] *= units.Hartree

        # Calculate relative energy for each entry.
        for _, row in tqdm(df_refs.iterrows(), dataset, total=df_refs.shape[0]):
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
                if any(element in exclude_elements for element in atoms.numbers):
                    print(f"Skipping {specy}")
                    continue
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                e_rel_model += stoi * energy
                atoms.info["model_energy"] = energy
                atoms.calc = None
                atoms_list.append(atoms)
            if len(atoms_list) == 0:
                print(f"No calculations run for {identifier}")
                continue
            atoms_list[0].info["model_rel_energy"] = e_rel_model
            atoms_list[0].info["ref_rel_energy"] = e_rel_ref
            # Write dataset name to first atoms for bookkeeping.
            atoms_list[0].info["dataset"] = dataset
            write(write_dir / f"{dataset}_{identifier}.xyz", atoms_list)
