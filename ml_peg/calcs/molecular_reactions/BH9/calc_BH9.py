"""
Calculate the BH9 reaction barriers dataset.

Journal of Chemical Theory and Computation 2022 18 (1), 151-166
DOI: 10.1021/acs.jctc.1c00694
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"


def process_atoms(path: Path):
    """
    Get the ASE Atoms object with prepared charge and spin states.

    Parameters
    ----------
    path
        Path to the system xyz.

    Returns
    -------
    ase.Atoms
        ASE Atoms object of the system.
    """
    with open(path) as lines:
        for i, line in enumerate(lines):
            if i == 1:
                charge, spin = map(int, line.split())
                break

    atoms = read(path)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    return atoms


def parse_cc_energy(fname: Path) -> float:
    """
    Get the CCSD barrier from the data file.

    Parameters
    ----------
    fname
        Path to the reference data file.

    Returns
    -------
    float
        Reaction barrier in eV.
    """
    with open(fname) as lines:
        for line in lines:
            if "ref" in line:
                items = line.strip().split()
                break
    return float(items[1]) * KCAL_TO_EV


def sort_labels(path: Path) -> tuple[int, int]:
    """
    Sort structure labels.

    Parameters
    ----------
    path
        Globbed path of the form 01_02TS.xyz.

    Returns
    -------
    tuple[int, int]
        Integer values of of the first and second numbers in the filename (e.g. 1 and
        2 in the above example) to for numerical sorting.
    """
    x, y = path.stem.split("_")
    return int(x), int(y.removesuffix("TS"))


def get_ref_energies(data_path: Path) -> dict[str, dict[str, float]]:
    """
    Get the reference barriers.

    Parameters
    ----------
    data_path
        Path to the dataset directory.

    Returns
    -------
    dict[str, dict[str, float]]
        Loaded reference energies.
    """
    ref_energies = {}
    labels = [
        path.stem.replace("TS", "")
        for path in sorted(
            (data_path / "BH9_SI" / "XYZ_files").glob("*TS.xyz"), key=sort_labels
        )
    ]
    rxn_count = 0

    for label in labels:
        ref_energies[label] = {}
        rxn_count += 1
        for direction in ["forward", "reverse"]:
            ref_fname = (
                data_path
                / "BH9_SI"
                / "DB_files"
                / "BH"
                / f"BH9-BH_{rxn_count}_{direction}.db"
            )
            ref_energies[label][direction] = parse_cc_energy(ref_fname)

    return ref_energies


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bh9(mlip: tuple[str, Any]) -> None:
    """
    Run BH9 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="BH9.zip",
            key="inputs/molecular_reactions/BH9/BH9.zip",
        )
        / "BH9"
    )
    # Read in data and attach calculator
    ref_energies = get_ref_energies(data_path)
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # sort_labels not required here as we have the correct labels
    for fname in tqdm(sorted((data_path / "BH9_SI" / "XYZ_files").glob("*TS.xyz"))):
        atoms = process_atoms(fname)
        atoms.calc = calc
        atoms.info["model_energy"] = atoms.get_potential_energy()

        # Write both forward and reverse barriers, only forward used in analysis here
        label = fname.stem
        label = label.replace("TS", "")
        atoms.info["ref_forward_barrier"] = ref_energies[label]["forward"]
        atoms.info["ref_reverse_barrier"] = ref_energies[label]["reverse"]

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{fname.stem}.xyz", atoms)
