"""
Calculate the CYCLO70 dataset for pericyclic reaction barriers.

CYCLO70: A New Challenging Pericyclic Benchmarking Set for Kinetics
and Thermochemistry Evaluation
Javier E. Alfonso-Ramos, Carlo Adamo, Éric Brémond, and Thijs Stuyver
Journal of Chemical Theory and Computation 2025 21 (18), 8907-8917
DOI: 10.1021/acs.jctc.5c00925
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


@pytest.mark.parametrize("mlip", MODELS.items())
def test_cyclo70(mlip: tuple[str, Any]) -> None:
    """
    Run CYCLO70 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Read in data and attach calculator
    data_path = (
        download_s3_data(
            filename="CYCLO70.zip",
            key="inputs/molecular_reactions/CYCLO70/CYCLO70.zip",
        )
        / "CYCLO70"
    )

    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    with open(data_path / "dlpno-ccsdt-34.dat") as lines:
        # Skip header
        next(lines)
        for line in tqdm(lines, total=70):
            items = line.strip().split()
            if len(items) == 0:
                break
            rxn = items[0]

            bh_forward_ref = float(items[1]) * KCAL_TO_EV
            bh_reverse_ref = float(items[2]) * KCAL_TO_EV
            r_labels = [
                path.stem for path in (data_path / "XYZ_CYCLO70" / rxn).glob("r*")
            ]
            ts_labels = [
                path.stem for path in (data_path / "XYZ_CYCLO70" / rxn).glob("TS*")
            ]
            p_labels = [
                path.stem for path in (data_path / "XYZ_CYCLO70" / rxn).glob("p*")
            ]

            bh_forward_model = 0
            bh_reverse_model = 0

            write_dir = OUT_PATH / model_name
            write_dir.mkdir(parents=True, exist_ok=True)

            for atoms_label in r_labels:
                atoms = read(data_path / "XYZ_CYCLO70" / rxn / f"{atoms_label}.xyz")
                atoms.calc = calc
                if "mult" in atoms.info:
                    atoms.info["spin"] = int(atoms.info["mult"])
                else:
                    atoms.info["spin"] = 1
                if "charge" in atoms.info:
                    atoms.info["charge"] = int(atoms.info["charge"])
                else:
                    atoms.info["charge"] = 0
                bh_forward_model -= atoms.get_potential_energy()
                write(write_dir / f"{atoms_label}.xyz", atoms)

            for atoms_label in p_labels:
                atoms = read(data_path / "XYZ_CYCLO70" / rxn / f"{atoms_label}.xyz")
                atoms.calc = calc
                if "mult" in atoms.info:
                    atoms.info["spin"] = int(atoms.info["mult"])
                else:
                    atoms.info["spin"] = 1
                if "charge" in atoms.info:
                    atoms.info["charge"] = int(atoms.info["charge"])
                else:
                    atoms.info["charge"] = 0
                bh_reverse_model -= atoms.get_potential_energy()
                write(write_dir / f"{atoms_label}.xyz", atoms)

            for atoms_label in ts_labels:
                atoms = read(data_path / "XYZ_CYCLO70" / rxn / f"{atoms_label}.xyz")
                atoms.calc = calc
                if "mult" in atoms.info:
                    atoms.info["spin"] = int(atoms.info["mult"])
                else:
                    atoms.info["spin"] = 1
                if "charge" in atoms.info:
                    atoms.info["charge"] = int(atoms.info["charge"])
                else:
                    atoms.info["charge"] = 0
                bh_forward_model += atoms.get_potential_energy()
                bh_reverse_model += atoms.get_potential_energy()

                atoms.info["ref_forward_bh"] = bh_forward_ref
                atoms.info["ref_reverse_bh"] = bh_reverse_ref
                atoms.info["model_forward_bh"] = bh_forward_model
                atoms.info["model_reverse_bh"] = bh_reverse_model
                write(write_dir / f"{atoms_label}.xyz", atoms)
