"""Run calculations for Elemental Transition Metal Vacancy Formation tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_vacancy_formation_energy(mlip: tuple[str, Any]) -> None:
    """
    Run vacancy formation energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Download dataset
    elemental_tm_vacancies_dir = (
        download_s3_data(
            key="inputs/bulk_crystal/elemental_tm_vacancies/Elemental_TM_Vacancy.zip",
            filename="Elemental_TM_Vacancy.zip",
        )
        / "Elemental_TM_Vacancy"
    )

    with open(elemental_tm_vacancies_dir / "list") as f:
        systems = f.read().splitlines()

    for system in systems:
        bulk_path = elemental_tm_vacancies_dir / system / "CONTCAR_bulk"
        vacancy_path = elemental_tm_vacancies_dir / system / "CONTCAR_vacancy"
        ref_path = elemental_tm_vacancies_dir / system / "vacancy_formation_energy_PBE"

        bulk = read(bulk_path, index=0, format="vasp")
        # Set default charge and spin
        bulk.info.setdefault("charge", 0)
        bulk.info.setdefault("spin", 1)
        bulk.calc = calc
        bulk.get_potential_energy()

        vacancy = read(vacancy_path, index=0, format="vasp")
        # Set default charge and spin
        vacancy.info.setdefault("charge", 0)
        vacancy.info.setdefault("spin", 1)
        vacancy.calc = copy(calc)
        vacancy.get_potential_energy()

        ref = np.loadtxt(ref_path)

        bulk.info["ref"] = ref
        bulk.info["system"] = system
        vacancy.info["ref"] = ref
        vacancy.info["system"] = system

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{system}.xyz", [bulk, vacancy])
