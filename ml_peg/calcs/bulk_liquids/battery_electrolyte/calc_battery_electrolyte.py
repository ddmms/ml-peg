"""Run calculations for battery system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
DENS_FACT = (units.m / 1.0e2) ** 3 / units.mol


@pytest.mark.parametrize("mlip", MODELS.items())
def test_intra_inter(mlip: tuple[str, Any]) -> None:
    """
    Run calculations required for intra/inter molecule property comparison.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    struct_paths = (DATA_PATH / "Intra_Inter_data").glob("*.xyz")

    for struct_path in struct_paths:
        file_prefix = (
            OUT_PATH
            / "Intra_Inter_output"
            / f"{struct_path.stem[:-4]}_{model_name}.xyz"
        )
        configs = read(struct_path, ":")
        for at in configs:
            at.calc = calc
            at.info["energy"] = at.get_potential_energy()
            at.arrays["forces"] = at.get_forces()
            at.info["virial"] = -at.get_stress(voigt=False) * at.get_volume()
            at.calc = None
        write(file_prefix, configs)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_volume_scans(mlip: tuple[str, Any]) -> None:
    """
    Run calculations required for Volume Scan tests.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    struct_paths = (DATA_PATH / "Volume_Scan_data").glob("*.xyz")

    for struct_path in struct_paths:
        file_prefix = (
            OUT_PATH
            / "Volume_Scan_output"
            / f"{struct_path.stem[:-6]}_{model_name}_D3.xyz"
        )
        configs = read(struct_path, ":")
        for at in configs:
            at.calc = calc
            at.info["energy"] = at.get_potential_energy()
            at.arrays["forces"] = at.get_forces()
            at.info["virial"] = -at.get_stress(voigt=False) * at.get_volume()
            at.calc = None
        write(file_prefix, configs)
