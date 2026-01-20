"""Run calculations for Volume Scans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
MODELS = dict(list(MODELS.items())[:-1])
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_volume_scans(mlip: tuple[str, Any]) -> None:
    """
    Run calculations required for Volume Scan tests.

    Parameters
    ----------
    mlip
        Name of models and models used for Volume Scan calculations.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    struct_paths = DATA_PATH.glob("*.xyz")

    for struct_path in struct_paths:
        file_prefix = (
            OUT_PATH
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
