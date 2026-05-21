"""Run calculations for Volume Scans."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

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
    calc = model.get_calculator(precision="low")
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    data_path = download_s3_data(
        key="inputs/battery_electrolyte/volume_scans/volume_scans.zip",
        filename="volume_scans.zip",
    )

    for struct_path in data_path:
        file_prefix = OUT_PATH / f"{struct_path.stem[:-6]}_{model_name}_D3.xyz"
        configs = read(struct_path, ":")
        for at in configs:
            at.calc = copy(calc)
            at.info["energy"] = at.get_potential_energy()
            at.arrays["forces"] = at.get_forces()
            at.info["virial"] = -at.get_stress(voigt=False) * at.get_volume()
            at.calc = None
        write(file_prefix, configs)
