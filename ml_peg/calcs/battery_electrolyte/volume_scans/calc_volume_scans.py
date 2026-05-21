"""Run calculations for Volume Scans."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest

from ml_peg.app import APP_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

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
    calc = model.get_calculator(precision="high")
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    assets_dir = APP_ROOT / "data/assets/battery_electrolyte/volume_scans" / model_name
    assets_dir.mkdir(parents=True, exist_ok=True)

    data_path = (
        download_s3_data(
            key="inputs/battery_electrolyte/volume_scans/volume_scans.zip",
            filename="volume_scans.zip",
        )
        / "volume_scans"
    )

    structure_paths = data_path.glob("*.extxyz")

    for struct_path in structure_paths:
        file_prefix = out_dir / f"{struct_path.stem[:-6]}_{model_name}_D3.xyz"
        asset_prefix = assets_dir / f"{struct_path.stem[:-6]}_{model_name}_D3.xyz"
        configs = read(struct_path, ":")
        for at in configs:
            at.calc = copy(calc)
            at.info["energy"] = at.get_potential_energy()
            at.arrays["forces"] = at.get_forces()
            at.info["virial"] = -at.get_stress(voigt=False) * at.get_volume()
            at.calc = None
        write(file_prefix, configs)
        write(asset_prefix, configs)
