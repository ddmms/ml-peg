from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import pytest
from ase.io import read, write

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models import current_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_benchmark(mlip: tuple[str, Any]) -> None:
    """
    Run calculations required for lithium diffusion along path B.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_path = (
        download_s3_data(
            key="inputs/[category]/[benchmark_name]/[benchmark_name].zip",
            filename="[benchmark_name].zip",
        )
        / "[benchmark_name]"
    )
    struct = read(data_path / "struct.xyz")
    struct.info["charge"] = 0
    struct.info["spin"] = 1
    struct.calc = calc

    try:
        struct.info["pred_energy"] = struct.get_potential_energy()
    except Exception as exc:
        warn(f"Error calculating {model_name}: {exc}", stacklevel=2)
        struct.info["pred_energy"] = np.nan

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / "struct.extxyz", struct)