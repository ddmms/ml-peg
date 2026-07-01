"""
Run molecular dynamics stability simulations.

Short MD runs for small molecules, peptides and proteins in vacuum and
solvent, checking how many simulations complete without error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegStabilityBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_stability(mlip: tuple[str, Any]) -> None:
    """
    Benchmark molecular dynamics stability.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.get_calculator(precision="low")

    data_input_dir = download_s3_data(
        key="inputs/molecular_dynamics/stability/stability.zip",
        filename="stability.zip",
    )

    benchmark = MlPegStabilityBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    benchmark.run_model()

    write_model_output_to_disk(
        "stability", benchmark.model_output, OUT_PATH / model_name
    )
