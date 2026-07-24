"""
Measure model inference speed and how it scales with system size.

For each structure in a size-stratified protein dataset, the model forward pass
(energy + forces) is timed and a short molecular dynamics run is executed per
backend. The timings are later aggregated into a scaling curve and a per-atom
forward-time metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.inference_speed.inference_speed import (
    InferenceSpeedModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegInferenceSpeedBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_inference_speed(mlip: tuple[str, Any]) -> None:
    """
    Benchmark model inference speed.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular_dynamics/inference_speed/inference_speed.zip",
        filename="inference_speed.zip",
    )

    benchmark = MlPegInferenceSpeedBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running inference speed benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        # Empty structure lists are treated as a failed benchmark by analyze().
        benchmark.model_output = InferenceSpeedModelOutput(structure_names=[])

    write_model_output_to_disk(
        "inference_speed", benchmark.model_output, OUT_PATH / model_name
    )
