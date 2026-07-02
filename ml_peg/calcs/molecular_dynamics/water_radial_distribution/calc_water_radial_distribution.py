"""
Compute the water oxygen-oxygen radial distribution function.

A short NVT molecular dynamics simulation of a box of 500 water molecules is
run, and the O-O radial distribution function is compared to experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.water_radial_distribution.water_radial_distribution import (
    WaterRadialDistributionModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegWaterRadialDistributionBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_water_radial_distribution(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the water radial distribution function.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular_dynamics/water_radial_distribution/water_radial_distribution.zip",
        filename="water_radial_distribution.zip",
    )

    benchmark = MlPegWaterRadialDistributionBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running water RDF benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        benchmark.model_output = WaterRadialDistributionModelOutput(failed=True)

    write_model_output_to_disk(
        "water_radial_distribution", benchmark.model_output, OUT_PATH / model_name
    )
