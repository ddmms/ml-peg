"""
Minimize reference geometries of small organic molecules.

Short energy minimizations of neutral and charged organic molecules from the
OpenFF industry dataset, measuring how far the relaxed geometry drifts from the
reference structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.reference_geometry_stability.reference_geometry_stability import (  # noqa: E501
    ReferenceGeometryStabilityModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegReferenceGeometryStabilityBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_reference_geometry_stability(mlip: tuple[str, Any]) -> None:
    """
    Benchmark reference geometry stability.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular/reference_geometry_stability/reference_geometry_stability.zip",
        filename="reference_geometry_stability.zip",
    )

    benchmark = MlPegReferenceGeometryStabilityBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running reference geometry stability benchmark "
            f"for {model_name}: {exc}",
            stacklevel=2,
        )
        benchmark.model_output = ReferenceGeometryStabilityModelOutput(
            openff_neutral=[], openff_charged=[]
        )

    write_model_output_to_disk(
        MlPegReferenceGeometryStabilityBenchmark.name,
        benchmark.model_output,
        OUT_PATH / model_name,
    )
