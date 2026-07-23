"""
Sample protein conformations with molecular dynamics.

A molecular dynamics simulation is run for each of a set of small proteins, and
the sampled backbone and side-chain dihedral angles are compared to reference
distributions to assess how well the model explores conformational space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.sampling.sampling import (
    STRUCTURE_NAMES,
    SamplingModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegSamplingBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_protein_sampling(mlip: tuple[str, Any]) -> None:
    """
    Benchmark protein conformational sampling during MD.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/biomolecules/protein_sampling/protein_sampling.zip",
        filename="protein_sampling.zip",
    )

    benchmark = MlPegSamplingBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running protein sampling benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        # Simulation states of None for every system are treated as failed
        # simulations by analyze(), which then reports a failed benchmark.
        benchmark.model_output = SamplingModelOutput(
            structure_names=list(STRUCTURE_NAMES),
            simulation_states=[None] * len(STRUCTURE_NAMES),
        )

    write_model_output_to_disk(
        MlPegSamplingBenchmark.name, benchmark.model_output, OUT_PATH / model_name
    )
