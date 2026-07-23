"""
Assess protein folding stability during molecular dynamics.

A molecular dynamics simulation is run for each of a set of small proteins
starting from their native (folded) conformation, and the ability of the model
to keep each protein folded is measured along the trajectory via the RMSD,
TM score, and radius of gyration relative to the reference structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.folding_stability.folding_stability import (
    STRUCTURE_NAMES,
    FoldingStabilityModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegFoldingStabilityBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_protein_folding_stability(mlip: tuple[str, Any]) -> None:
    """
    Benchmark protein folding stability during MD.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/biomolecules/protein_folding_stability/protein_folding_stability.zip",
        filename="protein_folding_stability.zip",
    )

    benchmark = MlPegFoldingStabilityBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running protein folding stability benchmark for "
            f"{model_name}: {exc}",
            stacklevel=2,
        )
        # Structures with a ``None`` simulation state are treated as failed by
        # analyze(), so this yields a failed result for every structure.
        benchmark.model_output = FoldingStabilityModelOutput(
            structure_names=list(STRUCTURE_NAMES),
            simulation_states=[None] * len(STRUCTURE_NAMES),
        )

    write_model_output_to_disk(
        MlPegFoldingStabilityBenchmark.name,
        benchmark.model_output,
        OUT_PATH / model_name,
    )
