"""
Assess protein folding stability during molecular dynamics.

A molecular dynamics simulation is run for each of a set of small proteins
starting from their native (folded) conformation, and the ability of the model
to keep each protein folded is measured along the trajectory via the RMSD
relative to the reference structure.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from warnings import warn

import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.folding_stability.folding_stability import (
    FoldingStabilityModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
from mlipaudit.utils.biomolecules import STRUCTURE_NAMES

from ml_peg.calcs.utils.mlipaudit import MlPegFoldingStabilityBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Directory the downloaded input data is extracted to.
# TODO: rename the uploaded data directory to "folding_stability" and drop this
# constant, so the download matches the directory mlipaudit reads from.
DOWNLOAD_DATA_DIR = "protein_folding_stability"

# mlipaudit reads the input structures from ``{data_input_dir}/{data_name or name}``,
# so the data must be copied to a directory of this name for the benchmark to find it.
BENCHMARK_DATA_DIR = (
    MlPegFoldingStabilityBenchmark.data_name or MlPegFoldingStabilityBenchmark.name
)


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
    calc = model.get_calculator(precision="low")
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/biomolecules/protein_folding_stability/protein_folding_stability.zip",
        filename="protein_folding_stability.zip",
    )

    # Save the input data to the calculation outputs, using the directory name
    # mlipaudit expects, so the benchmark runs on the downloaded data rather than
    # fetching it again, and the analysis is self contained.
    shutil.copytree(
        data_input_dir / DOWNLOAD_DATA_DIR,
        OUT_PATH / BENCHMARK_DATA_DIR,
        dirs_exist_ok=True,
    )

    benchmark = MlPegFoldingStabilityBenchmark(
        force_field=calc,
        data_input_dir=OUT_PATH,
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
