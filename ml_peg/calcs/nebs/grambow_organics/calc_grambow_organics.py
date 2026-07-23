"""
Run nudged elastic band simulations for the Grambow organics dataset.

100 elementary organic reactions sampled from the Grambow dataset, each run as
a NEB simulation to assess whether the model can converge the reaction path.

Grambow, C.A., Pattanaik, L. & Green, W.H.
Reactants, products, and transition states of elementary chemical reactions
based on quantum chemistry.
Sci Data 7, 137 (2020).
DOI: 10.1038/s41597-020-0460-4
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import NEBModelOutput
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegGrambowOrganicsBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_grambow_organics(mlip: tuple[str, Any]) -> None:
    """
    Benchmark NEB convergence for the Grambow organics dataset.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/nebs/grambow_organics/grambow_organics.zip",
        filename="grambow_organics.zip",
    )

    benchmark = MlPegGrambowOrganicsBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running Grambow organics NEB benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        benchmark.model_output = NEBModelOutput(simulation_states=[])

    write_model_output_to_disk(
        MlPegGrambowOrganicsBenchmark.name,
        benchmark.model_output,
        OUT_PATH / model_name,
    )
