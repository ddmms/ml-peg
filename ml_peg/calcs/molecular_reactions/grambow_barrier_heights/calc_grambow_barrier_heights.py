"""
Compute the Grambow reaction barrier dataset.

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

from mlipaudit.benchmarks.reactivity.reactivity import ReactivityModelOutput
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegGrambowBarrierHeightsBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_grambow_barrier_heights(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the Grambow reaction barrier dataset.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular_reactions/grambow_barrier_heights/grambow_barrier_heights.zip",
        filename="grambow_barrier_heights.zip",
    )

    out_path = OUT_PATH / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    benchmark = MlPegGrambowBarrierHeightsBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running Grambow barrier heights benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        benchmark.model_output = ReactivityModelOutput(
            reaction_ids=[], energy_predictions=[]
        )

    (out_path / "model_output.json").write_text(
        benchmark.model_output.model_dump_json()
    )
