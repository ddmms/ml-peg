"""
Compute the Folmsbee dataset of molecular conformers.

Assessing conformer energies using electronic structure and
machine learning methods

Dakota Folmsbee, Geoffrey Hutchison
International Journal of Quantum Chemistry 2020 121 (1) e26381
DOI: 10.1002/qua.26381
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegConformerSelectionBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_folmsbee(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the Folmsbee dataset.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/conformers/Folmsbee/conformer_selection.zip",
        filename="conformer_selection.zip",
    )

    out_path = OUT_PATH / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    benchmark = MlPegConformerSelectionBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    benchmark.run_model()

    (out_path / "model_output.json").write_text(
        benchmark.model_output.model_dump_json()
    )
