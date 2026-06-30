"""
Compute the Tautobase dataset of tautomer relative energies.

Tautobase: an open tautomer database.
Wahl, O.; Sander, T. J. Chem. Inf. Model. 2020, 60 (3), 1085-1089.
DOI: 10.1021/acs.jcim.0c00035
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegTautomersBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_tautomers(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the Tautobase dataset.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular_reactions/tautomers/tautomers.zip",
        filename="tautomers.zip",
    )

    out_path = OUT_PATH / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    benchmark = MlPegTautomersBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    benchmark.run_model()

    (out_path / "model_output.json").write_text(
        benchmark.model_output.model_dump_json()
    )
