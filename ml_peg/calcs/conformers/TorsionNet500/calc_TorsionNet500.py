"""
Compute the TorsionNet500 dihedral scan dataset.

TorsionNet: A Deep Neural Network to Rapidly Predict Small-Molecule
Torsional Energy Profiles with the Accuracy of Quantum Mechanics.
Rai, B. K. et al. J. Chem. Inf. Model. 2022, 62 (4), 785-800.
DOI: 10.1021/acs.jcim.1c01346
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

import pytest

# Optional extra (ml-peg[mlipaudit]); skip if not installed.
pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")

from mlipaudit.benchmarks.dihedral_scan.dihedral_scan import DihedralScanModelOutput

from ml_peg.calcs.utils.mlipaudit import MlPegDihedralScanBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_torsionnet500(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the TorsionNet500 dihedral scan dataset.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/conformers/TorsionNet500/TorsionNet500.zip",
        filename="TorsionNet500.zip",
    )

    out_path = OUT_PATH / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    benchmark = MlPegDihedralScanBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running dihedral scan benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        benchmark.model_output = DihedralScanModelOutput(fragments=[])

    (out_path / "model_output.json").write_text(
        benchmark.model_output.model_dump_json()
    )
