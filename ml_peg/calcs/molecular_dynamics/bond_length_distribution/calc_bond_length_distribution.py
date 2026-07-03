"""
Measure covalent bond length deviations during molecular dynamics.

A molecular dynamics simulation is run for each of a set of small organic
molecules, and the deviation of a tracked covalent bond from its reference
equilibrium length is measured over the trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.bond_length_distribution.bond_length_distribution import (
    BondLengthDistributionModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegBondLengthDistributionBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bond_length_distribution(mlip: tuple[str, Any]) -> None:
    """
    Benchmark covalent bond length deviations during MD.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular_dynamics/bond_length_distribution/bond_length_distribution.zip",
        filename="bond_length_distribution.zip",
    )

    benchmark = MlPegBondLengthDistributionBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running bond length distribution benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        # An empty set of molecules is treated as a failed benchmark by analyze().
        benchmark.model_output = BondLengthDistributionModelOutput(molecules=[])

    write_model_output_to_disk(
        "bond_length_distribution", benchmark.model_output, OUT_PATH / model_name
    )
