"""
Compute solvent radial distribution functions from molecular dynamics.

A short NVT molecular dynamics simulation is run for a box of each solvent
(carbon tetrachloride, methanol and acetonitrile), and the radial distribution
function of the atom of interest is compared to experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from mlipaudit.benchmarks.solvent_radial_distribution.solvent_radial_distribution import (  # noqa: E501
    SolventRadialDistributionModelOutput,
)
from mlipaudit.io import write_model_output_to_disk
import pytest

from ml_peg.calcs.utils.mlipaudit import MlPegSolventRadialDistributionBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_solvent_radial_distribution(mlip: tuple[str, Any]) -> None:
    """
    Benchmark the solvent radial distribution functions.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    calc = model.add_d3_calculator(calc)

    data_input_dir = download_s3_data(
        key="inputs/molecular_dynamics/solvent_radial_distribution/solvent_radial_distribution.zip",
        filename="solvent_radial_distribution.zip",
    )

    benchmark = MlPegSolventRadialDistributionBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running solvent RDF benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        # Empty structure lists are treated as a failed benchmark by analyze().
        benchmark.model_output = SolventRadialDistributionModelOutput(
            structure_names=[], simulation_states=[]
        )

    write_model_output_to_disk(
        "solvent_radial_distribution", benchmark.model_output, OUT_PATH / model_name
    )
