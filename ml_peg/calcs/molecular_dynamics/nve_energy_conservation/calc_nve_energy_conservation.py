"""
Measure total energy conservation during NVE molecular dynamics.

A microcanonical (NVE) molecular dynamics simulation is run for each of a set of
representative systems, and the total mechanical energy ``E = PE + KE`` is recorded
along the trajectory. Without a thermostat, the total energy is a conserved quantity
of the dynamics, so any systematic drift is a defect of the potential or of the
forces derived from it.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from warnings import warn

from ase.data import chemical_symbols
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.nve_energy_conservation.nve_energy_conservation import (
    NVEEnergyConservationBenchmark,
    NVEEnergyConservationModelOutput,
)

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

BENCHMARK = NVEEnergyConservationBenchmark.name

OUT_PATH = Path(__file__).parent / "outputs"

# mlipaudit requires an external ASE calculator to declare the elements it
# supports as `allowed_atomic_numbers`, and uses it to skip systems the model
# cannot handle. ml-peg's calculators do not narrow this per model, so the whole
# periodic table is declared and systems a model cannot handle are reported as
# failed rather than skipped, consistent with the other MLIP Audit benchmarks.
ALLOWED_ATOMIC_NUMBERS = set(range(1, len(chemical_symbols)))


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_nve_energy_conservation(mlip: tuple[str, Any]) -> None:
    """
    Benchmark total energy conservation during NVE molecular dynamics.

    Parameters
    ----------
    mlip
        Name of model and model object to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="low")
    calc = model.add_d3_calculator(calc)
    calc.allowed_atomic_numbers = ALLOWED_ATOMIC_NUMBERS

    data_input_dir = download_s3_data(
        key=f"inputs/molecular_dynamics/{BENCHMARK}/{BENCHMARK}.zip",
        filename=f"{BENCHMARK}.zip",
    )

    out_path = OUT_PATH / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Copy the input structures into the outputs, so the analysis can rebuild the
    # benchmark without downloading the data again.
    shutil.copytree(
        data_input_dir / BENCHMARK, OUT_PATH / BENCHMARK, dirs_exist_ok=True
    )

    benchmark = NVEEnergyConservationBenchmark(
        force_field=calc,
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    try:
        benchmark.run_model()
    except Exception as exc:
        warn(
            f"Error running NVE energy conservation benchmark for {model_name}: {exc}",
            stacklevel=2,
        )
        # An empty set of systems is treated as a failed benchmark by analyze().
        benchmark.model_output = NVEEnergyConservationModelOutput(
            structure_names=[],
            num_atoms=[],
            times_ps=[],
            potential_energies_ev=[],
            kinetic_energies_ev=[],
            skipped_structures=[],
        )

    (out_path / "model_output.json").write_text(
        benchmark.model_output.model_dump_json()
    )
