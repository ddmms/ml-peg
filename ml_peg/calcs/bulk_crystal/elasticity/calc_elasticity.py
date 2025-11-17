"""Run calculations for elasticity benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matcalc.benchmark import ElasticityBenchmark
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"


def run_elasticity_benchmark(
    *,
    calc,
    model_name: str,
    out_dir: Path,
    n_jobs: int = -1,
    norm_strains: tuple[float, float, float, float] = (-0.1, -0.05, 0.05, 0.1),
    shear_strains: tuple[float, float, float, float] = (-0.02, -0.01, 0.01, 0.02),
    relax_structure: bool = True,
    relax_deformed_structures: bool = True,
    use_checkpoint: bool = True,
    n_materials: int = 5,
    fmax: float = 0.05,
) -> None:
    """
    Run elasticity benchmark and write results to CSV.

    Parameters
    ----------
    calc
        ASE calculator for evaluating structures.
    model_name
        Name of MLIP model.
    out_dir
        Directory to write per-model outputs.
    n_jobs
        Number of parallel workers for the benchmark.
    norm_strains
        Tuple of normal strains to apply.
    shear_strains
        Tuple of shear strains to apply.
    relax_structure
        Whether to relax the equilibrium structure before deformations.
    relax_deformed_structures
        Whether to relax each strained structure.
    use_checkpoint
        If True, writes intermediate checkpoints inside ``out_dir/checkpoints``.
    n_materials
        Number of materials sampled from the benchmark set.
    fmax
        Force threshold for structural relaxations.
    """
    benchmark = ElasticityBenchmark(
        n_samples=n_materials,
        seed=2025,
        fmax=fmax,
        relax_structure=relax_structure,
        relax_deformed_structures=relax_deformed_structures,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
        benchmark_name="mp-pbe-elasticity-2025.3.json.gz",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = None
    if use_checkpoint:
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{model_name}_checkpoint.json"

    results = benchmark.run(
        calc,
        model_name,
        n_jobs=n_jobs,
        checkpoint_file=str(checkpoint_file) if checkpoint_file else None,
        checkpoint_freq=100,
        delete_checkpoint_on_finish=False,
    )

    results.to_csv(out_dir / "moduli_results.csv", index=False)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_elasticity(mlip: tuple[str, Any]) -> None:
    """
    Run elasticity benchmark for a single model.

    Parameters
    ----------
    mlip
        Model entry containing name and object capable of providing a calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    run_elasticity_benchmark(
        calc=calc,
        model_name=model_name,
        out_dir=OUT_PATH / model_name,
    )
