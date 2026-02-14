"""Run calculations for PLA15 benchmark."""

from __future__ import annotations

from pathlib import Path

import mlipx
from mlipx.abc import NodeWithCalculator
import zntrack

from ml_peg.calcs.supramolecular.utils.plf547_pla15_utils import run_benchmark
from ml_peg.calcs.utils.utils import chdir
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"


class PLA15Benchmark(zntrack.Node):
    """
    Benchmark model for PLA15 dataset.

    Evaluates protein-ligand interaction energies for 15 complete active site complexes.
    Each complex consists of protein, ligand, and complex structures from PDB files.
    Computes interaction energy = E(complex) - E(protein) - E(ligand)
    """

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def run(self):
        """Run PLA15 benchmark calculations."""
        run_benchmark(self, "PLA15", OUT_PATH)


def build_project(repro: bool = False) -> None:
    """
    Build mlipx project.

    Parameters
    ----------
    repro
        Whether to call dvc repro -f after building.
    """
    project = mlipx.Project()
    benchmark_node_dict = {}

    for model_name, model in MODELS.items():
        with project.group(model_name):
            benchmark = PLA15Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_pla15():
    """Run PLA15 benchmark via pytest."""
    build_project(repro=True)
