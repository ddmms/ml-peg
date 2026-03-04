"""Calculate PLF547 benchmark. 10.1021/acs.jcim.9b01171."""

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

OUT_PATH = Path(__file__).parent / "outputs"


class PLF547Benchmark(zntrack.Node):
    """Benchmarking PLF547 subset ionic hydrogen bonds benchmark dataset."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def run(self):
        """Run new benchmark."""
        run_benchmark(self, "PLF547", OUT_PATH)


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
            benchmark = PLF547Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_plf547():
    """Run PLF547 conformation energies benchmark via pytest."""
    build_project(repro=True)
