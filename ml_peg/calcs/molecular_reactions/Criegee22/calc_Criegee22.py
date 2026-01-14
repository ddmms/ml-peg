"""
Calculate the Criegee22 dataset for reactions involving Crieegee intermediates.

C. D. Smith, A. Karton. J. Comput. Chem. 2020, 41, 328–339.
DOI: 10.1002/jcc.26106
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
import zntrack

from ml_peg.calcs.utils.utils import chdir, download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV
KJ_TO_EV = units.kJ / units.mol
EV_TO_KJ = 1 / KJ_TO_EV

OUT_PATH = Path(__file__).parent / "outputs"


class Criegee22Benchmark(zntrack.Node):
    """Benchmark the Criegee22 dataset."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def run(self):
        """Run new benchmark."""
        data_path = (
            download_s3_data(
                filename="Criegee22.zip",
                key="inputs/molecular_reactions/Criegee22/Criegee22.zip",
            )
            / "Criegee22"
        )

        # Read in data and attach calculator
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        with open(data_path / "reference.txt") as lines:
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                items = line.strip().split()
                label = items[0]
                bh_ref = float(items[8]) * KJ_TO_EV
                atoms_reac = read(data_path / "structures" / f"{label}-reac.xyz")
                atoms_reac.calc = calc
                atoms_reac.info["charge"] = 0
                atoms_reac.info["spin"] = 1
                atoms_reac.info["model_energy"] = atoms_reac.get_potential_energy()
                atoms_reac.info["ref_energy"] = 0

                atoms_ts = read(data_path / "structures" / f"{label}-TS.xyz")
                atoms_ts.calc = calc
                atoms_ts.info["charge"] = 0
                atoms_ts.info["spin"] = 1
                atoms_ts.info["model_energy"] = atoms_ts.get_potential_energy()
                atoms_ts.info["ref_energy"] = bh_ref

                write_dir = OUT_PATH / self.model_name
                write_dir.mkdir(parents=True, exist_ok=True)
                write(write_dir / f"{label}_rct.xyz", atoms_reac)
                write(write_dir / f"{label}_ts.xyz", atoms_ts)


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
            benchmark = Criegee22Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_criegee22_barrier_heights():
    """Run Criegee22 benchmark via pytest."""
    build_project(repro=True)
