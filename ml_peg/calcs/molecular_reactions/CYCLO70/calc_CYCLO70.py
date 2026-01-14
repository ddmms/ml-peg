"""
Calculate the CYCLO70 dataset for pericyclic reaction barriers.

CYCLO70: A New Challenging Pericyclic Benchmarking Set for Kinetics
and Thermochemistry Evaluation
Javier E. Alfonso-Ramos, Carlo Adamo, Éric Brémond, and Thijs Stuyver
Journal of Chemical Theory and Computation 2025 21 (18), 8907-8917
DOI: 10.1021/acs.jctc.5c00925
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
from tqdm import tqdm
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


class CYCLO70Benchmark(zntrack.Node):
    """Benchmark the CYCLO70 reaction benchmark."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def run(self):
        """Run new benchmark."""
        # Read in data and attach calculator
        data_path = (
            download_s3_data(
                filename="CYCLO70.zip",
                key="inputs/molecular_reactions/CYCLO70/CYCLO70.zip",
            )
            / "CYCLO70"
        )

        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        with open(data_path / "dlpno-ccsdt-34.dat") as lines:
            for i, line in tqdm(enumerate(lines)):
                if i == 0:
                    continue
                items = line.strip().split()
                if len(items) == 0:
                    break
                rxn = items[0]

                bh_forward_ref = float(items[1]) * KCAL_TO_EV
                bh_reverse_ref = float(items[2]) * KCAL_TO_EV
                r_labels = [
                    path.stem for path in (data_path / "XYZ_CYCLO70" / rxn).glob("r*")
                ]
                ts_labels = [
                    path.stem for path in (data_path / "XYZ_CYCLO70" / rxn).glob("TS*")
                ]
                p_labels = [
                    path.stem for path in (data_path / "XYZ_CYCLO70" / rxn).glob("p*")
                ]

                bh_forward_model = 0
                bh_reverse_model = 0

                write_dir = OUT_PATH / self.model_name
                write_dir.mkdir(parents=True, exist_ok=True)

                for atoms_label in r_labels:
                    atoms = read(data_path / "XYZ_CYCLO70" / rxn / f"{atoms_label}.xyz")
                    atoms.calc = calc
                    bh_forward_model -= atoms.get_potential_energy()
                    write(write_dir / f"{atoms_label}.xyz", atoms)

                for atoms_label in p_labels:
                    atoms = read(data_path / "XYZ_CYCLO70" / rxn / f"{atoms_label}.xyz")
                    atoms.calc = calc
                    bh_reverse_model -= atoms.get_potential_energy()
                    write(write_dir / f"{atoms_label}.xyz", atoms)

                for atoms_label in ts_labels:
                    atoms = read(data_path / "XYZ_CYCLO70" / rxn / f"{atoms_label}.xyz")
                    atoms.calc = calc
                    bh_forward_model += atoms.get_potential_energy()
                    bh_reverse_model += atoms.get_potential_energy()

                    atoms.info["ref_forward_bh"] = bh_forward_ref
                    atoms.info["ref_reverse_bh"] = bh_reverse_ref
                    atoms.info["model_forward_bh"] = bh_forward_model
                    atoms.info["model_reverse_bh"] = bh_reverse_model
                    write(write_dir / f"{atoms_label}.xyz", atoms)


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
            benchmark = CYCLO70Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_cyclo70_barrier_heights():
    """Run CYCLO70 benchmark via pytest."""
    build_project(repro=True)
