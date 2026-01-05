"""
Calculate the BH9 reaction barriers dataset.

Journal of Chemical Theory and Computation 2022 18 (1), 151-166
DOI: 10.1021/acs.jctc.1c00694
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

OUT_PATH = Path(__file__).parent / "outputs"


def process_atoms(path):
    """
    Get the ASE Atoms object with prepared charge and spin states.

    Parameters
    ----------
    path
        Path to the system xyz.

    Returns
    -------
    ase.Atoms
        ASE Atoms object of the system.
    """
    with open(path) as lines:
        for i, line in enumerate(lines):
            if i == 1:
                items = line.strip().split()
                charge = int(items[0])
                spin = int(items[1])

    atoms = read(path)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    return atoms


def parse_cc_energy(fname):
    """
    Get the CCSD barrier from the data file.

    Parameters
    ----------
    fname
        Path to the reference data file.

    Returns
    -------
    float
        Reaction barrier in eV.
    """
    with open(fname) as lines:
        for line in lines:
            if "ref" in line:
                items = line.strip().split()
                break
    return float(items[1]) * KCAL_TO_EV


class BH9Benchmark(zntrack.Node):
    """Benchmark the BH9 reaction benchmark."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def get_ref_energies(self, data_path):
        """
        Get the reference barriers.

        Parameters
        ----------
        data_path
            Path to the dataset directory.
        """
        self.ref_energies = {}
        labels = [
            path.stem.replace("TS", "")
            for path in sorted((data_path / "BH9_SI" / "XYZ_files").glob("*TS.xyz"))
        ]
        rxn_count = 0
        for label in labels:
            self.ref_energies[label] = {}
            rxn_count += 1
            for direction in ["forward", "reverse"]:
                ref_fname = (
                    data_path
                    / "BH9_SI"
                    / "DB_files"
                    / "BH"
                    / f"BH9-BH_{rxn_count}_{direction}.db"
                )
                self.ref_energies[label][direction] = parse_cc_energy(ref_fname)

    def run(self):
        """Run new benchmark."""
        data_path = (
            download_s3_data(
                filename="BH9.zip",
                key="inputs/molecular_reactions/BH9/BH9.zip",
            )
            / "BH9"
        )
        # Read in data and attach calculator
        self.get_ref_energies(data_path)
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        for fname in tqdm(sorted((data_path / "BH9_SI" / "XYZ_files").glob("*TS.xyz"))):
            atoms = process_atoms(fname)
            atoms.calc = calc
            atoms.info["model_energy"] = atoms.get_potential_energy()

            """
            Write both forward and reverse barriers,
            only forward will be used in analysis here.
            """
            label = fname.stem
            if "TS" in label:
                label = label.replace("TS", "")
                atoms.info["ref_forward_barrier"] = self.ref_energies[label]["forward"]
                atoms.info["ref_reverse_barrier"] = self.ref_energies[label]["reverse"]

            write_dir = OUT_PATH / self.model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{fname.stem}.xyz", atoms)


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
            benchmark = BH9Benchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_bh9_barrier_heights():
    """Run BH9 barriers benchmark via pytest."""
    build_project(repro=True)
