"""
Compute the DipCONFS dataset for amino acid and peptide conformers.

Toward Reliable Conformational Energies of Amino Acids and
Dipeptides─The DipCONFS Benchmark and DipCONL Datasets

Christoph Plett, Stefan Grimme, and Andreas Hansen
Journal of Chemical Theory and Computation 2024 20 (18), 8329-8339
DOI: 10.1021/acs.jctc.4c00801
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import mlipx
from mlipx.abc import NodeWithCalculator
import pandas as pd
from tqdm import tqdm
import zntrack

from ml_peg.calcs.utils.utils import chdir, download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"


class DipCONFSBenchmark(zntrack.Node):
    """Benchmarking DipCONFS conformers."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    @staticmethod
    def get_atoms(atoms_path):
        """
        Read atoms object and add charge and spin.

        Parameters
        ----------
        atoms_path
            Path to atoms object.

        Returns
        -------
        atoms
            ASE atoms object.
        """
        atoms = read(atoms_path)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        return atoms

    def run(self):
        """Run new benchmark."""
        # Read in data and attach calculator
        data_path = (
            download_s3_data(
                filename="DipCONFS.zip",
                key="inputs/conformers/DipCONFS/DipCONFS.zip",
            )
            / "DipCONFS"
        )

        # Read in data and attach calculator
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        df = pd.read_excel(
            data_path / "ct4c00801_si_004.xlsx",
            sheet_name="Conformational Energies in kcal",
        )

        for zero_conf_label, label, e_rel_ref in tqdm(
            zip(
                df["Reference Conformer"].tolist(),
                df["Conformer"].tolist(),
                df["PNO-LCCSD(T)-F12b/AVQZ’"].tolist(),
                strict=True,
            )
        ):
            # Get reference energy
            e_rel_ref = float(e_rel_ref) * KCAL_TO_EV
            zero_conf_label = zero_conf_label.replace("/", "-")
            label = label.replace("/", "-")

            # Get zero ref conformer model energy
            zero_conf = self.get_atoms(data_path / zero_conf_label / "struc.xyz")
            zero_conf.calc = calc
            e_model_zero_conf = zero_conf.get_potential_energy()

            # Get current conformer model energy
            atoms = self.get_atoms(data_path / label / "struc.xyz")
            atoms.calc = calc
            atoms.info["model_rel_energy"] = (
                atoms.get_potential_energy() - e_model_zero_conf
            )
            atoms.info["ref_energy"] = e_rel_ref

            write_dir = OUT_PATH / self.model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{label}.xyz", atoms)


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
            benchmark = DipCONFSBenchmark(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_dipconfs():
    """Run DipCONFS benchmark via pytest."""
    build_project(repro=True)
