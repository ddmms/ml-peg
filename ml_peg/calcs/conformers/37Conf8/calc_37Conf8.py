"""
Compute the 37Conf8 dataset for molecular conformer relative energies.

10.1002/cphc.201801063.
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


class Benchmark37Conf8(zntrack.Node):
    """Benchmark the 37Conf8 dataset."""

    model: NodeWithCalculator = zntrack.deps()
    model_name: str = zntrack.params()

    def run(self):
        """Run new benchmark."""
        data_path = (
            download_s3_data(
                filename="37CONF8.zip",
                key="inputs/conformers/37Conf8/37Conf8.zip",
            )
            / "37CONF8"
        )

        df = pd.read_excel(
            data_path / "37Conf8_data.xlsx", sheet_name="Rel_Energy_SP", header=2
        )
        calc = self.model.get_calculator()
        # Add D3 calculator for this test
        calc = self.model.add_d3_calculator(calc)

        write_dir = OUT_PATH / self.model_name
        write_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(len(df) - 3)):
            molecule_name = df.iloc[i][0].strip()
            conf_id = int(df.iloc[i][1])
            label = f"{molecule_name}_{conf_id}"
            if conf_id == 1:
                zero_conf = read(data_path / "PBEPBE-D3" / f"{label}_PBEPBE-D3.xyz")
                zero_conf.info["charge"] = 0
                zero_conf.info["spin"] = 1
                zero_conf.calc = calc
                e_model_zero_conf = zero_conf.get_potential_energy()
            else:
                atoms = read(data_path / "PBEPBE-D3" / f"{label}_PBEPBE-D3.xyz")
                atoms.info["charge"] = 0
                atoms.info["spin"] = 1
                atoms.calc = calc
                atoms.info["model_rel_energy"] = (
                    atoms.get_potential_energy() - e_model_zero_conf
                )
                atoms.info["ref_energy"] = float(df.iloc[i][2]) * KCAL_TO_EV
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
            benchmark = Benchmark37Conf8(
                model=model,
                model_name=model_name,
            )
            benchmark_node_dict[model_name] = benchmark

    if repro:
        with chdir(Path(__file__).parent):
            project.repro(build=True, force=True)
    else:
        project.build()


def test_37conf8_conformer_energies():
    """Run 37Conf8 benchmark via pytest."""
    build_project(repro=True)
