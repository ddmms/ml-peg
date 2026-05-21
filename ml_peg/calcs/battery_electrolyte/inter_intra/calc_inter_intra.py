"""Run calculations for inter intra benchmark."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
from aseMolec import anaAtoms
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_intra_inter(mlip: tuple[str, Any]) -> None:
    """
    Run calculations required for intra/inter molecule property comparison.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = (
        download_s3_data(
            key="inputs/battery_electrolyte/inter_intra/inter_intra.zip",
            filename="inter_intra.zip",
        )
        / "inter_intra"
    )

    structure_paths = data_path.glob("*.xyz")

    for struct_path in structure_paths:
        file_prefix = out_dir / f"{struct_path.stem[:-6]}_{model_name}_D3.xyz"
        configs = read(struct_path, ":")
        for struct in configs:
            struct.calc = copy(calc)
            struct.info["energy"] = struct.get_potential_energy()
            struct.arrays["forces"] = struct.get_forces()
            struct.info["virial"] = (
                -struct.get_stress(voigt=False) * struct.get_volume()
            )
            struct.calc = None
        write(file_prefix, configs)

    eval_file_prefix = out_dir
    test = read(eval_file_prefix / f"output_{model_name}_D3.xyz", ":")
    single_molecule_test = []
    for molsym in ["EMC", "EC", "PF6", "Li"]:
        single_molecule_test += read(
            eval_file_prefix / f"output{molsym}_{model_name}_D3.xyz", ":"
        )
    anaAtoms.collect_molec_results_dict(test, single_molecule_test)
    write(eval_file_prefix / f"intrainter_{model_name}_D3.xyz", test)
