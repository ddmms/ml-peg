"""
Calculate the glucose dataset for pyrolysis.

Zhao, Qiyuan, and Brett M. Savoie.
Proceedings of the National Academy of Sciences
120.34 (2023): e2305884120
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.fast
@pytest.mark.framework("mace-polar-1")
@pytest.mark.parametrize("mlip", MODELS.items())
def test_glucose(mlip: tuple[str, Any]) -> None:
    """
    Run glucose benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    try:
        calc = model.get_calculator(precision="high")
    except Exception:
        print(f"{model_name} is not available in your results.")
        return
    if hasattr(model, "add_d3_calculator"):
        calc = model.add_d3_calculator(calc)

    out_path = OUT_PATH / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    data_file = Path(__file__).parent / "glucose_dataset.xyz"

    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} is missing.")

    atom_list = read(data_file, index=":")
    print(f"{model_name} is calculating glucose database....")

    for atoms in tqdm(atom_list, desc=f"calculating with {model_name}"):
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc

        atoms.info["model_energy"] = float(atoms.get_potential_energy())

        atoms.calc = None

    output_xyz = out_path / "glucose_calculated.xyz"
    write(output_xyz, atom_list, format="extxyz")

    return
