"""Calculate plutonium dioxide metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

EV_TO_KJ_PER_MOL = units.mol / units.kJ


@pytest.mark.parametrize("mlip", MODELS.items())
def test_puo2_parity(mlip: tuple[str, Any]) -> None:
    """
    Generate data for MAE analysis and density plots.

    Parameters
    ----------
    mlip : tuple[str, Any]
        A tuple of (model_name, model_object).
    """
    model_name, model = mlip

    # puo2_data_dir = (
    # Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    # / "puo2_data")

    # Download data.
    puo2_data_dir = (
        download_s3_data(
            key="inputs/actinides/plutonium_dioxide/puo2.zip",
            filename="puo2_data.zip",
        )
        / "puo2_data"
    )

    ref_file = puo2_data_dir / "dft_ref_data.xyz"
    ref_structures = read(ref_file, ":")

    results_to_save = []

    for atoms in ref_structures:
        atoms.calc = (
            model.get_calculator()
        )  # must be called each time as number of atoms changes.
        atoms.get_potential_energy()
        atoms.get_forces()
        atoms.get_stress()

        results_to_save.append(atoms)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / "puo2_results.xyz", results_to_save)
