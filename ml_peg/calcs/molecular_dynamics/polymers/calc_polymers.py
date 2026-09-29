"""Run the 21-step polymer-density benchmark for each MLIP × polymer system."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import typing as ty

import ase
import ase.io
import pandas as pd
import pytest

from ml_peg.calcs.molecular_dynamics.polymers import protocol
from ml_peg.calcs.utils import utils as calc_utils
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = pathlib.Path(__file__).parent / "outputs"
REFERENCE_PATH = OUT_PATH / "reference"

REFERENCE_TEMP_K: ty.Final[float] = 300.0
REFERENCE_PRESSURE_ATM: ty.Final[float] = 1.0
DEFAULT_SEED: ty.Final[int] = 42


def _load_polymer_table() -> pd.DataFrame:
    """
    Load and sort the polymer reference table indexed by polymer id.

    Returns
    -------
    pd.DataFrame
        The polymer table indexed by ``id``, sorted alphabetically.
    """
    df = pd.read_csv(
        pathlib.Path(__file__).parent / "resources" / "data.csv",
        na_values=["NaN"],
        encoding="utf-8",
        comment="%",
    )
    return df.set_index("id").sort_index()


POLYMER_TABLE = _load_polymer_table()


def _copy_reference_structures(input_dir: pathlib.Path) -> None:
    """
    Copy every input structure into the shared reference directory.

    Each file is written via a process-unique temporary name and then moved into
    place, so concurrent jobs cannot leave a partial file behind.

    Parameters
    ----------
    input_dir
        Downloaded directory of input structures.
    """
    REFERENCE_PATH.mkdir(parents=True, exist_ok=True)
    for input_xyz_path in input_dir.glob("*.xyz"):
        reference_xyz = REFERENCE_PATH / input_xyz_path.name
        if reference_xyz.exists():
            continue
        tmp = reference_xyz.with_suffix(f".xyz.{os.getpid()}.tmp")
        shutil.copyfile(input_xyz_path, tmp)
        tmp.replace(reference_xyz)


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_polymer_densities(
    mlip: tuple[str, ty.Any],
    poly_id: str,
    time_prefactor: float,
) -> None:
    """
    Run the 21-step polymer protocol for one (model, polymer) combination.

    Parameters
    ----------
    mlip
        Pair ``(model_name, model)`` produced by the parametrization.
    poly_id
        Polymer identifier (e.g. ``"PS"``); must be present as an ``id``
        in ``data.csv``.
    time_prefactor
        Scale factor applied to every stage duration (default 1.0). Use a
        small value (e.g. 0.05) for end-to-end smoke tests.
    """
    if poly_id not in POLYMER_TABLE.index:
        raise pytest.UsageError(
            f"Unknown poly_id '{poly_id}'; see resources/data.csv for valid ids"
        )

    s3_dir = calc_utils.download_s3_data(
        filename="polymers.zip",
        key="inputs/molecular_dynamics/polymers/polymers.zip",
    )
    input_xyz_path = s3_dir / "polymers" / f"{poly_id}.xyz"
    _copy_reference_structures(s3_dir / "polymers")

    model_name, model = mlip
    calc = model.get_calculator(precision="low")
    calc = model.add_d3_calculator(calc)

    out_dir = OUT_PATH / model_name / poly_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=out_dir / f"{poly_id}.log",
        filemode="a",
        force=True,
    )

    atoms = ase.io.read(str(input_xyz_path))
    assert isinstance(atoms, ase.Atoms), (
        f"Expected single Atoms object from {input_xyz_path}"
    )
    # Defaults for models that require these fields; existing values from the
    # extxyz (set during generation) are preserved.
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)

    try:
        protocol.run_polymer_protocol(
            atoms,
            calc,
            out_dir=out_dir,
            temp_final_k=REFERENCE_TEMP_K,
            p_final_atm=REFERENCE_PRESSURE_ATM,
            time_prefactor=time_prefactor,
            seed=DEFAULT_SEED,
        )
    except Exception as err:  # noqa: BLE001 - we want to catch any model failure
        logging.exception(f"Polymer {poly_id} with {model_name} failed: {err}")
        pytest.skip(f"{model_name} failed on {poly_id}: {err}")
