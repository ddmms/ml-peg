"""
Compute carbon binding curves: energy vs nearest-neighbour distance.

Six carbon structures (isolated dimer, graphene, diamond, simple cubic, BCC and
FCC) are scanned over a common C-C nearest-neighbour distance grid. Energies are
compared against digitised PBE+D2 reference curves in the analysis stage.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.build import bulk, graphene
from ase.io import write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

MIN_NN = 0.5
MAX_NN = 8.0
STEP_NN = 0.05
VACUUM = 15.0

STRUCTURE_NAMES = ("dimer", "graphene", "diamond", "sc", "bcc", "fcc")


def bond_lengths() -> list[float]:
    """
    Return the nearest-neighbour distance grid.

    Returns
    -------
    list[float]
        Monotonic list of C-C nearest-neighbour distances in Angstrom.
    """
    n = int(round((MAX_NN - MIN_NN) / STEP_NN))
    return [round(MIN_NN + i * STEP_NN, 10) for i in range(n + 1)]


def structures(nn: float) -> dict[str, Atoms]:
    """
    Build the six carbon structures at a given nearest-neighbour distance.

    Parameters
    ----------
    nn
        C-C nearest-neighbour distance in Angstrom.

    Returns
    -------
    dict[str, Atoms]
        Mapping of structure name to ASE ``Atoms`` object.
    """
    dimer = Atoms("C2", positions=[[0.0, 0.0, 0.0], [nn, 0.0, 0.0]], pbc=False)
    dimer.center(vacuum=VACUUM)

    graph = graphene(formula="C2", a=math.sqrt(3.0) * nn, size=(1, 1, 1), vacuum=VACUUM)
    graph.pbc = (True, True, False)

    return {
        "dimer": dimer,
        "graphene": graph,
        "diamond": bulk("C", "diamond", a=4.0 * nn / math.sqrt(3.0)),
        "sc": bulk("C", "sc", a=nn),
        "bcc": bulk("C", "bcc", a=2.0 * nn / math.sqrt(3.0)),
        "fcc": bulk("C", "fcc", a=math.sqrt(2.0) * nn),
    }


def run_binding_curves(model_name: str, model: Any) -> None:
    """
    Evaluate carbon binding curves for a single model.

    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    model
        Model wrapper providing ``get_calculator`` and ``add_d3_calculator``.
    """
    data_dir = download_s3_data(
        key="inputs/carbon/binding_curves/binding_curves.zip",
        filename="binding_curves.zip",
    )
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    ref_src = data_dir / "binding_curves" / "reference.json"
    (OUT_PATH / "reference.json").write_bytes(ref_src.read_bytes())

    calc = model.get_calculator(precision="high")
    calc = model.add_d3_calculator(calc)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, list[Atoms]] = {name: [] for name in STRUCTURE_NAMES}
    for nn in tqdm(bond_lengths(), desc=f"{model_name} binding curves"):
        for name, atoms in structures(nn).items():
            atoms.calc = calc
            try:
                energy = float(atoms.get_potential_energy()) / len(atoms)
            except Exception as exc:
                warn(
                    f"Error calculating energy for {name} at nn={nn}: {exc}",
                    stacklevel=2,
                )
                energy = np.nan
            snapshot = atoms.copy()
            snapshot.calc = None
            snapshot.info.update(
                {
                    "structure": name,
                    "nn_distance": nn,
                    "energy_per_atom": energy,
                }
            )
            frames[name].append(snapshot)

    for name, structure_frames in frames.items():
        write(write_dir / f"{name}.xyz", structure_frames, format="extxyz")


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_binding_curves(mlip: tuple[str, Any]) -> None:
    """
    Run the carbon binding-curve benchmark for each registered model.

    Parameters
    ----------
    mlip
        Tuple of model name and model wrapper.
    """
    model_name, model = mlip
    run_binding_curves(model_name, model)
