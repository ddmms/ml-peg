"""
Compute the bilayer-graphene interlayer binding curve.

Bilayer graphene energy is scanned as a function of interlayer separation and
compared against digitised PBE+D2 reference values in the analysis stage. This
benchmark is dominated by dispersion, so the interlayer well depth and position
are sensitive probes of a model's long-range interactions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import write
from ase.lattice.hexagonal import Graphite
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

GRAPHENE_A = 2.46
MIN_SEPARATION = 2.0
MAX_SEPARATION = 8.0
STEP_SEPARATION = 0.1
VACUUM = 20.0


def separations() -> list[float]:
    """
    Return the interlayer-separation grid.

    Returns
    -------
    list[float]
        Monotonic list of interlayer separations in Angstrom.
    """
    n = int(round((MAX_SEPARATION - MIN_SEPARATION) / STEP_SEPARATION))
    return [round(MIN_SEPARATION + i * STEP_SEPARATION, 10) for i in range(n + 1)]


def bilayer_graphene(separation: float):
    """
    Build an AB-stacked bilayer graphene cell at a given interlayer separation.

    Parameters
    ----------
    separation
        Interlayer separation in Angstrom.

    Returns
    -------
    ase.Atoms
        Bilayer graphene structure with vacuum along the stacking axis.
    """
    atoms = Graphite(
        symbol="C",
        latticeconstant={"a": GRAPHENE_A, "c": 2.0 * separation},
        size=(1, 1, 1),
    )
    atoms.pbc = (True, True, False)
    atoms.cell[2, 2] = 2.0 * VACUUM + separation
    atoms.positions[:, 2] += VACUUM
    return atoms


def run_graphene_interlayer(model_name: str, model: Any) -> None:
    """
    Evaluate the bilayer-graphene interlayer curve for a single model.

    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    model
        Model wrapper providing ``get_calculator`` and ``add_d3_calculator``.
    """
    calc = model.get_calculator(precision="high")
    # Add dispersion corrections (skipped automatically for models already
    # trained with dispersion).
    calc = model.add_d3_calculator(calc)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory ordered by increasing interlayer separation; read back by the
    # analysis stage.
    frames = []
    for separation in tqdm(separations(), desc=f"{model_name} interlayer"):
        atoms = bilayer_graphene(separation)
        atoms.calc = calc
        try:
            energy = float(atoms.get_potential_energy()) / len(atoms)
        except Exception as exc:
            warn(
                f"Error calculating energy at separation={separation}: {exc}",
                stacklevel=2,
            )
            energy = np.nan
        snapshot = atoms.copy()
        snapshot.calc = None
        snapshot.info.update(
            {
                "interlayer_separation": separation,
                "energy_per_atom": energy,
            }
        )
        frames.append(snapshot)

    write(write_dir / "interlayer.extxyz", frames, format="extxyz")


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_graphene_interlayer(mlip: tuple[str, Any]) -> None:
    """
    Run the bilayer-graphene interlayer benchmark for each registered model.

    Parameters
    ----------
    mlip
        Tuple of model name and model wrapper.
    """
    model_name, model = mlip
    run_graphene_interlayer(model_name, model)
