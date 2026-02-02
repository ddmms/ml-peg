"""
Run MD simulation of BMIMCl ionic liquid to test for unphysical Cl-C bond formation.

This benchmark tests whether MLIPs incorrectly predict covalent bond formation
between chloride anions and carbon atoms in BMIM cations. Such bonds should NOT
form in ionic liquids at normal conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
import molify
import pytest
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# MD parameters
# Density from https://pubs.acs.org/doi/full/10.1021/acs.jced.7b00654 at 353.15 K
TEMPERATURE = 353.15
STEPS = 10_000
TIMESTEP = 0.5  # fs
FRICTION = 0.01  # 1/fs
N_ION_PAIRS = 10
DENSITY = 1052  # kg/m³ at 353.15 K


@pytest.mark.parametrize("mlip", MODELS.items())
def test_bmimcl_md(mlip: tuple[str, Any]) -> None:
    """
    Run NVT MD simulation of BMIMCl ionic liquid.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    bmim = molify.smiles2atoms("CCCCN1C=C[N+](=C1)C")
    cl = molify.smiles2atoms("[Cl-]")
    ion_pair = molify.pack(data=[[bmim], [cl]], counts=[1, 1], density=900)
    box = molify.pack(data=[[ion_pair]], counts=[N_ION_PAIRS], density=DENSITY)
    box.calc = calc

    # Optimize structure
    opt = LBFGS(box)
    opt.run(fmax=0.1)

    # Initialize velocities
    MaxwellBoltzmannDistribution(box, temperature_K=TEMPERATURE)

    # MD setup
    dyn = Langevin(
        box,
        timestep=TIMESTEP * units.fs,
        temperature_K=TEMPERATURE,
        friction=FRICTION / units.fs,
    )

    # Output directory
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    traj_file = write_dir / "md.xyz"

    # Remove existing trajectory file if present
    if traj_file.exists():
        traj_file.unlink()

    # Run MD and save trajectory
    for _ in tqdm(range(STEPS), desc=f"{model_name} MD"):
        dyn.run(1)
        write(traj_file, box, append=True)
