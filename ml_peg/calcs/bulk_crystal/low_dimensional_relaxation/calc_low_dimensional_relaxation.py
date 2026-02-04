"""Run calculations for low-dimensional (2D/1D) crystal relaxation benchmark."""

from __future__ import annotations

import bz2
from copy import copy
import json
from pathlib import Path
import random
from typing import Any
import urllib.request

from ase import Atoms
from ase.constraints import FixSymmetry
from janus_core.calculations.geom_opt import GeomOpt
import numpy as np
import pandas as pd
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
import pytest
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# URLs for Alexandria database
ALEXANDRIA_2D_URL = "https://alexandria.icams.rub.de/data/pbe_2d"
ALEXANDRIA_1D_URL = "https://alexandria.icams.rub.de/data/pbe_1d"

# Local cache directory for downloaded data
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Relaxation parameters
FMAX = 0.0002  # eV/A - tight convergence
MAX_STEPS = 500
RANDOM_SEED = 42

# Number of structures to use for testing
N_STRUCTURES = 3000

# Dimensionality configurations
# For 2D: mask allows in-plane cell relaxation only (fix z)
# For 1D: mask allows along-chain relaxation only (fix y and z)
CELL_MASKS = {
    "2D": np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    "1D": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool),
}


def download_2d_data(filename: str = "alexandria_2d_001.json.bz2") -> Path:
    """
    Download 2D structure data from Alexandria database if not already cached.

    Parameters
    ----------
    filename
        Name of the file to download.

    Returns
    -------
    Path
        Path to the downloaded/cached file.
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    local_path = DATA_PATH / filename

    if not local_path.exists():
        url = f"{ALEXANDRIA_2D_URL}/{filename}"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")

    return local_path


def get_area_per_atom(atoms: Atoms) -> float:
    """
    Calculate the in-plane area per atom for 2D structures.

    For 2D materials, the area is calculated as a x b x sin(gamma),
    where a and b are the in-plane lattice vectors.

    Parameters
    ----------
    atoms
        ASE Atoms object.

    Returns
    -------
    float
        In-plane area per atom in Å².
    """
    cell = atoms.get_cell()
    # Calculate cross product of first two lattice vectors (in-plane)
    a = cell[0]
    b = cell[1]
    area = np.linalg.norm(np.cross(a, b))
    return area / len(atoms)


def load_2d_structures(
    filename: str = "alexandria_2d_001.json.bz2",
    n_structures: int = N_STRUCTURES,
) -> list[dict]:
    """
    Load 2D structures from Alexandria database.

    Downloads data if not already cached locally.

    Parameters
    ----------
    filename
        Name of the data file to load.
    n_structures
        Number of structures to load. Default is N_STRUCTURES.

    Returns
    -------
    list[dict]
        List of structure dictionaries with atoms, mat_id, and reference values.
    """
    json_path = download_2d_data(filename)

    with bz2.open(json_path, "rt") as f:
        data = json.load(f)

    adaptor = AseAtomsAdaptor()
    structures = []

    entries = data["entries"]
    if n_structures > len(entries):
        n_structures = len(entries)
        print(f"Only {len(entries)} structures available, using all of them")

    rng = random.Random(RANDOM_SEED)
    selected_indices = sorted(rng.sample(range(len(entries)), k=n_structures))

    for idx in selected_indices:
        entry_dict = entries[idx]
        entry = ComputedStructureEntry.from_dict(entry_dict)
        mat_id = entry.data.get("mat_id", f"struct_{idx}")

        atoms = adaptor.get_atoms(entry.structure)
        n_atoms = len(entry.structure)

        # For 2D, use area per atom instead of volume
        ref_area_per_atom = get_area_per_atom(atoms)

        structures.append(
            {
                "mat_id": mat_id,
                "atoms": atoms,
                "ref_energy_per_atom": entry.energy / n_atoms,
                "ref_area_per_atom": ref_area_per_atom,
            }
        )

    return structures


def relax_2d(
    atoms: Atoms,
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
) -> tuple[Atoms | None, bool, float | None]:
    """
    Relax 2D structure with cell mask to prevent out-of-plane relaxation.

    Parameters
    ----------
    atoms
        ASE Atoms object with calculator attached.
    fmax
        Maximum force tolerance in eV/A.
    max_steps
        Maximum number of optimization steps.

    Returns
    -------
    tuple[Atoms | None, bool, float | None]
        Relaxed atoms (or None if failed), convergence status, energy per atom.
    """
    try:
        converged = False
        counter = 0
        # repeat up to 3 times if not converged
        while counter < 3 and not converged:
            atoms.set_constraint(FixSymmetry(atoms))
            geom_opt = GeomOpt(
                struct=atoms,
                fmax=fmax,
                steps=max_steps,
                # Use cell mask to only allow in-plane relaxation
                filter_kwargs={"mask": CELL_MASKS["2D"]},
                calc_kwargs={"default_dtype": "float64"},
            )
            geom_opt.run()
            relaxed = geom_opt.struct
            # assess forces to determine convergence
            max_force = max(relaxed.get_forces().flatten(), key=abs)
            if max_force < fmax:
                converged = True
            counter += 1
            atoms = relaxed
    except Exception as e:
        print(f"Relaxation failed: {e}")
        return None, False, None

    if not converged:
        return None, False, None

    energy = relaxed.get_potential_energy()
    n_atoms = len(relaxed)

    return relaxed, converged, energy / n_atoms


@pytest.mark.parametrize("mlip", MODELS.items())
def test_low_dimensional_relaxation(mlip: tuple[str, Any]) -> None:
    """
    Run 2D relaxation benchmark.

    Parameters
    ----------
    mlip
        Tuple of model name and model object.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Load 2D structures (downloads from Alexandria if not cached)
    structures = load_2d_structures()

    results = []
    for struct_data in tqdm(structures, desc=f"{model_name} 2D", leave=False):
        atoms = struct_data["atoms"].copy()
        atoms.calc = copy(calc)
        mat_id = struct_data["mat_id"]

        relaxed_atoms, converged, energy_per_atom = relax_2d(atoms)

        if relaxed_atoms is not None:
            pred_area = get_area_per_atom(relaxed_atoms)
        else:
            pred_area = None

        results.append(
            {
                "mat_id": mat_id,
                "dimensionality": "2D",
                "ref_area_per_atom": struct_data["ref_area_per_atom"],
                "ref_energy_per_atom": struct_data["ref_energy_per_atom"],
                "pred_area_per_atom": pred_area,
                "pred_energy_per_atom": energy_per_atom,
                "converged": converged,
            }
        )

    # Save results
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results_2D.csv", index=False)
