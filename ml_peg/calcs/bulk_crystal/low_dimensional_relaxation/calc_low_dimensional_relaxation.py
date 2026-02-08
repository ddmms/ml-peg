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
from janus_core.calculations.geom_opt import GeomOpt
import numpy as np
import pandas as pd
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
import pytest
import spglib
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# URLs for Alexandria database
ALEXANDRIA_URLS = {
    "2D": "https://alexandria.icams.rub.de/data/pbe_2d",
    "1D": "https://alexandria.icams.rub.de/data/pbe_1d",
}

# Default data files for each dimensionality (list of files)
DEFAULT_DATA_FILES: dict[str, list[str]] = {
    "2D": ["alexandria_2d_001.json.bz2", "alexandria_2d_000.json.bz2"],
    "1D": ["alexandria_1d_000.json.bz2"],
}

# Local cache directory for downloaded data
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Relaxation parameters
FMAX = 0.0002  # eV/A - tight convergence
MAX_STEPS = 500
RANDOM_SEED = 42

# Number of structures to use for testing
N_STRUCTURES = 3000

# Vacuum padding for non-periodic directions (Angstrom)
VACUUM_PADDING = 100.0

# Dimensionality configurations
# For 2D: mask allows in-plane cell relaxation only (fix z)
# For 1D: mask allows along-chain relaxation only (fix y and z)
CELL_MASKS = {
    "2D": np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    "1D": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool),
}


def download_data(dimensionality: str, filename: str) -> Path:
    """
    Download a single structure data file from Alexandria database if not cached.

    Parameters
    ----------
    dimensionality
        Either "2D" or "1D".
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
        url = f"{ALEXANDRIA_URLS[dimensionality]}/{filename}"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")

    return local_path


def download_all_data(
    dimensionality: str, filenames: list[str] | None = None
) -> list[Path]:
    """
    Download all structure data files for a dimensionality.

    Parameters
    ----------
    dimensionality
        Either "2D" or "1D".
    filenames
        List of filenames to download. If None, uses defaults for dimensionality.

    Returns
    -------
    list[Path]
        Paths to the downloaded/cached files.
    """
    if filenames is None:
        filenames = DEFAULT_DATA_FILES[dimensionality]

    return [download_data(dimensionality, f) for f in filenames]


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


def get_length_per_atom(atoms: Atoms) -> float:
    """
    Calculate the chain length per atom for 1D structures.

    For 1D materials, the length is the magnitude of the first lattice vector.

    Parameters
    ----------
    atoms
        ASE Atoms object.

    Returns
    -------
    float
        Chain length per atom in Å.
    """
    cell = atoms.get_cell()
    length = np.linalg.norm(cell[0])
    return length / len(atoms)


def symmetrize_structure(atoms: Atoms, symprec: float = 1e-5) -> Atoms:
    """
    Symmetrize atomic positions and cell using spglib.

    Parameters
    ----------
    atoms
        ASE Atoms object.
    symprec
        Symmetry precision for spglib.

    Returns
    -------
    Atoms
        Symmetrized atoms object.
    """
    # Convert to spglib format
    cell = (
        atoms.get_cell().array,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )

    # Get symmetrized cell
    symmetrized = spglib.standardize_cell(cell, to_primitive=False, symprec=symprec)

    if symmetrized is None:
        # If symmetrization fails, return original structure
        return atoms

    lattice, scaled_positions, numbers = symmetrized

    # Create new atoms object with symmetrized structure
    return Atoms(
        numbers=numbers,
        scaled_positions=scaled_positions,
        cell=lattice,
        pbc=atoms.pbc,
    )


def set_vacuum_padding(
    atoms: Atoms, dimensionality: str, vacuum: float = VACUUM_PADDING
) -> Atoms:
    """
    Set lattice constants in non-periodic directions to specified vacuum padding.

    For 2D materials, sets the c lattice vector magnitude to the vacuum value.
    For 1D materials, sets both b and c lattice vector magnitudes to the vacuum value.
    The direction of each lattice vector is preserved.

    Parameters
    ----------
    atoms
        ASE Atoms object.
    dimensionality
        Either "2D" or "1D".
    vacuum
        Target length for non-periodic lattice vectors in Angstrom.

    Returns
    -------
    Atoms
        Modified atoms object with updated cell.
    """
    cell = np.array(atoms.get_cell())

    if dimensionality == "2D":
        # Set c vector magnitude to vacuum while preserving direction
        c_vec = cell[2]
        c_norm = np.linalg.norm(c_vec)
        if c_norm > 0:
            cell[2] = c_vec / c_norm * vacuum
        else:
            # If c vector is zero, set it along z direction
            cell[2] = np.array([0.0, 0.0, vacuum])
    else:  # 1D
        # Set b vector magnitude to vacuum while preserving direction
        b_vec = cell[1]
        b_norm = np.linalg.norm(b_vec)
        if b_norm > 0:
            cell[1] = b_vec / b_norm * vacuum
        else:
            cell[1] = np.array([0.0, vacuum, 0.0])

        # Set c vector magnitude to vacuum while preserving direction
        c_vec = cell[2]
        c_norm = np.linalg.norm(c_vec)
        if c_norm > 0:
            cell[2] = c_vec / c_norm * vacuum
        else:
            cell[2] = np.array([0.0, 0.0, vacuum])

    atoms.set_cell(cell, scale_atoms=False)
    return atoms


def load_structures(
    dimensionality: str = "2D",
    filenames: list[str] | None = None,
    n_structures: int = N_STRUCTURES,
) -> list[dict]:
    """
    Load structures from Alexandria database.

    Downloads data if not already cached locally. When multiple files are provided,
    structures are pooled together and randomly sampled from the combined set.

    Parameters
    ----------
    dimensionality
        Either "2D" or "1D".
    filenames
        List of data filenames to load. If None, uses defaults for dimensionality.
    n_structures
        Number of structures to load. Default is N_STRUCTURES.

    Returns
    -------
    list[dict]
        List of structure dictionaries with atoms, mat_id, and reference values.
    """
    json_paths = download_all_data(dimensionality, filenames)

    # Load all entries from all files
    all_entries = []
    for json_path in json_paths:
        with bz2.open(json_path, "rt") as f:
            data = json.load(f)
        all_entries.extend(data["entries"])

    if n_structures > len(all_entries):
        n_structures = len(all_entries)
        print(f"Only {len(all_entries)} structures available, using all of them")

    rng = random.Random(RANDOM_SEED)
    selected_indices = sorted(rng.sample(range(len(all_entries)), k=n_structures))

    adaptor = AseAtomsAdaptor()
    structures = []

    for idx in selected_indices:
        entry_dict = all_entries[idx]
        entry = ComputedStructureEntry.from_dict(entry_dict)
        mat_id = entry.data.get("mat_id", f"struct_{idx}")

        atoms = adaptor.get_atoms(entry.structure)
        n_atoms = len(entry.structure)

        struct_data = {
            "mat_id": mat_id,
            "atoms": atoms,
            "ref_energy_per_atom": entry.energy / n_atoms,
        }

        # Use appropriate geometric metric based on dimensionality
        if dimensionality == "2D":
            struct_data["ref_area_per_atom"] = get_area_per_atom(atoms)
        else:  # 1D
            struct_data["ref_length_per_atom"] = get_length_per_atom(atoms)

        structures.append(struct_data)

    return structures


def relax_low_dimensional(
    atoms: Atoms,
    dimensionality: str = "2D",
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
) -> tuple[Atoms | None, bool, float | None]:
    """
    Relax low-dimensional structure with cell mask to constrain relaxation.

    Parameters
    ----------
    atoms
        ASE Atoms object with calculator attached.
    dimensionality
        Either "2D" or "1D".
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
            geom_opt = GeomOpt(
                struct=atoms,
                fmax=fmax,
                steps=max_steps,
                # Use cell mask to constrain relaxation to appropriate dimensions
                filter_kwargs={"mask": CELL_MASKS[dimensionality]},
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


DIMENSIONALITIES = ["2D", "1D"]


@pytest.mark.parametrize("mlip", MODELS.items())
@pytest.mark.parametrize("dimensionality", DIMENSIONALITIES)
def test_low_dimensional_relaxation(mlip: tuple[str, Any], dimensionality: str) -> None:
    """
    Run low-dimensional relaxation benchmark.

    Parameters
    ----------
    mlip
        Tuple of model name and model object.
    dimensionality
        Either "2D" or "1D".
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Load structures (downloads from Alexandria if not cached)
    structures = load_structures(dimensionality)

    results = []
    for struct_data in tqdm(
        structures, desc=f"{model_name} {dimensionality}", leave=False
    ):
        atoms = struct_data["atoms"].copy()
        # Symmetrize structure before relaxation
        atoms = symmetrize_structure(atoms)
        # Set vacuum padding in non-periodic directions
        atoms = set_vacuum_padding(atoms, dimensionality)
        atoms.calc = copy(calc)
        mat_id = struct_data["mat_id"]

        relaxed_atoms, converged, energy_per_atom = relax_low_dimensional(
            atoms, dimensionality
        )

        result = {
            "mat_id": mat_id,
            "dimensionality": dimensionality,
            "ref_energy_per_atom": struct_data["ref_energy_per_atom"],
            "pred_energy_per_atom": energy_per_atom,
            "converged": converged,
        }

        # Add dimension-specific geometric metrics
        if dimensionality == "2D":
            result["ref_area_per_atom"] = struct_data["ref_area_per_atom"]
            if relaxed_atoms is not None:
                result["pred_area_per_atom"] = get_area_per_atom(relaxed_atoms)
            else:
                result["pred_area_per_atom"] = None
        else:  # 1D
            result["ref_length_per_atom"] = struct_data["ref_length_per_atom"]
            if relaxed_atoms is not None:
                result["pred_length_per_atom"] = get_length_per_atom(relaxed_atoms)
            else:
                result["pred_length_per_atom"] = None

        results.append(result)

    # Save results
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"results_{dimensionality}.csv", index=False)
