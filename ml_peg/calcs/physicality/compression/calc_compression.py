"""Run calculations for uniform crystal compression benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.build import bulk, make_supercell
from ase.data import covalent_radii, atomic_numbers
from ase.io import write
import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory for calculator outputs
OUT_PATH = Path(__file__).parent / "outputs"
DATA_PATH = Path(__file__).parent / "data"

# Benchmark configuration
# Compression/expansion factors applied uniformly to the cell volume.
# A factor of 1.0 corresponds to the equilibrium structure.
MIN_SCALE = 0.25
MAX_SCALE = 3.0
N_POINTS = 20

#ELEMENTS: list[str] = [symbol for symbol in chemical_symbols if symbol]
ELEMENTS: list[str] = ["H", "C", "N", "O"]  # limit to common elements for testing
PROTOTYPES: list[str] = ["sc", "bcc","fcc", "hcp", "diamond"]  # common crystal prototypes

def _scale_grid(
    min_scale: float,
    max_scale: float,
    n_points: int,
) -> np.ndarray:
    """
    Return a uniformly spaced grid of isotropic scaling factors.

    Each factor multiplies the *linear* cell dimensions so that the
    corresponding volume scales as ``factor**3``.

    Parameters
    ----------
    min_scale, max_scale
        Bounds of the linear scaling grid.
    n_points
        Number of points in the grid.

    Returns
    -------
    np.ndarray
        Monotonic array of scaling factors.
    """
    return np.linspace(min_scale, max_scale, n_points, dtype=float)


def _generate_prototype_structures(
    elements: list[str],
    prototypes: list[str],
) -> dict[str, Atoms]:
    """
    Generate crystal structures from element/prototype combinations using ASE.

    For each element and crystal prototype, ``ase.build.bulk`` is called
    with a lattice constant of 1.0.  The resulting structure is then
    rescaled so that the shortest interatomic distance equals twice the
    covalent radius of the element (from ``ase.data.covalent_radii``).
    Combinations that ASE cannot build are silently skipped.

    Parameters
    ----------
    elements
        Chemical symbols to build structures for.
    prototypes
        Crystal-structure prototypes recognised by ``ase.build.bulk``
        (e.g. ``"fcc"``, ``"bcc"``, ``"hcp"``, ``"diamond"``, ``"sc"``).

    Returns
    -------
    dict[str, Atoms]
        Mapping of ``"{element}_{prototype}"`` labels to ASE ``Atoms``.
    """
    structures: dict[str, Atoms] = {}
    for element in elements:
        z = atomic_numbers[element]
        target_bond = 2.0 * covalent_radii[z]
        for proto in prototypes:
            label = f"{element}_{proto}"
            #try:
            atoms = bulk(element, crystalstructure=proto, a=1.0)

            # Find shortest interatomic distance at a=1.0
            sc = make_supercell(atoms, 2 * np.eye(3))  # ensure we have a full unit cell for distance calculation
            distances = sc.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            min_dist = float(distances.min())

            if min_dist <= 0.0:
                print(f"Skipping {label}: degenerate geometry at a=1.0")
                continue

            # Scale so shortest distance equals the covalent bond length
            scale_factor = target_bond / min_dist
            atoms.set_cell(atoms.cell * scale_factor, scale_atoms=True)
            print("Atoms is", label, atoms)
            assert np.all(atoms.pbc), f"Generated non-periodic structure for {label}"
            assert atoms.get_volume() > 0.0, f"Generated zero-volume structure for {label}"

            structures[label] = atoms
            # except (ValueError, KeyError, RuntimeError) as exc:
            #     # Not every element/prototype combination is supported
            #     print(f"Skipping {label}: {exc}")
            #     continue
    return structures


def _load_structures(data_path: Path) -> dict[str, Atoms]:
    """
    Load reference crystal structures from the data directory.

    The directory is expected to contain individual ``.xyz`` or
    ``.extxyz`` files, each holding a single periodic structure with a
    cell defined.

    Parameters
    ----------
    data_path
        Directory containing reference crystal structure files.

    Returns
    -------
    dict[str, Atoms]
        Mapping of structure label (stem of filename) to ASE ``Atoms``.
    """
    from ase.io import read

    structures: dict[str, Atoms] = {}
    for ext in ("*.xyz", "*.extxyz", "*.cif"):
        for filepath in sorted(data_path.glob(ext)):
            atoms = read(filepath)
            if not np.any(atoms.pbc):
                atoms.pbc = True
            structures[filepath.stem] = atoms
    return structures


def _collect_structures(
    elements: list[str],
    prototypes: list[str],
    data_path: Path,
) -> dict[str, Atoms]:
    """
    Build the full set of structures to benchmark.

    Combines prototype-generated structures (one per element/prototype
    combination produced by ``ase.build.bulk``) with any additional
    structures loaded from files in ``data_path``.

    Parameters
    ----------
    elements
        Chemical symbols for prototype generation.
    prototypes
        Crystal prototypes for ``ase.build.bulk``.
    data_path
        Directory containing additional reference structure files.

    Returns
    -------
    dict[str, Atoms]
        Combined mapping of structure labels to ASE ``Atoms``.
        Prototype labels have the form ``"{element}_{prototype}"``;
        file-loaded labels use the filename stem.  If a file label
        collides with a prototype label, the file-loaded structure
        takes precedence.
    """
    structures = _generate_prototype_structures(elements, prototypes)
    file_structures = _load_structures(data_path)
    # File-loaded structures override any prototype with the same label
    structures.update(file_structures)
    return structures


def run_compression(model_name: str, model) -> None:
    """
    Evaluate energy-volume compression curves for a single model.

    For every reference crystal structure, the cell is uniformly scaled
    over a range of linear factors.  At each scale point the energy,
    volume (per atom), and stress tensor are recorded.

    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    model
        Model wrapper providing ``get_calculator``.
    """
    calc = model.get_calculator()
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = write_dir / "compression"
    traj_dir.mkdir(parents=True, exist_ok=True)
    

    reference_structures = _collect_structures(ELEMENTS, PROTOTYPES, DATA_PATH)
    if not reference_structures:
        print(f"[{model_name}] No structures generated or found in {DATA_PATH}")
        return

    records: list[dict[str, float | str]] = []
    supported_structures: set[str] = set()
    failed_structures: dict[str, str] = {}

    scales = _scale_grid(MIN_SCALE, MAX_SCALE, N_POINTS)

    for struct_label, ref_atoms in tqdm(
        reference_structures.items(),
        desc=f"{model_name} compression",
        unit="struct",
    ):
        try:
            ref_cell = ref_atoms.cell.copy()
            ref_positions = ref_atoms.get_scaled_positions()
            n_atoms = len(ref_atoms)
            trajectory: list[Atoms] = []

            for scale in scales:
                atoms = ref_atoms.copy()
                # Uniformly scale the cell (isotropic compression/expansion)
                atoms.set_cell(ref_cell * scale, scale_atoms=True)
                atoms.set_scaled_positions(ref_positions)

                atoms.info.setdefault("charge", 0)
                atoms.info.setdefault("spin", 1)

                atoms.calc = calc

                energy = float(atoms.get_potential_energy())
                volume = float(atoms.get_volume())
                volume_per_atom = volume / n_atoms
                energy_per_atom = energy / n_atoms

                # Stress tensor (Voigt notation, eV/Å³)
                try:
                    stress_voigt = atoms.get_stress(voigt=False)
                    pressure = -1/3 * np.trace(stress_voigt)  # hydrostatic pressure
                except Exception:
                    stress_voigt = np.zeros(6)
                    pressure = np.nan

                # Store trajectory snapshot
                atoms_copy = atoms.copy()
                atoms_copy.calc = None
                atoms_copy.info.update(
                    {
                        "structure": struct_label,
                        "scale": float(scale),
                        "energy": energy,
                        "energy_per_atom": energy_per_atom,
                        "volume": volume,
                        "volume_per_atom": volume_per_atom,
                        "pressure": pressure,
                        "model": model_name,
                    }
                )
                trajectory.append(atoms_copy)

                records.append(
                    {
                        "structure": struct_label,
                        "scale": float(scale),
                        "energy": energy,
                        "energy_per_atom": energy_per_atom,
                        "volume": volume,
                        "volume_per_atom": volume_per_atom,
                        "pressure": pressure,
                    }
                )

            if trajectory:
                write(
                    traj_dir / f"{struct_label}.xyz",
                    trajectory,
                    format="extxyz",
                )
                supported_structures.add(struct_label)

        except Exception as exc:
            failed_structures[struct_label] = str(exc)
            print(f"[{model_name}] Skipping {struct_label}: {exc}")
            continue

    if records:
        df = pd.DataFrame.from_records(records)
        df.to_csv(write_dir / "compression.csv", index=False)

    metadata = {
        "supported_structures": sorted(supported_structures),
        "failed_structures": failed_structures,
        "config": {
            "min_scale": MIN_SCALE,
            "max_scale": MAX_SCALE,
            "n_points": N_POINTS,
        },
    }
    (write_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_compression(model_name: str) -> None:
    """
    Run compression benchmark for each registered model.

    Parameters
    ----------
    model_name
        Name of the model to evaluate.
    """
    run_compression(model_name, MODELS[model_name])
