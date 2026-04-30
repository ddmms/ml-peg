"""Run calculations for uniform crystal compression benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms, units
from ase.build import bulk, make_supercell
from ase.data import atomic_numbers, covalent_radii, chemical_symbols
from ase.io import read as ase_read
from ase.io import write
import numpy as np
import pandas as pd
import pytest
from pyxtal.tolerance import Tol_matrix
from pyxtal.database.element import Element
import tqdm

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
N_POINTS = 100

#filter out element where pyxtal doesn't have a covalent radius
ELEMENTS: list[str] = [e for e in chemical_symbols if Element(e).covalent_radius is not None ]

PROTOTYPES: list[str] = [
    "sc",
    "bcc",
    "fcc",
    "hcp",
    "diamond",
]  # common crystal prototypes
MAX_ATOMS_PER_CELL = 10  # limit to small cells for testing
RANDOM_STRUCTURES: list[dict[str, int]] = [
    [1, len(ELEMENTS), 5],
    [2, 20, 10],
    [3, 20, 10],
]  # [[Number of elements, number of compositions, repeats per composition], ...]


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


def _scale_using_isotropic_guess(atoms: Atoms) -> Atoms:
    """
    Isotropically scale a structure so nearest-neighbour distances match covalent radii.

    For each pair of atoms the target bond length is the sum of their
    covalent radii (from ``ase.data.covalent_radii``).  The structure is
    uniformly rescaled so that the shortest interatomic distance equals
    the smallest such target across all pairs.

    Parameters
    ----------
    atoms
        Input structure (must be periodic with ``pbc=True``).

    Returns
    -------
    Atoms
        A copy of *atoms* with the cell and positions rescaled.

    Raises
    ------
    ValueError
        If the shortest interatomic distance at the current cell is zero
        or negative (degenerate geometry).
    """
    atoms = atoms.copy()
    sc = make_supercell(atoms, 2 * np.eye(3))
    distances = sc.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)

    # Build a matrix of target bond lengths (sum of covalent radii per pair)
    radii = np.array(
        [covalent_radii[atomic_numbers[s]] for s in sc.get_chemical_symbols()]
    )
    target_matrix = radii[:, None] + radii[None, :]

    # Mask infinite entries (self-distances)
    mask = np.isfinite(distances)
    if not np.any(mask):
        raise ValueError("No finite interatomic distances found")

    # Find the pair with the smallest distance and its corresponding target
    ratios = np.full_like(distances, np.inf)
    ratios[mask] = target_matrix[mask] / distances[mask]

    # The required scale factor is the ratio for the closest pair
    closest_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_dist = distances[closest_idx]
    target_bond = target_matrix[closest_idx]

    if min_dist <= 0.0:
        raise ValueError("Degenerate geometry: shortest interatomic distance <= 0")

    scale_factor = target_bond / min_dist
    atoms.set_cell(atoms.cell * scale_factor, scale_atoms=True)
    return atoms


def _generate_prototype_structures(
    elements: list[str],
    prototypes: list[str],
) -> dict[str, Atoms]:
    """
    Generate crystal structures from element/prototype combinations using ASE.

    For each element and crystal prototype, ``ase.build.bulk`` is called
    with a lattice constant of 1.0.  The resulting structure is then
    rescaled via ``_scale_using_isotropic_guess`` so that the shortest
    interatomic distance matches the sum of covalent radii for the
    closest pair.  Combinations that ASE cannot build are silently
    skipped.

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
        for proto in prototypes:
            label = f"{element}_{proto}"
            try:
                atoms = bulk(element, crystalstructure=proto, a=1.0)
                atoms = _scale_using_isotropic_guess(atoms)

                assert np.all(atoms.pbc), (
                    f"Generated non-periodic structure for {label}"
                )
                assert atoms.get_volume() > 0.0, (
                    f"Generated zero-volume structure for {label}"
                )

                structures[label] = atoms
            except (ValueError, KeyError, RuntimeError) as exc:
                print(f"Skipping {label}: {exc}")
                continue
    return structures


def _gen_random_structure(
    composition: dict[str, int],
    space_group: int | None = None,
    dim: int = 3,
    seed: int | None = None,
    max_attempts: int = 230,
    volume_factor: float = 0.75,
) -> Atoms:
    """
    Generate a random crystal structure using PyXtal.

    Parameters
    ----------
    composition
        Mapping of chemical symbol to number of atoms in the
        conventional cell (e.g. ``{"C": 12}`` or ``{"Ba": 1, "Ti": 1, "O": 3}``).
    space_group
        International space-group number (1–230).  When ``None`` (default),
        a space group is chosen at random.  If the chosen space group is
        incompatible with the composition, additional random space groups
        are tried until one succeeds.
    dim
        Dimensionality of the crystal (3 for bulk).  Default is 3.
    seed
        Random seed for reproducibility.  When ``None`` (default), a
        random seed is generated automatically.
    max_attempts
        Maximum number of space-group attempts before giving up.
    volume_factor
        Volume factor used when generating random structures.
        Smaller factors will results in denser structures.

    Returns
    -------
    Atoms
        ASE ``Atoms`` with periodic boundary conditions and a cell
        scaled so that nearest-neighbour distances match the sum of
        covalent radii.

    Raises
    ------
    RuntimeError
        If no valid structure could be generated within *max_attempts*.
    """
    from pyxtal import pyxtal as pyxtal_cls

    rng = np.random.default_rng(seed)

    species = list(composition.keys())
    num_atoms = list(composition.values())

    # If a specific space group was requested, try it first
    sg_candidates: list[int] = []
    if space_group is not None:
        # sg_candidates.append(space_group)
        sg_candidates = [
            space_group
        ] * max_attempts  # try the same one repeatedly to allow for
        # randomness in the structure generation

    # Fill remaining attempts with random space groups
    while len(sg_candidates) < max_attempts:
        sg_candidates.append(int(rng.integers(1, 231)))

    custom_tol = Tol_matrix(prototype="atomic", factor=1.0)
    for sg in sg_candidates:
        crystal = pyxtal_cls()
        crystal.from_random(
            dim=dim,
            group=sg,
            species=species,
            numIons=num_atoms,
            random_state=int(rng.integers(0, 2**31)),
            tm=custom_tol,
            max_count=1000,
            factor=volume_factor,
        )
        if not crystal.valid:
            continue

        atoms = crystal.to_ase()
        atoms.pbc = True
        atoms = _scale_using_isotropic_guess(atoms)

        return atoms  # noqa: RET504
        # except Exception:
        #     continue

    raise RuntimeError(
        f"PyXtal failed to generate a valid structure for "
        f"composition={composition} after {max_attempts} attempts"
    )


def _composition_label(composition: dict[str, int]) -> str:
    """
    Build a canonical string label for a composition.

    Elements are sorted alphabetically and counts are appended only when
    greater than one (e.g. ``{"C": 1, "O": 2}`` → ``"C-O2"``).
    Elements are split by ``"-"`` to avoid confusion with the
    underscore used for separating the composition from the rest
    of the label (e.g. ``"C-O2_pyxtal_0"``).

    Parameters
    ----------
    composition
        Mapping of element symbol to atom count.

    Returns
    -------
    str
        Compact composition label.
    """
    parts: list[str] = []
    for elem in sorted(composition):
        count = composition[elem]
        parts.append(f"{elem}{count if count > 1 else ''}")
    return "-".join(parts)


def _generate_all_random(
    random_specs: list[list[int]],
    elements: list[str],
    max_atoms: int,
    data_path: Path,
    seed: int | None = None,
) -> dict[str, Atoms]:
    """
    Generate random crystal structures for multiple compositions.

    For each entry ``[n_elements, n_compositions, repeats]`` in
    *random_specs*, *n_compositions* unique compositions are sampled
    (drawing *n_elements* species from *elements* and random atom
    counts respecting *max_atoms*).  For every composition,
    ``_gen_random_structure`` is called *repeats* times, producing
    structures labelled ``"{composition}_RSS_{i}"``.

    Single-element structures are additionally saved to *data_path* so
    they can be reloaded later via ``_load_structures``.

    Parameters
    ----------
    random_specs
        List of ``[n_elements, n_compositions, repeats]`` triples.
    elements
        Pool of chemical symbols to draw from.
    max_atoms
        Maximum total number of atoms in any generated cell.
    data_path
        Directory where the configs ``.xyz`` files are written.
    seed
        Master random seed for reproducibility.

    Returns
    -------
    dict[str, Atoms]
        All successfully generated structures keyed by label.
    """
    rng = np.random.default_rng(seed)
    structures: dict[str, Atoms] = {}
    data_path.mkdir(parents=True, exist_ok=True)

    # Collect frames grouped by number of elements for saving
    frames_by_n_elements: dict[int, list[Atoms]] = {}

    for n_elements, n_compositions, repeats in random_specs:
        filepath = data_path / f"{n_elements}.xyz"
        if Path(filepath).exists():
            print(
                f"File {filepath} already exists — skipping generation of "
                f"{n_compositions} compositions with {n_elements} elements."
            )
            continue
        if n_elements > len(elements):
            print(
                f"Requested {n_elements} elements but only {len(elements)} "
                f"available — skipping this spec."
            )
            continue

        # Generate unique compositions
        generated_compositions: list[dict[str, int]] = []
        seen_labels: set[str] = set()
        max_comp_attempts = n_compositions * 50  # safety limit
        attempts = 0

        while (
            len(generated_compositions) < n_compositions
            and attempts < max_comp_attempts
        ):
            attempts += 1
            if n_elements == 1:
                # Pick the element which occurs least frequently
                # in the already generated compositions to
                # increase diversity.
                elem_counts = dict.fromkeys(elements, 0)
                for comp in generated_compositions:
                    for elem in comp:
                        elem_counts[elem] += comp[elem]
                least_common = sorted(elem_counts, key=elem_counts.get)
                chosen_elements = [least_common[0]]
            else:
                chosen_elements = sorted(
                    rng.choice(elements, size=n_elements, replace=False).tolist()
                )
            # Random atom counts that sum to at most max_atoms (each >= 1)
            counts = rng.integers(
                1, max(2, max_atoms - n_elements + 2), size=n_elements
            )
            if len(counts) == 1:
                counts = np.random.choice(
                    range(max_atoms // 2, max_atoms + 1), size=1
                )  # for single element, pick a random count
                # between max_atoms//2 and max_atoms

            # Clip total to max_atoms
            total = int(counts.sum())
            if total > max_atoms:
                counts = (counts * max_atoms / total).astype(int)
                counts = np.maximum(counts, 1)

            composition = dict(zip(chosen_elements, counts.tolist(), strict=True))
            label = _composition_label(composition)
            if label not in seen_labels:
                seen_labels.add(label)
                generated_compositions.append(composition)

        print(
            f"Generated {len(generated_compositions)} unique "
            f"compositions with {n_elements} elements: "
            f"{generated_compositions}"
        )
        # Generate structures for each composition
        for composition in tqdm.tqdm(generated_compositions):
            # comp_label = _composition_label(composition)
            for i in range(repeats):
                # struct_label = f"{comp_label}_pyxtal_{i}"
                try:
                    atoms = _gen_random_structure(
                        composition,
                        seed=int(rng.integers(0, 2**31)),
                        space_group=1,  # random space group
                    )

                    struct_label = f"{atoms.get_chemical_formula()}_pyxtal_{i}"
                    atoms.info["label"] = struct_label
                    structures[struct_label] = atoms

                    # Collect for saving grouped by number of elements
                    frames_by_n_elements.setdefault(n_elements, []).append(atoms)
                except Exception as exc:
                    print(f"Failed to generate {struct_label}: {exc}")
                    continue

    # Save structures to data_path as multi-frame xyz, grouped by element count
    for n_elem, atom_list in frames_by_n_elements.items():
        filepath = data_path / f"{n_elem}.xyz"
        write(filepath, atom_list, format="extxyz")
        print(f"Saved {len(atom_list)} structures to {filepath}")

    return structures


def _load_structures(data_path: Path) -> dict[str, Atoms]:
    """
    Load reference crystal structures from the data directory.

    Each ``.xyz`` / ``.extxyz`` file may contain multiple frames.  For
    single-frame files the label is the filename stem; for multi-frame
    files each frame is labelled ``"{stem}_{index}"``.

    Parameters
    ----------
    data_path
        Directory containing reference crystal structure files.

    Returns
    -------
    dict[str, Atoms]
        Mapping of structure label to ASE ``Atoms``.
    """
    structures: dict[str, Atoms] = {}
    for ext in ("*.xyz", "*.extxyz", "*.cif"):
        for filepath in sorted(data_path.glob(ext)):
            try:
                frames = ase_read(filepath, index=":")
            except Exception:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            stem = filepath.stem
            if len(frames) == 1:
                atoms = frames[0]
                # Prefer the stored label if present
                label = atoms.info.get("label", stem)
                if not np.any(atoms.pbc):
                    atoms.pbc = True
                structures[label] = atoms
            else:
                for idx, atoms in enumerate(frames):
                    label = atoms.info.get("label", f"{stem}_{idx}")
                    if not np.any(atoms.pbc):
                        atoms.pbc = True
                    structures[label] = atoms
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
    
    _generate_all_random(
        RANDOM_STRUCTURES,
        ELEMENTS,
        MAX_ATOMS_PER_CELL,
        DATA_PATH,
        seed=42,
    )

    reference_structures = _collect_structures(ELEMENTS, PROTOTYPES, DATA_PATH)
    if not reference_structures:
        print(f"[{model_name}] No structures generated or found in {DATA_PATH}")
        return

    records: list[dict[str, float | str]] = []
    supported_structures: set[str] = set()
    failed_structures: dict[str, str] = {}

    scales = _scale_grid(MIN_SCALE, MAX_SCALE, N_POINTS)

    for struct_label, ref_atoms in tqdm.tqdm(
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
                    pressure = (
                        -1 / 3 * np.trace(stress_voigt) / units.GPa
                    )  # hydrostatic pressure
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


@pytest.mark.very_slow
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


if __name__ == "__main__":
    # generate the random structures and save to data directory
    # this only needs to be done once
    # code included here so that the structures can be easily
    # regenerated in the same way if needed, or more can be
    # created etc.
    _generate_all_random(
        RANDOM_STRUCTURES,
        ELEMENTS,
        MAX_ATOMS_PER_CELL,
        DATA_PATH,
        seed=42,
    )
