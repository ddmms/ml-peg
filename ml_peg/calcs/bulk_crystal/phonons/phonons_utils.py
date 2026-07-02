"""Local phonon utilities — extracted from mlipx to avoid that dependency."""

from __future__ import annotations

import bz2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator
import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import requests
from tqdm import tqdm

BENCHMARK_DATA_DIR = Path.home() / ".cache" / "ml_peg"


# ---------------------------------------------------------------------------
# ASE <-> Phonopy conversions (Author: Balázs Póta)
# ---------------------------------------------------------------------------


def _ase_atoms_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    """
    Convert ASE atoms to a PhonopyAtoms object.

    Parameters
    ----------
    atoms
        ASE atoms to convert.

    Returns
    -------
    PhonopyAtoms
        Equivalent phonopy atom container.
    """
    return PhonopyAtoms(
        atoms.symbols, cell=atoms.cell, positions=atoms.positions, pbc=True
    )


def _ase_atoms_to_phonopy(
    atoms: Atoms, fc2_supercell, primitive_matrix=None, symprec=1e-5, **kwargs
) -> Phonopy:
    """
    Build a Phonopy object from ASE atoms and benchmark cell metadata.

    Parameters
    ----------
    atoms
        ASE atoms used as the phonopy unit cell.
    fc2_supercell
        Supercell matrix used for second-order force constant displacements.
    primitive_matrix
        Primitive matrix passed through to phonopy.
    symprec
        Symmetry tolerance used by phonopy.
    **kwargs
        Additional keyword arguments forwarded to ``Phonopy``.

    Returns
    -------
    Phonopy
        Initialized phonopy object.
    """
    return Phonopy(
        unitcell=_ase_atoms_to_phonopy_atoms(atoms),
        supercell_matrix=fc2_supercell,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        **kwargs,
    )


def phonopy_to_ase_atoms(phonons: Phonopy, primitive: bool | None = None) -> Atoms:
    """
    Convert a phonopy unit or primitive cell to ASE atoms.

    Parameters
    ----------
    phonons
        Phonopy object containing the cell and supercell metadata.
    primitive
        If true, convert the primitive cell instead of the unit cell.

    Returns
    -------
    Atoms
        ASE atoms with ``fc2_supercell`` and ``primitive_matrix`` metadata.
    """
    phonopy_atoms = phonons.primitive if primitive else phonons.unitcell
    atoms = Atoms(
        phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        positions=phonopy_atoms.positions,
        pbc=True,
    )
    if phonons.supercell_matrix is not None:
        atoms.info["fc2_supercell"] = phonons.supercell_matrix
    if phonons.primitive_matrix is not None:
        atoms.info["primitive_matrix"] = phonons.primitive_matrix
    return atoms


# ---------------------------------------------------------------------------
# Force constant calculation (Author: Balázs Póta)
# ---------------------------------------------------------------------------


def _calculate_fc2_set(phonons: Phonopy, calculator: Calculator) -> np.ndarray:
    """
    Calculate forces for the phonopy second-order displacement set.

    Parameters
    ----------
    phonons
        Phonopy object with generated supercells with displacements.
    calculator
        ASE calculator used to evaluate forces.

    Returns
    -------
    np.ndarray
        Force array assigned to ``phonons.forces``.
    """
    forces = []
    nat = len(phonons.supercell)
    for sc in tqdm(
        phonons.supercells_with_displacements, desc="FC2 calculation", leave=False
    ):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.info.setdefault("charge", 0)
            atoms.info.setdefault("spin", 1)
            atoms.calc = calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)
    phonons.forces = np.array(forces)
    return phonons.forces


def init_phonopy_from_ref(
    atoms: Atoms,
    fc2_supercell: np.ndarray | None = None,
    primitive_matrix: Any = "auto",
    symprec: float = 1e-5,
    displacement_dataset: dict | None = None,
    displacement_distance: float | None = None,
    is_plusminus: bool | str = "auto",
    **kwargs: Any,
) -> Phonopy:
    """
    Initialize phonopy from relaxed atoms and reference displacement metadata.

    Parameters
    ----------
    atoms
        Relaxed ASE atoms used as the phonopy unit cell.
    fc2_supercell
        Supercell matrix for second-order force constants. If omitted, this is
        read from ``atoms.info["fc2_supercell"]``.
    primitive_matrix
        Primitive matrix passed to phonopy.
    symprec
        Symmetry tolerance used by phonopy.
    displacement_dataset
        Reference displacement dataset to reuse. If omitted, phonopy generates
        new displacements when ``displacement_distance`` is provided.
    displacement_distance
        Displacement distance used to generate new displacements when
        ``displacement_dataset`` is omitted. Set to ``None`` to require an
        explicit displacement dataset.
    is_plusminus
        Passed to ``Phonopy.generate_displacements`` when new displacements are
        generated. Defaults to phonopy's ``"auto"`` behaviour.
    **kwargs
        Additional keyword arguments forwarded to ``Phonopy``.

    Returns
    -------
    Phonopy
        Phonopy object with the requested displacement dataset.
    """
    if fc2_supercell is None:
        fc2_supercell = atoms.info["fc2_supercell"]
    phonons = _ase_atoms_to_phonopy(
        atoms,
        fc2_supercell=fc2_supercell,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        **kwargs,
    )
    if displacement_dataset is not None:
        phonons.dataset = displacement_dataset
    elif displacement_distance is not None:
        phonons.generate_displacements(
            distance=displacement_distance, is_plusminus=is_plusminus
        )
    else:
        raise ValueError(
            'Either "displacement_dataset" or "displacement_distance" is required when '
            "initializing phonopy from reference data."
        )
    return phonons


def get_fc2_and_freqs(
    phonons: Phonopy,
    calculator: Calculator,
    q_mesh: np.ndarray | None = None,
    symmetrize_fc2: bool = True,
) -> tuple[Phonopy, np.ndarray, np.ndarray]:
    """
    Calculate second-order force constants and optional mesh frequencies.

    Parameters
    ----------
    phonons
        Phonopy object with displacement supercells.
    calculator
        ASE calculator used to evaluate displacement forces.
    q_mesh
        Optional q-point mesh for frequency evaluation.
    symmetrize_fc2
        Whether to symmetrize the produced force constants.

    Returns
    -------
    tuple[Phonopy, np.ndarray, np.ndarray]
        Updated phonopy object, force set, and mesh frequencies.
    """
    fc2_set = _calculate_fc2_set(phonons, calculator)
    phonons.produce_force_constants(show_drift=False)
    if symmetrize_fc2:
        phonons.symmetrize_force_constants(show_drift=False)
    if q_mesh is not None:
        phonons.run_mesh(q_mesh)
        freqs = phonons.get_mesh_dict()["frequencies"]
    else:
        freqs = np.array([])
    return phonons, fc2_set, freqs


def qpath_distances(qpoints: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Cumulative reciprocal-space distances along a q-point path.

    Uses the same convention as phonopy band structures (fractional q-point
    steps mapped through ``inv(cell).T``, no 2π factor), so distances computed
    here for reference data align with phonopy-generated model band distances.

    Parameters
    ----------
    qpoints
        Fractional q-point coordinates, shape ``(nq, 3)``.
    cell
        Real-space cell matrix (rows are lattice vectors) of the primitive cell
        used for the band structure.

    Returns
    -------
    np.ndarray
        Cumulative distances, shape ``(nq,)``, starting at 0.
    """
    qpoints = np.asarray(qpoints, dtype=float)
    rec = np.linalg.inv(np.asarray(cell, dtype=float)).T
    steps = np.linalg.norm(np.diff(qpoints, axis=0) @ rec, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


# ---------------------------------------------------------------------------
# Alexandria phonono dataset download
# ---------------------------------------------------------------------------


def _download_one(mp_id: str, directory: Path) -> str:
    """
    Download and decompress one ALEX phonopy YAML file.

    Parameters
    ----------
    mp_id
        Materials Project identifier to download.
    directory
        Destination directory for the decompressed YAML file.

    Returns
    -------
    str
        Status string describing whether the file existed, succeeded, or failed.
    """
    save_path = directory / f"{mp_id}.yaml"
    if save_path.exists():
        return f"{mp_id}: exists"
    try:
        url = f"https://alexandria.icams.rub.de/data/phonon_benchmark/pbe/{mp_id}.yaml.bz2"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        save_path.write_bytes(bz2.decompress(response.content))
        return f"{mp_id}: success"
    except Exception as e:
        return f"{mp_id}: failed ({e})"


def download_alex_parallel(sample_every: int = 1, max_threads: int = 16) -> None:
    """
    Download ALEX phonon benchmark YAML files in parallel.

    Parameters
    ----------
    sample_every
        Keep every ``sample_every``-th MP ID from the remote index.
    max_threads
        Maximum number of download threads.
    """
    from bs4 import BeautifulSoup

    local_path = BENCHMARK_DATA_DIR / "alex_phonons" / "alex_phonon_data"
    local_path.mkdir(parents=True, exist_ok=True)

    url = "https://alexandria.icams.rub.de/data/phonon_benchmark/pbe/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    mp_ids = [
        re.match(r"(mp-\d+)\.yaml\.bz2$", link["href"]).group(1)
        for link in soup.find_all("a", href=True)
        if re.match(r"(mp-\d+)\.yaml\.bz2$", link["href"])
    ]
    print(f"Found {len(mp_ids)} mp-ids.")

    (local_path.parent / "mp_ids.txt").write_text("\n".join(mp_ids) + "\n")

    mp_ids_sub = mp_ids[::sample_every]
    (local_path.parent / "mp_ids_subsampled.txt").write_text(
        "\n".join(mp_ids_sub) + "\n"
    )

    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(_download_one, mid, local_path): mid for mid in mp_ids_sub
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading phonons"
        ):
            results.append(future.result())

    success = sum("success" in r for r in results)
    fail = sum("failed" in r for r in results)
    if success == 0 and fail == 0:
        print("Found cached files, no download needed")
    else:
        print(f"Download complete: {success} success, {fail} failed")
