"""
Calculate diamond phonon bands (phonopy version-friendly).

This module computes phonon force constants and phonon band dispersions for
diamond using a set of MLIP models. Outputs are written to ``outputs/<model>/``
as:

- ``FORCE_CONSTANTS``
- ``band.yaml``

The band path is taken from the DFT reference NPZ so that predicted dispersions
are evaluated on the exact same q-path as the reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
import pytest

from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

EXTRACTED_ROOT = Path(
    download_github_data(
        filename="diamond_data/data.zip",
        github_uri=GITHUB_BASE,
    )
)

DATA_PATH = EXTRACTED_ROOT / "data"
DIAMOND_YAML = DATA_PATH / "diamond.yaml"
DFT_BAND_NPZ = DATA_PATH / "dft_band.npz"
OUT_PATH = Path(__file__).parent / "outputs"

MODELS = load_models(current_models)


def phonopy_atoms_to_ase(ph_atoms: PhonopyAtoms) -> Atoms:
    """
    Convert ``PhonopyAtoms`` to ASE ``Atoms``.

    Parameters
    ----------
    ph_atoms
        Phonopy atoms object.

    Returns
    -------
    ase.Atoms
        ASE representation with periodic boundary conditions enabled.
    """
    return Atoms(
        symbols=ph_atoms.symbols,
        cell=ph_atoms.cell,
        scaled_positions=ph_atoms.scaled_positions,
        pbc=True,
    )


def load_phonon_yaml(yaml_path: Path) -> Any:
    """
    Load a phonopy object from a phonopy YAML file (version-tolerant).

    Parameters
    ----------
    yaml_path
        Path to a phonopy YAML file (e.g. ``diamond.yaml``).

    Returns
    -------
    object
        A phonopy object as returned by ``phonopy.load`` or the older YAML reader.

    Raises
    ------
    FileNotFoundError
        If ``yaml_path`` does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing phonopy YAML: {yaml_path}")

    try:
        import phonopy  # type: ignore

        return phonopy.load(str(yaml_path))
    except Exception:
        from phonopy.interface.phonopy_yaml import read_phonopy_yaml  # type: ignore

        obj = read_phonopy_yaml(str(yaml_path))
        return getattr(obj, "phonopy", obj)


def get_displaced_supercells(phonon: Any) -> list[PhonopyAtoms]:
    """
    Return displaced supercells from a phonopy object (version-tolerant).

    Parameters
    ----------
    phonon
        Phonopy object.

    Returns
    -------
    list[phonopy.structure.atoms.PhonopyAtoms]
        Displaced supercells. Returns an empty list if not available.
    """
    if hasattr(phonon, "get_supercells_with_displacements"):
        return phonon.get_supercells_with_displacements()
    return getattr(phonon, "supercells_with_displacements", []) or []


def ref_band_path() -> list[list[list[float]]]:
    """
    Build a q-point path for ``phonon.run_band_structure`` from the DFT NPZ.

    The DFT reference NPZ is used so that predicted dispersions are computed on
    the exact same q-path as the reference.

    The NPZ is expected to contain:

    - ``qpoints``: array with shape ``(Nq, 3)``
    - ``distance``: array with shape ``(Nq,)`` (optional; used to detect segment
      boundaries). If missing, the q-path is treated as a single segment.

    Returns
    -------
    list[list[list[float]]]
        List of q-point segments, each a list of ``[qx, qy, qz]`` points.

    Raises
    ------
    FileNotFoundError
        If the DFT reference NPZ does not exist.
    KeyError
        If the NPZ does not contain ``qpoints``.
    ValueError
        If ``qpoints`` or ``distance`` have invalid shapes.
    """
    if not DFT_BAND_NPZ.exists():
        raise FileNotFoundError(f"Missing DFT reference NPZ: {DFT_BAND_NPZ}")

    obj = np.load(DFT_BAND_NPZ, allow_pickle=False)
    if "qpoints" not in obj:
        raise KeyError(f"{DFT_BAND_NPZ} missing required key 'qpoints'.")

    q = np.asarray(obj["qpoints"], float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"Bad qpoints shape in {DFT_BAND_NPZ}: {q.shape}")

    x = np.asarray(obj["distance"], float) if "distance" in obj else None
    if x is None:
        return [q.tolist()]

    if x.ndim != 1 or x.shape[0] != q.shape[0]:
        raise ValueError(f"Bad distance shape in {DFT_BAND_NPZ}: {x.shape}")

    cuts = [0] + [i for i in range(1, len(x)) if x[i] <= x[i - 1] + 1e-12] + [len(x)]
    return [q[cuts[i] : cuts[i + 1]].tolist() for i in range(len(cuts) - 1)]


def write_force_constants(phonon: Any, fc_path: Path) -> None:
    """
    Write force constants to disk in a phonopy-version-tolerant way.

    Parameters
    ----------
    phonon
        Phonopy object with computed force constants.
    fc_path
        Output path for ``FORCE_CONSTANTS``.
    """
    fc_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        phonon.write_force_constants(filename=str(fc_path))
    except AttributeError:
        from phonopy.file_IO import write_FORCE_CONSTANTS  # type: ignore

        write_FORCE_CONSTANTS(phonon.force_constants, filename=str(fc_path))


def write_band_yaml(phonon: Any, bands_path: Path) -> None:
    """
    Write phonopy band-structure YAML to disk.

    Parameters
    ----------
    phonon
        Phonopy object that has already run ``run_band_structure``.
    bands_path
        Output path for ``band.yaml``.

    Raises
    ------
    RuntimeError
        If a YAML writer is not available for the current phonopy version.
    """
    bands_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(phonon, "write_yaml_band_structure"):
        phonon.write_yaml_band_structure(filename=str(bands_path))
        return

    bs = getattr(phonon, "band_structure", None)
    if bs is None or not hasattr(bs, "write_yaml"):
        raise RuntimeError(
            "Phonopy band-structure YAML writer not found in this phonopy version."
        )
    bs.write_yaml(filename=str(bands_path))


@pytest.mark.parametrize("mlip", MODELS.items())
def test_diamond_phonons_band(mlip) -> None:
    """
    Compute diamond phonon bands for one MLIP and write output files.

    Parameters
    ----------
    mlip
        Tuple ``(model_name, model)`` from ``MODELS.items()``.

    Raises
    ------
    RuntimeError
        If the phonopy YAML lacks a displacement dataset, has no displaced
        supercells, or cannot write band YAML for the current phonopy version.
    """
    model_name, model = mlip

    phonon = load_phonon_yaml(DIAMOND_YAML)
    if getattr(phonon, "dataset", None) is None:
        raise RuntimeError(f"{DIAMOND_YAML} has no displacement dataset.")

    scells = get_displaced_supercells(phonon)
    if not scells:
        raise RuntimeError(f"No displaced supercells found in {DIAMOND_YAML}.")

    calc = model.get_calculator()

    forces = []
    for scp in scells:
        ase_sc = phonopy_atoms_to_ase(scp)
        ase_sc.calc = calc
        forces.append(ase_sc.get_forces())

    if hasattr(phonon, "set_forces"):
        phonon.set_forces(forces)
    else:
        phonon.forces = forces

    phonon.produce_force_constants()
    phonon.run_band_structure(ref_band_path())

    write_dir = OUT_PATH / model_name
    fc_path = write_dir / "FORCE_CONSTANTS"
    bands_path = write_dir / "band.yaml"

    write_force_constants(phonon, fc_path)
    write_band_yaml(phonon, bands_path)

    assert fc_path.exists(), f"Missing FORCE_CONSTANTS for {model_name}: {fc_path}"
    assert bands_path.exists(), f"Missing band.yaml for {model_name}: {bands_path}"
