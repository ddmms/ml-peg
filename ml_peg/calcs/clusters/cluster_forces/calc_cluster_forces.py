"""Evaluate MLIPs on small atomistic clusters using reference forces."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from ase.atoms import Atoms
from ase.io import iread, write
import numpy as np
import pytest
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_model_configs, load_models

MODELS = load_models(current_models)
MODEL_CONFIGS, _ = load_model_configs(tuple(MODELS))

OUT_PATH = Path(__file__).parent / "outputs"
BENCHMARK_FILENAME = "cluster_forces.extxyz"
S3_KEY = "inputs/clusters/cluster_forces/cluster_forces.zip"
S3_FILENAME = "cluster_forces.zip"
INPUT_FILENAME = "clusters.xyz"

MAD2_FORCES_KEY = "mad2_forces"
OMOL25_FORCES_KEY = "omol25_forces"
ORGANIC_MODEL_MARKERS = ("omol", "off", "polar")


def _is_organic_focused_model(model_name: str, model_config: dict[str, Any]) -> bool:
    """
    Return whether a model should be compared to OMOL25 reference forces.

    Parameters
    ----------
    model_name
        Model identifier from ``models.yml``.
    model_config
        Model configuration from ``models.yml``.

    Returns
    -------
    bool
        Whether the model is organic-chemistry focused.
    """
    kwargs = model_config.get("kwargs") or {}
    searchable_fields = [
        model_name,
        model_config.get("class_name", ""),
        model_config.get("module", ""),
        kwargs.get("task_name", ""),
        kwargs.get("name", ""),
        kwargs.get("model", ""),
        kwargs.get("head", ""),
    ]
    searchable = " ".join(str(field).lower() for field in searchable_fields)
    return any(marker in searchable for marker in ORGANIC_MODEL_MARKERS)


def _reference_forces_key(model_name: str) -> str:
    """
    Select the force reference array for a model.

    Organic-focused models are routed to OMOL25 forces. Models trained on broad
    atomistic data or primarily on materials are routed to MAD2 forces.

    Parameters
    ----------
    model_name
        Model identifier from ``models.yml``.

    Returns
    -------
    str
        Extended XYZ array key containing the appropriate reference forces.
    """
    if _is_organic_focused_model(model_name, MODEL_CONFIGS.get(model_name, {})):
        return OMOL25_FORCES_KEY
    return MAD2_FORCES_KEY


def _cluster_data_path() -> Path:
    """
    Download and locate the cluster extended XYZ file.

    Returns
    -------
    pathlib.Path
        Path to ``clusters.xyz``.
    """
    data_dir = download_s3_data(key=S3_KEY, filename=S3_FILENAME)
    candidates = [
        data_dir / INPUT_FILENAME,
        data_dir / "cluster_forces" / INPUT_FILENAME,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {INPUT_FILENAME} after downloading {S3_KEY}. "
        f"Searched:\n{searched}"
    )


def _spin_multiplicity(atoms: Atoms) -> int:
    """
    Return singlet/doublet spin multiplicity for a neutral cluster.

    Parameters
    ----------
    atoms
        Cluster to inspect.

    Returns
    -------
    int
        ``1`` for an even number of electrons, otherwise ``2``.
    """
    n_electrons = int(np.sum(atoms.get_atomic_numbers()))
    return 1 if n_electrons % 2 == 0 else 2


def _set_charge_and_spin(atoms: Atoms) -> None:
    """
    Set neutral charge and parity-based spin metadata in-place.

    Parameters
    ----------
    atoms
        Cluster whose metadata should be updated.
    """
    spin_multiplicity = _spin_multiplicity(atoms)
    atoms.info["charge"] = 0
    atoms.info["spin"] = spin_multiplicity
    atoms.info["spin_multiplicity"] = spin_multiplicity


def _finite_reference_forces(atoms: Atoms, reference_key: str) -> np.ndarray | None:
    """
    Get finite reference forces from an input cluster.

    Parameters
    ----------
    atoms
        Cluster read from ``clusters.xyz``.
    reference_key
        Extended XYZ array key to load.

    Returns
    -------
    numpy.ndarray | None
        Reference forces, or ``None`` when the selected reference is non-finite.
    """
    if reference_key not in atoms.arrays:
        raise KeyError(f"Missing '{reference_key}' in cluster input.")

    ref_forces = np.asarray(atoms.arrays[reference_key], dtype=float)
    if not np.isfinite(ref_forces).all():
        return None
    return ref_forces


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_cluster_forces(mlip: tuple[str, Any]) -> None:
    """
    Compare MLIP forces to the routed MAD2 or OMOL25 cluster-force reference.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    reference_key = _reference_forces_key(model_name)
    reference_target = "OMOL25" if reference_key == OMOL25_FORCES_KEY else "MAD2"

    calc = model.get_calculator(precision="high")
    data_path = _cluster_data_path()
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = iread(data_path, index=":")
    if sys.stderr.isatty():
        frames = tqdm(frames, desc=f"{model_name} cluster forces", unit="cluster")

    results: list[Atoms] = []
    skipped_non_finite = 0
    for structure_index, atoms in enumerate(frames):
        ref_forces = _finite_reference_forces(atoms, reference_key)
        if ref_forces is None:
            skipped_non_finite += 1
            continue

        atoms_pred = atoms.copy()
        _set_charge_and_spin(atoms_pred)
        atoms_pred.calc = calc

        out_atoms = atoms.copy()
        _set_charge_and_spin(out_atoms)
        out_atoms.arrays.pop(MAD2_FORCES_KEY, None)
        out_atoms.arrays.pop(OMOL25_FORCES_KEY, None)
        out_atoms.info["structure_index"] = structure_index
        out_atoms.info["reference_forces_key"] = reference_key
        out_atoms.info["reference_target"] = reference_target
        out_atoms.arrays["ref_forces"] = ref_forces
        out_atoms.arrays["pred_forces"] = np.asarray(
            atoms_pred.get_forces(), dtype=float
        )
        results.append(out_atoms)

    if not results:
        raise ValueError(
            f"No finite {reference_target} force references found in {data_path}."
        )

    write(out_dir / BENCHMARK_FILENAME, results)
    print(
        f"{model_name}: wrote {len(results)} clusters using {reference_target} "
        f"references; skipped {skipped_non_finite} non-finite references."
    )
