"""Shared calculation helpers for the small-cluster force benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import download_s3_data

S3_KEY = "inputs/clusters/cluster_forces/cluster_forces.zip"
S3_FILENAME = "cluster_forces.zip"
INPUT_DIRNAME = "cluster_forces"
INPUT_FILENAME = "clusters.xyz"
OUTPUT_FILENAME = "cluster_forces.extxyz"

# Input arrays/info keys holding each set of reference labels, calculated with
# settings consistent with the MAD-1.5 (r2SCAN) and OMol25 (ωB97M-V) datasets
REFERENCE_KEYS = {
    "r2scan": {"forces": "mad2_forces", "energy": "mad2_energy"},
    "wb97mv": {"forces": "omol25_forces", "energy": "omol25_energy"},
}

REF_FORCES_KEY = "ref_forces"
REF_ENERGY_KEY = "ref_energy"
PRED_FORCES_KEY = "pred_forces"
PRED_ENERGY_KEY = "pred_energy"

# Individual failures are warned about up to this many times per model
MAX_FAILURE_WARNINGS = 5


def load_clusters() -> list[Atoms]:
    """
    Download (if needed) and read the cluster structures and reference labels.

    Returns
    -------
    list[Atoms]
        All clusters, with both sets of reference labels.
    """
    data_dir = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / INPUT_DIRNAME
    return read(data_dir / INPUT_FILENAME, index=":")


def spin_multiplicity(atoms: Atoms) -> int:
    """
    Return the lowest spin multiplicity of a neutral cluster.

    Parameters
    ----------
    atoms
        Cluster structure.

    Returns
    -------
    int
        1 for an even number of electrons, 2 for an odd number.
    """
    return 1 if int(atoms.get_atomic_numbers().sum()) % 2 == 0 else 2


def _output_cluster(atoms: Atoms, reference: str, index: int) -> Atoms:
    """
    Build the output structure for one cluster, keeping only one reference.

    Parameters
    ----------
    atoms
        Input cluster, containing all reference labels.
    reference
        Key of the reference labels to keep, from ``REFERENCE_KEYS``.
    index
        Index of the cluster in the input file.

    Returns
    -------
    Atoms
        Cluster with charge, spin and the selected reference labels.
    """
    keys = REFERENCE_KEYS[reference]
    out = Atoms(numbers=atoms.numbers, positions=atoms.positions, pbc=False)
    out.info["charge"] = 0
    out.info["spin"] = spin_multiplicity(atoms)
    out.info["structure_index"] = index
    out.info["config_type"] = atoms.info.get("config_type", f"{len(atoms)}mer")
    out.info[REF_ENERGY_KEY] = float(atoms.info.get(keys["energy"], np.nan))
    out.arrays[REF_FORCES_KEY] = np.asarray(atoms.arrays[keys["forces"]], dtype=float)
    return out


def run_cluster_forces(
    mlip: tuple[str, Any],
    reference: str,
    out_path: Path,
    add_d3: bool = False,
) -> None:
    """
    Evaluate a model on every cluster and save predictions with one reference.

    Parameters
    ----------
    mlip
        Tuple of model name and model, as provided by ``MODELS.items()``.
    reference
        Key of the reference labels to save, from ``REFERENCE_KEYS``.
    out_path
        Directory to write ``[model]/cluster_forces.extxyz`` to.
    add_d3
        Whether to add a D3 correction to models not trained with dispersion.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    if add_d3:
        calc = model.add_d3_calculator(calc)

    clusters = load_clusters()
    results = []
    n_failed = 0
    for index, atoms in enumerate(
        tqdm(clusters, desc=f"{model_name} cluster forces", unit="cluster")
    ):
        out = _output_cluster(atoms, reference, index)
        out.calc = calc
        try:
            forces = np.asarray(out.get_forces(), dtype=float)
            energy = float(out.get_potential_energy())
            failure = ""
        except Exception as exc:
            forces = np.full((len(out), 3), np.nan)
            energy = np.nan
            # Keep the message on one line and free of quotes for extended XYZ
            failure = f"{type(exc).__name__}: {exc}".replace('"', "'")
            failure = " ".join(failure.split())
            n_failed += 1
            if n_failed <= MAX_FAILURE_WARNINGS:
                warn(
                    f"{model_name}: calculation failed for cluster {index} "
                    f"({out.get_chemical_formula()}): {failure}",
                    stacklevel=2,
                )
        out.calc = None
        out.arrays[PRED_FORCES_KEY] = forces
        out.info[PRED_ENERGY_KEY] = energy
        out.info["calculation_failed"] = bool(failure)
        if failure:
            out.info["calculation_error"] = failure[:500]
        results.append(out)

    if n_failed:
        warn(
            f"{model_name}: {n_failed} of {len(clusters)} cluster calculations failed "
            "and were stored as NaN.",
            stacklevel=2,
        )

    write_dir = out_path / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / OUTPUT_FILENAME, results)
