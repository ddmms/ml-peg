"""Prepare the MAD-BENCH input subset from the MAD-1.6 r2SCAN test split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ase import Atoms
from ase.io import iread, write
import numpy as np

SOURCE_DATASET = "MAD-1.6"
SOURCE_SPLIT = "test"
SOURCE_RECORD_DOI = "10.24435/materialscloud:jc-9f"
SOURCE_PAPER = "arXiv:2603.02089v3"

REFERENCE_FORCES_KEY = "r2scan_forces"
REFERENCE_ENERGY_KEY = "r2scan_energy"
REFERENCE_ATOMIZATION_ENERGY_KEY = "r2scan_atomization_energy"
REFERENCE_FREE_ENERGY_KEY = "r2scan_free_energy"
REFERENCE_STRESS_KEY = "r2scan_stress"

REFERENCE_LABELS = {
    REFERENCE_FORCES_KEY: "eV/Angstrom, per atom, all structures",
    REFERENCE_ENERGY_KEY: "eV, all-electron total energy, all structures",
    REFERENCE_ATOMIZATION_ENERGY_KEY: (
        "eV, total energy minus isolated-atom energies, all structures"
    ),
    REFERENCE_FREE_ENERGY_KEY: "eV, only where the source has free_energy",
    REFERENCE_STRESS_KEY: (
        "eV/Angstrom^3, Voigt order xx yy zz yz xz xy, ASE sign convention, "
        "only where the source has stress"
    ),
}

N_STRUCTURES = 2000
SEED = 42

OUTPUT_DIRNAME = "mad_bench"
STRUCTURES_FILENAME = "mad_bench.xyz"
MANIFEST_FILENAME = "selection.json"


def sha256sum(path: Path) -> str:
    """
    Get the SHA-256 checksum of a file.

    Parameters
    ----------
    path
        File to checksum.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_label_problems(atoms: Atoms) -> list[str]:
    """
    Check the reference forces, energy and stress of one source structure.

    Forces of shape (n_atoms, 3) and a total energy are required. Stress is optional,
    but if present it must have six finite Voigt components on a fully periodic
    structure.

    Parameters
    ----------
    atoms
        Source structure.

    Returns
    -------
    list[str]
        Description of each problem, empty if all labels are valid.
    """
    results = atoms.calc.results if atoms.calc is not None else {}
    problems = []

    forces = results.get("forces")

    if forces is None:
        problems.append("missing forces")
    elif np.shape(forces) != (len(atoms), 3):
        problems.append(f"forces shape {np.shape(forces)}")
    elif not np.isfinite(forces).all():
        problems.append("non-finite forces")

    energy = results.get("energy")

    if energy is None or not np.isfinite(energy):
        problems.append("missing or non-finite energy")

    if not np.isfinite(atoms.info.get("atomization_energy", np.nan)):
        problems.append("missing or non-finite atomization energy")

    stress = results.get("stress")

    if stress is not None:
        if np.shape(stress) != (6,) or not np.isfinite(stress).all():
            problems.append("invalid stress")
        elif not atoms.pbc.all():
            problems.append("stress on a structure that is not fully periodic")

    return problems


def find_invalid_labels(structures: list[Atoms]) -> list[dict]:
    """
    Find structures with invalid reference labels.

    Parameters
    ----------
    structures
        Structures read from the source file, in source order.

    Returns
    -------
    list[dict]
        Source index, identity and problems for each invalid structure.
    """
    invalid = []

    for source_index, atoms in enumerate(structures):
        problems = find_label_problems(atoms)

        if problems:
            invalid.append(
                {
                    "source_index": source_index,
                    "subset": atoms.info.get("subset"),
                    "frame_id": atoms.info.get("frame_id"),
                    "problems": problems,
                }
            )

    return invalid


def select_indices(n_source: int, n_select: int, seed: int) -> np.ndarray:
    """
    Sample source indices uniformly without replacement.

    Parameters
    ----------
    n_source
        Number of structures in the source file.
    n_select
        Number of structures to select.
    seed
        Seed for the NumPy random generator.

    Returns
    -------
    np.ndarray
        Selected source indices in ascending order.
    """
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_source, size=n_select, replace=False)

    return np.sort(selected)


def build_subset_structure(atoms: Atoms, source_index: int) -> Atoms:
    """
    Copy the geometry, identity and reference labels of one source structure.

    Parameters
    ----------
    atoms
        Source structure.
    source_index
        Index of the structure in the source file.

    Returns
    -------
    Atoms
        Structure to include in the benchmark input file.
    """
    subset_atoms = Atoms(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )

    subset_atoms.info["source_index"] = int(source_index)
    subset_atoms.info["subset"] = atoms.info["subset"]
    subset_atoms.info["frame_id"] = int(atoms.info["frame_id"])

    results = atoms.calc.results

    subset_atoms.arrays[REFERENCE_FORCES_KEY] = np.array(results["forces"])
    subset_atoms.info[REFERENCE_ENERGY_KEY] = float(results["energy"])
    subset_atoms.info[REFERENCE_ATOMIZATION_ENERGY_KEY] = float(
        atoms.info["atomization_energy"]
    )

    if "free_energy" in results:
        subset_atoms.info[REFERENCE_FREE_ENERGY_KEY] = float(results["free_energy"])

    if "stress" in results:
        subset_atoms.info[REFERENCE_STRESS_KEY] = np.array(results["stress"])

    return subset_atoms


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to mad-1.6-r2scan-test.xyz from the Materials Cloud record",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which the mad_bench directory is written.",
    )
    parser.add_argument("--n-structures", type=int, default=N_STRUCTURES)
    parser.add_argument("--seed", type=int, default=SEED)

    return parser.parse_args()


def main() -> None:
    """Validate the source file, sample the subset and write it with a manifest."""
    args = parse_args()

    source_checksum = sha256sum(args.source)
    structures = list(iread(args.source, index=":"))
    print(f"read {len(structures)} structures from {args.source}")

    invalid = find_invalid_labels(structures)

    if invalid:
        for record in invalid:
            print(record)

        raise SystemExit(
            f"{len(invalid)} structures have invalid reference labels. "
            "No exclusions were applied. Review them before sampling."
        )

    selected_indices = select_indices(len(structures), args.n_structures, args.seed)

    subset = [
        build_subset_structure(structures[index], index) for index in selected_indices
    ]

    out_dir = args.output_dir / OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    structures_path = out_dir / STRUCTURES_FILENAME
    write(structures_path, subset, format="extxyz")

    n_stress = sum(REFERENCE_STRESS_KEY in atoms.info for atoms in subset)

    manifest = {
        "source_dataset": SOURCE_DATASET,
        "source_split": SOURCE_SPLIT,
        "source_record_doi": SOURCE_RECORD_DOI,
        "source_paper": SOURCE_PAPER,
        "source_file": args.source.name,
        "source_sha256": source_checksum,
        "source_structures": len(structures),
        "invalid_source_structures": len(invalid),
        "reference_labels": REFERENCE_LABELS,
        "sampling": "numpy.random.default_rng(seed).choice without replacement",
        "seed": args.seed,
        "numpy_version": np.__version__,
        "n_structures": len(subset),
        "n_stress_structures": n_stress,
        "structures_file": STRUCTURES_FILENAME,
        "structures_sha256": sha256sum(structures_path),
        "selected": [
            {
                "source_index": atoms.info["source_index"],
                "subset": atoms.info["subset"],
                "frame_id": atoms.info["frame_id"],
            }
            for atoms in subset
        ],
    }

    with (out_dir / MANIFEST_FILENAME).open("w", encoding="utf8") as file:
        json.dump(manifest, file, indent=1)

    subset_counts = Counter(atoms.info["subset"] for atoms in subset)
    n_atoms = sum(len(atoms) for atoms in subset)

    print(f"wrote {len(subset)} structures with {n_atoms} atoms to {structures_path}")
    print(f"structures per subset {dict(sorted(subset_counts.items()))}")
    print(f"structures with reference stress {n_stress}")


if __name__ == "__main__":
    main()
