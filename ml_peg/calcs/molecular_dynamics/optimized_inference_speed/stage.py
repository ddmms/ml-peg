"""Stage a campaign input directory from the spec template and base cells."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from warnings import catch_warnings, simplefilter, warn

from ase import Atoms
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np

from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.contract import (
    SPEC_TEMPLATE,
)

TEMPERATURE_K = 300.0
REFERENCE_TARGET = 16_000
MAX_EDGE_RATIO = 1.15


def sha256_file(path: Path) -> str:
    """
    Compute the sha256 checksum of a file.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_mlpeg_commit() -> str:
    """
    Get the current ML-PEG git commit.

    Returns
    -------
    str
        Commit hash, or ``"unknown"`` if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def choose_supercell(
    cell_lengths: np.ndarray, n_base: int, target: int
) -> tuple[int, int, int]:
    """
    Choose integer repeats giving a near-cubic box with close to `target` atoms.

    Candidates with box edges within ``MAX_EDGE_RATIO`` of each other are ranked by
    the log-distance of their atom count from `target`. If none are cubic enough,
    the most cubic candidate is used instead.

    Parameters
    ----------
    cell_lengths
        Lengths of the base cell vectors, in Å.
    n_base
        Number of atoms in the base cell.
    target
        Target number of atoms.

    Returns
    -------
    tuple[int, int, int]
        Repeats along each cell vector.
    """
    lengths = np.asarray(cell_lengths, dtype=float)
    # Edge of a cube with the target number of atoms bounds the useful repeats
    edge = (target / n_base * np.prod(lengths)) ** (1 / 3)
    max_repeats = [max(1, int(np.ceil(2 * edge / length))) for length in lengths]
    grids = np.meshgrid(*(np.arange(1, n + 1) for n in max_repeats), indexing="ij")
    repeats = np.stack([grid.ravel() for grid in grids], axis=1)

    edges = repeats * lengths
    ratio = edges.max(axis=1) / edges.min(axis=1)
    distance = np.abs(np.log(repeats.prod(axis=1) * n_base / target))

    cubic = ratio <= MAX_EDGE_RATIO
    if cubic.any():
        # Sort by distance, then cubicity, then repeats for determinism
        order = np.lexsort((*repeats.T[::-1], ratio, distance))
        best = next(i for i in order if cubic[i])
    else:
        best = np.lexsort((*repeats.T[::-1], distance, ratio))[0]
    return tuple(int(n) for n in repeats[best])


def momenta_seed(material: str, n_atoms: int) -> int:
    """
    Derive a fixed random seed from the material and size.

    Parameters
    ----------
    material
        Material identifier.
    n_atoms
        Number of atoms.

    Returns
    -------
    int
        Seed for the momenta.
    """
    digest = hashlib.sha256(f"{material}-{n_atoms}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def build_structure(base: Atoms, supercell: tuple[int, int, int], seed: int) -> Atoms:
    """
    Build a supercell with Maxwell-Boltzmann momenta at 300 K.

    Parameters
    ----------
    base
        Base cell.
    supercell
        Repeats along each cell vector.
    seed
        Seed for the momenta.

    Returns
    -------
    Atoms
        Supercell with positions, cell, PBC, masses and momenta only.
    """
    clean = Atoms(
        numbers=base.numbers, positions=base.positions, cell=base.cell, pbc=True
    )
    clean.wrap()
    atoms = clean.repeat(supercell)
    # Store masses explicitly so they are written to the extxyz file
    atoms.set_masses(atoms.get_masses())
    # Deprecated in favour of thermalize_momenta from ASE 3.29, but ASE 3.28 is
    # still supported
    with catch_warnings():
        simplefilter("ignore", DeprecationWarning)
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(seed)
        )
    Stationary(atoms)
    return atoms


def stage_campaign(
    base_dir: Path,
    out: Path,
    template: Path = SPEC_TEMPLATE,
    campaign_id: str | None = None,
    max_atoms: int | None = None,
) -> dict:
    """
    Stage a campaign input directory.

    Writes ``<out>/in/spec.json`` and ``<out>/in/structures/<material>_<n>.extxyz``.
    Materials without a base cell at ``<base_dir>/<material>.extxyz`` are skipped
    with a warning.

    Parameters
    ----------
    base_dir
        Directory containing base cells.
    out
        Campaign directory to create.
    template
        Spec template. Its size ladders are used as targets.
    campaign_id
        Campaign ID. Default is derived from the template and base cells.
    max_atoms
        Skip targets larger than this. Default is no limit.

    Returns
    -------
    dict
        The staged spec.
    """
    template_text = template.read_text(encoding="utf8")
    spec = json.loads(template_text)
    in_dir = out / "in"
    struct_dir = in_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    id_hash = hashlib.sha256(template_text.encode())
    systems = []
    for system in spec["systems"]:
        material = system["material"]
        base_path = base_dir / f"{material}.extxyz"
        if not base_path.is_file():
            warn(
                f"No base cell for '{material}' at {base_path}; skipping.",
                stacklevel=2,
            )
            continue
        id_hash.update(sha256_file(base_path).encode())
        base = read(base_path)
        if not base.pbc.all() or base.cell.rank != 3:
            raise ValueError(f"Base cell {base_path} must be periodic in 3D.")

        targets = [size["n_atoms"] for size in system["sizes"]]
        sizes = []
        for target in targets:
            supercell = choose_supercell(base.cell.lengths(), len(base), target)
            n_atoms = len(base) * int(np.prod(supercell))
            if max_atoms is not None and n_atoms > max_atoms:
                continue
            if any(size["n_atoms"] == n_atoms for size in sizes):
                warn(f"Duplicate size {n_atoms} for '{material}'.", stacklevel=2)
                continue
            filename = f"structures/{material}_{n_atoms}.extxyz"
            atoms = build_structure(base, supercell, momenta_seed(material, n_atoms))
            write(in_dir / filename, atoms, format="extxyz")
            sizes.append(
                {
                    "n_atoms": n_atoms,
                    "target_n_atoms": target,
                    "supercell": list(supercell),
                    "file": filename,
                    "sha256": sha256_file(in_dir / filename),
                    "validate": False,
                    "reference": False,
                }
            )
        if not sizes:
            warn(f"No sizes within limits for '{material}'; skipping.", stacklevel=2)
            continue
        sizes.sort(key=lambda size: size["n_atoms"])
        sizes[0]["validate"] = True
        reference = min(sizes, key=lambda size: abs(size["n_atoms"] - REFERENCE_TARGET))
        reference["reference"] = True

        elements = sorted(set(base.get_chemical_symbols()))
        staged = copy.deepcopy(system)
        staged.update(
            {
                "elements": elements,
                "contains_hydrogen": "H" in elements,
                "base_cell_atoms": len(base),
                "sizes": sizes,
            }
        )
        systems.append(staged)

    spec["systems"] = systems
    spec["campaign"] = {
        "id": campaign_id or f"campaign-{id_hash.hexdigest()[:12]}",
        "mlpeg_commit": get_mlpeg_commit(),
        "hardware_id": None,
    }
    (in_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n", "utf8")
    return spec
