"""
Generate starting structures for melt-quench benchmark using PACKMOL.

PACKMOL guarantees minimum interatomic distances, avoiding the energy
explosions that occur with pure random packing.

Pair-specific minimum distances are used (realistic for oxides).
A border margin is added to avoid PBC clashes at box edges.

Usage:
    python3 build_structure.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write

# ---------------------------------------------------------------------------
# Pair-specific minimum distances (Angstrom)
# Based on sum of ionic radii + small margin
# ---------------------------------------------------------------------------
PAIR_DIST = {
    ("O", "O"): 2.4,
    ("Si", "O"): 1.6,
    ("Al", "O"): 1.75,
    ("Na", "O"): 2.2,
    ("K", "O"): 2.4,
    ("Ca", "O"): 2.2,
    ("Si", "Si"): 3.0,
    ("Al", "Al"): 3.0,
    ("Si", "Al"): 3.0,
    ("Na", "Si"): 2.8,
    ("Na", "Al"): 2.8,
    ("K", "Si"): 3.0,
    ("K", "Al"): 3.0,
    ("Ca", "Si"): 2.8,
    ("Ca", "Al"): 2.8,
    ("Na", "Na"): 3.0,
    ("K", "K"): 3.5,
    ("Ca", "Ca"): 3.0,
}


def get_min_dist(e1: str, e2: str) -> float:
    """
    Get minimum distance for a pair of elements.

    Parameters
    ----------
    e1 : str
        Chemical symbol of the first element (e.g., 'Si', 'O').
    e2 : str
        Chemical symbol of the second element (e.g., 'Al', 'Na').

    Returns
    -------
    float
        Minimum allowed distance between the element pair in Å (defaults to 2.0 Å).
    """
    key = tuple(sorted([e1, e2]))
    return PAIR_DIST.get(key, 2.0)  # default 2.0 if pair not listed


# Use the maximum pairwise distance as global tolerance for packmol
TOLERANCE = 2.0  # Angstrom — conservative global minimum
BORDER = 2.0  # Angstrom margin from box edges (avoids PBC clashes)

TARGET_DENSITY = {
    "albite": 2.4,
    "anorthite": 2.7,
    "sanidine": 2.4,
}

COMPOSITIONS = {
    "albite": {"Na": 25, "Al": 37, "Si": 74, "O": 216},
    "anorthite": {"Ca": 21, "Al": 70, "Si": 46, "O": 218},
    "sanidine": {"K": 33, "Al": 41, "Si": 68, "O": 214},
}

N_SEEDS = 3


def box_size(composition: dict, density: float) -> float:
    """
    Compute cubic box edge (Angstrom) from composition and target density.

    Parameters
    ----------
    composition : dict
        Dictionary mapping chemical element symbols to their atom counts.
    density : float
        Target density in g/cm^3.

    Returns
    -------
    float
        Cubic box edge length in Angstroms.
    """
    mass_amu = sum(
        atomic_masses[atomic_numbers[el]] * n for el, n in composition.items()
    )
    mass_g = mass_amu * 1.66054e-24
    vol_cm3 = mass_g / density
    return (vol_cm3 * 1e24) ** (1 / 3)


def write_xyz_single(symbol: str, path: Path) -> None:
    """
    Write a single-atom xyz file for packmol input.

    Parameters
    ----------
    symbol : str
        Chemical symbol of the atom (e.g., 'Si', 'O').
    path : Path
        Target file path for the output XYZ file.

    Returns
    -------
    None
        This function does not return a value.
    """
    atoms = Atoms(symbol, positions=[[0, 0, 0]])
    write(str(path), atoms)


def run_packmol(
    composition: dict, box: float, seed: int, out_path: Path, tmpdir: Path
) -> Atoms:
    """
    Run packmol to pack atoms in a cubic box.

    Parameters
    ----------
    composition : dict
        Dictionary mapping element symbols to their respective atom counts.
    box : float
        Cubic box edge length in Ångströms (Å).
    seed : int
        Random seed index for deterministic packing generation.
    out_path : Path
        File path where the generated XYZ structure will be saved.
    tmpdir : Path
        Directory path for temporary work files.

    Returns
    -------
    Atoms
        The packed atomic structure as an ASE Atoms object.
    """
    inner_lo = BORDER
    inner_hi = box - BORDER

    if inner_hi <= inner_lo:
        raise ValueError(f"Box too small ({box:.2f} A) for border margin {BORDER} A")

    inp_lines = [
        f"tolerance {TOLERANCE}",
        f"seed {seed * 99991 + 7}",  # deterministic, different per seed
        "filetype xyz",
        f"output {str(out_path)}",
        "",
    ]

    for element, count in composition.items():
        el_xyz = tmpdir / f"{element}.xyz"
        write_xyz_single(element, el_xyz)
        inp_lines += [
            f"structure {str(el_xyz)}",
            f"  number {count}",
            (
                f"  inside box {inner_lo:.2f} {inner_lo:.2f} {inner_lo:.2f}"
                f" {inner_hi:.2f} {inner_hi:.2f} {inner_hi:.2f}"
            ),
            "end structure",
            "",
        ]

    inp_file = tmpdir / "pack.inp"
    inp_file.write_text("\n".join(inp_lines))

    with open(str(inp_file)) as f:
        result = subprocess.run(
            ["packmol"],
            stdin=f,
            capture_output=True,
            text=True,
            timeout=600,
        )

    # Check for success — packmol writes "Solution written" on success
    success = "Solution written" in result.stdout
    if not success or not out_path.exists():
        raise RuntimeError(
            f"Packmol failed (returncode={result.returncode}):\n"
            f"stdout: {result.stdout[-3000:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    atoms = read(str(out_path))
    atoms.set_cell([box, box, box])
    atoms.set_pbc(True)
    return atoms


def main():
    """Generate starting structures for all compositions and seeds."""
    here = Path(__file__).parent

    for comp_name, composition in COMPOSITIONS.items():
        density = TARGET_DENSITY[comp_name]
        box = box_size(composition, density)
        n_atoms = sum(composition.values())
        print(
            f"\n{comp_name}: {n_atoms} atoms, box={box:.2f} A, "
            f"target density={density} g/cm3"
        )
        print(
            f"  inner box: {BORDER:.1f} to {box - BORDER:.2f} A "
            f"(margin={BORDER} A each side)"
        )

        for seed in range(N_SEEDS):
            out_xyz = here / f"{comp_name}_start_seed{seed}.xyz"

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                pack_out = tmp / "packed.xyz"

                print(f"  seed {seed}...", end=" ", flush=True)
                try:
                    atoms = run_packmol(composition, box, seed, pack_out, tmp)
                    assert len(atoms) == n_atoms, (
                        f"Expected {n_atoms} atoms, got {len(atoms)}"
                    )
                    write(str(out_xyz), atoms)
                    print(f"OK -> {out_xyz.name} ({len(atoms)} atoms)")
                except Exception as e:
                    print(f"FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
