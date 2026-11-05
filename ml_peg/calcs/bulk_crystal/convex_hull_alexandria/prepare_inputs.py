"""
Build the input archive for the Alexandria convex-hull benchmark.

The archive collects the unary and binary PBE entries of the Alexandria database
(Schmidt et al., Cavignac et al.), release ``2025.07.02``:

* chemical systems with a PBE+U calculation (O or F with Co, Cr, Fe, Mn, Mo, Ni,
  V or W) are excluded, so that all remaining energies are plain PBE;
* the hull of each binary system is built from the raw PBE energies, without the
  MP2020-style corrections stored in Alexandria;
* every binary on the hull is kept, with the ground state of each element and up
  to ``N_UNSTABLE`` unstable binaries per system, drawn with a fixed seed from
  the energy-above-hull bins in ``E_HULL_BINS``.

The archive is hosted on the ML-PEG S3 bucket as
``inputs/bulk_crystal/convex_hull_alexandria/convex_hull_alexandria.zip``. This
script documents its provenance and can regenerate it in the ML-PEG cache. It
downloads about 3.5 GB.

Run with::

    python ml_peg/calcs/bulk_crystal/convex_hull_alexandria/prepare_inputs.py
"""

from __future__ import annotations

import bz2
from collections import defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

from ase import Atoms
from ase.io import write
import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
import requests
from tqdm import tqdm

from ml_peg.calcs.utils.utils import BENCHMARK_DATA_DIR

ARCHIVE_NAME = "convex_hull_alexandria"

ALEXANDRIA_URL = "https://alexandria.icams.rub.de/data/pbe/2025.07.02"
ALEXANDRIA_FILES = [f"alexandria_{i:05d}.json.bz2" for i in range(58)]

# Elements treated with PBE+U in oxides and fluorides (Materials Project scheme)
U_ELEMENTS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}
U_ANIONS = {"O", "F"}

# Unstable binaries drawn per system, one from each bin of energy above the hull
# (eV/atom) where possible; remaining slots are filled from the other entries
N_UNSTABLE = 5
E_HULL_BINS = ((0, 0.025), (0.025, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, np.inf))
STABLE_TOL = 1e-6  # eV/atom
SEED = 0

README = f"""\
Alexandria convex-hull benchmark inputs for ML-PEG.

structures.extxyz
    Unary and binary PBE structures from the Alexandria database, release
    2025.07.02 ({ALEXANDRIA_URL}, CC BY 4.0), with:

    mat_id        Alexandria identifier
    dft_energy    raw PBE total energy (eV), without energy corrections
    e_above_hull  energy above the raw-energy PBE hull of the system (eV/atom)

    Systems containing a PBE+U calculation ({", ".join(sorted(U_ANIONS))} with
    {", ".join(sorted(U_ELEMENTS))}) are excluded. All binaries on the hull are
    included, with the ground state of each element and up to {N_UNSTABLE}
    unstable binaries per system (seed {SEED}).

References: T. Cavignac et al., J. Phys. Mater. 9, 025014 (2026),
https://doi.org/10.1088/2515-7639/ae6620; J. Schmidt et al., Adv. Mater. 35,
2210788 (2023), https://doi.org/10.1002/adma.202210788
"""


def is_plus_u(elements: set[str]) -> bool:
    """
    Check whether a chemical system is treated with PBE+U.

    Parameters
    ----------
    elements
        Elements of the chemical system.

    Returns
    -------
    bool
        Whether the system contains an oxide or fluoride of a +U element.
    """
    return bool(elements & U_ELEMENTS) and bool(elements & U_ANIONS)


def load_entries(names: list[str] = ALEXANDRIA_FILES) -> list[dict]:
    """
    Download Alexandria files and keep their unary and binary entries.

    Parameters
    ----------
    names
        Names of the Alexandria files.

    Returns
    -------
    list[dict]
        Unary and binary entries, as pymatgen ``ComputedStructureEntry`` dicts.
    """
    kept = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in tqdm(names, desc="Alexandria files"):
            path = Path(tmp) / name
            with requests.get(f"{ALEXANDRIA_URL}/{name}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path, "wb") as file:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        file.write(chunk)
            with bz2.open(path) as file:
                entries = json.load(file)["entries"]
            kept += [entry for entry in entries if len(entry["composition"]) <= 2]
            path.unlink()
    return kept


def energies_above_hull(entries: list[dict]) -> dict[str, float]:
    """
    Get the energy above the raw-energy hull of each binary entry.

    Parameters
    ----------
    entries
        Unary and binary entries.

    Returns
    -------
    dict[str, float]
        Energy above the hull of its system (eV/atom), by ``mat_id``, for the
        binaries of systems without PBE+U.
    """
    unaries = defaultdict(list)
    binaries = defaultdict(list)
    for entry in entries:
        pd_entry = PDEntry(
            Composition(entry["composition"]),
            entry["data"]["energy_total"],
            name=entry["data"]["mat_id"],
        )
        elements = frozenset(entry["composition"])
        (unaries if len(elements) == 1 else binaries)[elements].append(pd_entry)

    e_hull = {}
    for elements, system in tqdm(binaries.items(), desc="Hulls"):
        if is_plus_u(set(elements)):
            continue
        hull = PhaseDiagram(
            system + [e for el in elements for e in unaries[frozenset([el])]]
        )
        for pd_entry in system:
            e_hull[pd_entry.name] = hull.get_e_above_hull(pd_entry)
    return e_hull


def select_entries(
    entries: list[dict], e_hull: dict[str, float], seed: int = SEED
) -> list[dict]:
    """
    Select the elemental ground states, stable binaries and sampled unstable ones.

    Parameters
    ----------
    entries
        Unary and binary entries.
    e_hull
        Energy above the hull of each binary, by ``mat_id``.
    seed
        Seed of the random sampling.

    Returns
    -------
    list[dict]
        Selected entries, with ``e_above_hull`` set to the raw-energy value.
    """
    rng = np.random.default_rng(seed)
    ground_states = {}
    systems = defaultdict(list)
    for entry in entries:
        data = entry["data"]
        elements = sorted(entry["composition"])
        if len(elements) == 1:
            n_atoms = sum(entry["composition"].values())
            current = ground_states.get(elements[0])
            if current is None or (
                data["energy_total"] / n_atoms
                < current["data"]["energy_total"] / sum(current["composition"].values())
            ):
                ground_states[elements[0]] = entry
        elif data["mat_id"] in e_hull:
            systems["-".join(elements)].append(entry)

    selected = []
    for system in sorted(systems):
        candidates = sorted(systems[system], key=lambda e: e["data"]["mat_id"])
        unstable = []
        for entry in candidates:
            entry["data"]["e_above_hull"] = e_hull[entry["data"]["mat_id"]]
            if entry["data"]["e_above_hull"] < STABLE_TOL:
                selected.append(entry)
            else:
                unstable.append(entry)

        picked = []
        for low, high in E_HULL_BINS:
            in_bin = [e for e in unstable if low < e["data"]["e_above_hull"] <= high]
            if in_bin:
                picked.append(in_bin[rng.integers(len(in_bin))])
        rest = [e for e in unstable if e not in picked]
        while len(picked) < N_UNSTABLE and rest:
            picked.append(rest.pop(rng.integers(len(rest))))
        selected += picked

    used = {el for entry in selected for el in entry["composition"]}
    for element in sorted(used):
        ground_states[element]["data"]["e_above_hull"] = 0.0
        selected.append(ground_states[element])
    return selected


def to_atoms(entry: dict) -> Atoms:
    """
    Convert an Alexandria entry to ASE atoms with the reference data.

    Parameters
    ----------
    entry
        Alexandria entry.

    Returns
    -------
    Atoms
        Structure, with ``mat_id``, ``dft_energy`` and ``e_above_hull`` in ``info``.
    """
    structure = entry["structure"]
    atoms = Atoms(
        symbols=[site["species"][0]["element"] for site in structure["sites"]],
        scaled_positions=[site["abc"] for site in structure["sites"]],
        cell=structure["lattice"]["matrix"],
        pbc=True,
    )
    atoms.info = {
        "mat_id": entry["data"]["mat_id"],
        "dft_energy": entry["data"]["energy_total"],
        "e_above_hull": entry["data"]["e_above_hull"],
    }
    return atoms


def build_archive(out_dir: Path = BENCHMARK_DATA_DIR) -> Path:
    """
    Download the Alexandria data and pack the selected structures into the archive.

    Parameters
    ----------
    out_dir
        Directory in which to write the archive. Default is the ML-PEG cache.

    Returns
    -------
    Path
        Path to the archive.
    """
    entries = load_entries()
    selected = select_entries(entries, energies_above_hull(entries))
    print(f"Selected {len(selected)} of {len(entries)} unary and binary entries")

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{ARCHIVE_NAME}.zip"
    with tempfile.TemporaryDirectory() as tmp:
        structures = Path(tmp) / "structures.extxyz"
        write(structures, [to_atoms(entry) for entry in selected])
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"{ARCHIVE_NAME}/README.txt", README)
            zip_file.write(structures, f"{ARCHIVE_NAME}/structures.extxyz")

    print(f"Written {archive}")
    return archive


if __name__ == "__main__":
    build_archive()
