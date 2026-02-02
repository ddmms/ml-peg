"""Run calculations for high-pressure crystal relaxation benchmark."""

from __future__ import annotations

import bz2
from copy import copy
import json
from pathlib import Path
import random
from typing import Any
import urllib.request

from ase import Atoms
from ase.constraints import FixSymmetry
from ase.units import GPa
from janus_core.calculations.geom_opt import GeomOpt
import pandas as pd
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
import pytest
from tqdm import tqdm

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# URL for Alexandria database
ALEXANDRIA_BASE_URL = "https://alexandria.icams.rub.de/data/pbe/benchmarks/pressure"

# Local cache directory for downloaded data
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Pressure conditions in GPa
PRESSURES = [0, 25, 50, 75, 100, 125, 150]
PRESSURE_LABELS = ["P000", "P025", "P050", "P075", "P100", "P125", "P150"]


# Relaxation parameters
FMAX = 0.0002  # eV/A - tight convergence
MAX_STEPS = 500
RANDOM_SEED = 42

# Number of structures to use for testing
N_STRUCTURES = 100


def download_pressure_data(pressure_label: str) -> Path:
    """
    Download pressure data from Alexandria database if not already cached.

    Parameters
    ----------
    pressure_label
        Pressure label (e.g., "P000", "P025").

    Returns
    -------
    Path
        Path to the downloaded/cached file.
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    local_path = DATA_PATH / f"{pressure_label}.json.bz2"

    if not local_path.exists():
        url = f"{ALEXANDRIA_BASE_URL}/{pressure_label}.json.bz2"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")

    return local_path


def load_structures(
    pressure_label: str, n_structures: int = N_STRUCTURES
) -> list[dict]:
    """
    Load structures using P000 starting structures and pressure-specific references.

    Downloads data from Alexandria database if not already cached locally.

    Parameters
    ----------
    pressure_label
        Pressure label (e.g., "P000", "P025") used for reference values.
    n_structures
        Number of structures to load. Default is N_STRUCTURES.

    Returns
    -------
    list[dict]
        List of structure dictionaries with atoms, mat_id, and reference values.
    """
    start_json_path = download_pressure_data("P000")
    ref_json_path = download_pressure_data(pressure_label)

    with bz2.open(start_json_path, "rt") as f:
        start_data = json.load(f)

    with bz2.open(ref_json_path, "rt") as f:
        ref_data = json.load(f)

    ref_map: dict[str, ComputedStructureEntry] = {}
    for entry_dict in ref_data["entries"]:
        entry = ComputedStructureEntry.from_dict(entry_dict)
        ref_map[entry.data["mat_id"]] = entry

    adaptor = AseAtomsAdaptor()
    structures = []

    start_entries = start_data["entries"]
    if n_structures > len(start_entries):
        raise ValueError(
            f"Requested {n_structures} structures but only"
            f" {len(start_entries)} available"
        )

    rng = random.Random(RANDOM_SEED)
    selected_indices = sorted(rng.sample(range(len(start_entries)), k=n_structures))

    for idx in selected_indices:
        entry_dict = start_entries[idx]
        entry = ComputedStructureEntry.from_dict(entry_dict)
        mat_id = entry.data["mat_id"]
        ref_entry = ref_map.get(mat_id)
        if ref_entry is None:
            raise ValueError(
                f"Missing reference entry for {mat_id} at {pressure_label}"
            )

        atoms = adaptor.get_atoms(entry.structure)
        n_atoms = len(ref_entry.structure)

        structures.append(
            {
                "mat_id": mat_id,
                "atoms": atoms,
                "ref_energy_per_atom": ref_entry.energy / n_atoms,
                "ref_volume_per_atom": ref_entry.structure.volume / n_atoms,
            }
        )

    return structures


def relax_with_pressure(
    atoms: Atoms,
    pressure_gpa: float,
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
) -> tuple[Atoms | None, bool, float | None]:
    """
    Relax structure under specified pressure using janus-core GeomOpt.

    Parameters
    ----------
    atoms
        ASE Atoms object with calculator attached.
    pressure_gpa
        Pressure in GPa.
    fmax
        Maximum force tolerance in eV/A.
    max_steps
        Maximum number of optimization steps.

    Returns
    -------
    tuple[Atoms | None, bool, float | None]
        Relaxed atoms (or None if failed), convergence status, enthalpy per atom.
    """
    try:
        converged = False
        counter = 0
        # repeat up to 3 times if not converged to be consistent with reference
        while counter < 3 and not converged:
            atoms.set_constraint(
                FixSymmetry(
                    atoms,
                )
            )
            geom_opt = GeomOpt(
                struct=atoms,
                fmax=fmax,
                steps=max_steps,
                filter_kwargs={"scalar_pressure": pressure_gpa},
                calc_kwargs={"default_dtype": "float64"},
            )
            geom_opt.run()
            relaxed = geom_opt.struct
            # asses forces to determine convergence
            max_force = max(relaxed.get_forces().flatten(), key=abs)
            if max_force < fmax:
                converged = True
            counter += 1
            atoms = relaxed
    except Exception as e:
        print(f"Relaxation failed: {e}")
        return None, False, None
    if not converged:
        return None, False, None
    # Calculate enthalpy: H = E + PV
    energy = relaxed.get_potential_energy()
    volume = relaxed.get_volume()
    enthalpy = energy + pressure_gpa * GPa * volume
    n_atoms = len(relaxed)

    return relaxed, converged, enthalpy / n_atoms


@pytest.mark.parametrize("mlip", MODELS.items())
@pytest.mark.parametrize("pressure_idx", range(len(PRESSURES)))
def test_high_pressure_relaxation(mlip: tuple[str, Any], pressure_idx: int) -> None:
    """
    Run high-pressure relaxation benchmark.

    Parameters
    ----------
    mlip
        Tuple of model name and model object.
    pressure_idx
        Index into PRESSURES list.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    pressure_gpa = PRESSURES[pressure_idx]
    pressure_label = PRESSURE_LABELS[pressure_idx]

    # Load structures (downloads from Alexandria if not cached)
    structures = load_structures(pressure_label)

    results = []
    for struct_data in tqdm(
        structures, desc=f"{model_name} @ {pressure_gpa} GPa", leave=False
    ):
        atoms = struct_data["atoms"].copy()
        atoms.calc = copy(calc)
        mat_id = struct_data["mat_id"]

        relaxed_atoms, converged, enthalpy_per_atom = relax_with_pressure(
            atoms, pressure_gpa
        )

        if relaxed_atoms is not None:
            pred_volume = relaxed_atoms.get_volume() / len(relaxed_atoms)
        else:
            pred_volume = None

        results.append(
            {
                "mat_id": mat_id,
                "pressure_gpa": pressure_gpa,
                "ref_volume_per_atom": struct_data["ref_volume_per_atom"],
                "ref_energy_per_atom": struct_data["ref_energy_per_atom"],
                "pred_volume_per_atom": pred_volume,
                "pred_energy_per_atom": enthalpy_per_atom,
                "converged": converged,
            }
        )

    # Save results
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"results_{pressure_label}.csv", index=False)
