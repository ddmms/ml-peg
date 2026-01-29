"""Run lanthanide isomer complex energy calculations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ase.io import read
import pytest

from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = CALCS_ROOT / "lanthanides" / "isomer_complexes" / "outputs"
STRUCT_ENV_VAR = "ML_PEG_LANTHANIDE_STRUCTURES"
KCAL_PER_EV = 23.060547


def _resolve_structure_root() -> Path | None:
    """
    Resolve the root directory containing isomer structures.

    Returns
    -------
    Path | None
        Structure root path if found, otherwise ``None``.
    """
    env_path = os.environ.get(STRUCT_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return None


def _load_isomer_entries(struct_root: Path) -> list[dict[str, Any]]:
    """
    Load isomer entries from the structure root.

    Parameters
    ----------
    struct_root
        Root directory containing system/iso*/orca.xyz and optional .CHRG/.UHF.

    Returns
    -------
    list[dict[str, Any]]
        Entry dictionaries with system, isomer, xyz path, charge, multiplicity.
    """
    entries: list[dict[str, Any]] = []
    for system_dir in sorted(struct_root.glob("*")):
        if not system_dir.is_dir():
            continue
        for iso_dir in sorted(system_dir.glob("iso*")):
            xyz_path = iso_dir / "orca.xyz"
            if not xyz_path.exists():
                continue
            charge_path = iso_dir / ".CHRG"
            uhf_path = iso_dir / ".UHF"
            charge = (
                float(charge_path.read_text().strip()) if charge_path.exists() else 0.0
            )
            multiplicity = (
                int(float(uhf_path.read_text().strip())) if uhf_path.exists() else 1
            )
            entries.append(
                {
                    "system": system_dir.name,
                    "isomer": iso_dir.name,
                    "xyz": xyz_path,
                    "charge": charge,
                    "multiplicity": multiplicity,
                }
            )
    return entries


def _write_model_csv(
    model_name: str, rows: list[dict[str, Any]], out_dir: Path
) -> None:
    """
    Write a per-model CSV of isomer energies.

    Parameters
    ----------
    model_name
        Model identifier.
    rows
        Rows containing per-isomer energies and metadata.
    out_dir
        Output directory for the CSV file.
    """
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "isomer_energies.csv"
    fieldnames = [
        "model",
        "system",
        "isomer",
        "energy_ev",
        "energy_kcal",
        "rel_energy_kcal",
        "charge",
        "multiplicity",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


@pytest.mark.parametrize("mlip", MODELS.items())
def test_isomer_complexes(mlip: tuple[str, Any]) -> None:
    """
    Run single-point energy calculations for lanthanide isomer complexes.

    Parameters
    ----------
    mlip
        Model name and MLIP calculator wrapper.
    """
    struct_root = _resolve_structure_root()
    if struct_root is None or not struct_root.exists():
        pytest.skip(
            "No lanthanide structure root found. "
            "Set ML_PEG_LANTHANIDE_STRUCTURES to the isomer_structures path."
        )

    entries = _load_isomer_entries(struct_root)
    if not entries:
        pytest.skip(f"No isomer structures found under {struct_root}.")

    model_name, model = mlip
    calc = model.get_calculator()

    results: list[dict[str, Any]] = []
    for entry in entries:
        atoms = read(entry["xyz"])
        atoms.info["charge"] = entry["charge"]
        atoms.info["spin_multiplicity"] = entry["multiplicity"]
        atoms.info["spin"] = entry["multiplicity"]
        atoms.calc = calc
        energy_ev = float(atoms.get_potential_energy())
        energy_kcal = energy_ev * KCAL_PER_EV
        results.append(
            {
                "model": model_name,
                "system": entry["system"],
                "isomer": entry["isomer"],
                "energy_ev": energy_ev,
                "energy_kcal": energy_kcal,
                "charge": entry["charge"],
                "multiplicity": entry["multiplicity"],
            }
        )

    results.sort(key=lambda row: (row["model"], row["system"], row["isomer"]))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (row["model"], row["system"])
        grouped.setdefault(key, []).append(row)

    for rows in grouped.values():
        min_energy = min(row["energy_kcal"] for row in rows)
        for row in rows:
            row["rel_energy_kcal"] = row["energy_kcal"] - min_energy

    _write_model_csv(model_name, results, OUT_PATH / model_name)
