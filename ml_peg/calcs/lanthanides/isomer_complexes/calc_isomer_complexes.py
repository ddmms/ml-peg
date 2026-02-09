"""Run lanthanide isomer complex energy calculations."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = CALCS_ROOT / "lanthanides" / "isomer_complexes" / "outputs"

EXCLUDE_ELEMENTS = (89, 90)


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
            charge = int(charge_path.read_text().strip()) if charge_path.exists() else 0
            multiplicity = int(uhf_path.read_text().strip()) if uhf_path.exists() else 1
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


def get_ref_energy(data_path: Path) -> float:
    """
    Get reference energy from an xyz file.

    Files have info of the form "Coordinates from ORCA-job orca E -1221"

    Parameters
    ----------
    data_path
        Path to data.

    Returns
    -------
    float
        Loaded reference energy.
    """
    with open(data_path) as lines:
        # Skip header
        next(lines)
        line = next(lines)
        if "E" == line.split()[-2]:
            return float(line.split()[-1]) * units.Hartree

    raise ValueError("Unable to extract energy")


@pytest.mark.parametrize("mlip", MODELS.items())
def test_isomer_complexes(mlip: tuple[str, Any]) -> None:
    """
    Run single-point energy calculations for lanthanide isomer complexes.

    Parameters
    ----------
    mlip
        Model name and MLIP calculator wrapper.
    """
    # download lanthanide isomer complexes dataset
    isomer_complexes_dir = (
        download_s3_data(
            key="inputs/lanthanides/isomer_complexes/isomer_complexes.zip",
            filename="isomer_complexes.zip",
        )
        / "isomer_complexes"
    )

    entries = _load_isomer_entries(isomer_complexes_dir)
    if not entries:
        pytest.skip(f"No isomer structures found under {isomer_complexes_dir}.")

    model_name, model = mlip
    calc = model.get_calculator()

    for entry in tqdm(entries, desc=f"Calculating energies for {model_name}"):
        atoms = read(entry["xyz"])

        if any(element in EXCLUDE_ELEMENTS for element in atoms.numbers):
            print(f"Skipping {entry['xyz']}")
            continue

        atoms.info = {}
        atoms.info["charge"] = entry["charge"]
        atoms.info["spin_multiplicity"] = entry["multiplicity"]
        atoms.info["spin"] = entry["multiplicity"]
        atoms.calc = copy(calc)
        atoms.info["model_energy"] = atoms.get_potential_energy()

        atoms.info["model"] = model_name
        atoms.info["system"] = entry["system"]
        atoms.info["isomer"] = entry["isomer"]

        atoms.info["ref_energy"] = get_ref_energy(entry["xyz"])

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{entry['system']}_{entry['isomer']}.xyz", atoms)
