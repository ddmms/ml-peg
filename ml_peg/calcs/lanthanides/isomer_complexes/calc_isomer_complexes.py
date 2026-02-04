"""Run lanthanide isomer complex energy calculations."""

from __future__ import annotations

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
KCAL_PER_EV = units.mol / units.kcal


# r2SCAN-3c references (kcal/mol) from Table S4 (lanthanides only)
R2SCAN_REF: dict[str, dict[str, float]] = {
    "Ac_f1a50d": {"iso1": 0.02, "iso2": 0.0, "iso3": 3.52},
    "Ce_1d271a": {"iso1": 0.0, "iso2": 2.2, "iso3": 1.67},
    "Ce_ff6372": {"iso1": 2.47, "iso2": 7.13, "iso3": 0.0, "iso4": 2.17},
    "Eu_ff6372": {"iso1": 0.0, "iso2": 6.74},
    "La_f1a50d": {"iso1": 0.23, "iso2": 0.0, "iso3": 3.11},
    "Lu_ff6372": {"iso1": 2.15, "iso2": 12.96, "iso3": 0.0, "iso4": 2.08},
    "Nd_c5f44a": {"iso1": 0.0, "iso2": 1.61, "iso3": 0.82},
    "Sm_ed79e8": {"iso1": 2.99, "iso2": 8.97, "iso3": 0.0},
    "Th_ff6372": {"iso1": 2.13, "iso2": 8.03, "iso3": 0.0, "iso4": 1.23},
}


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
        if system_dir.name not in R2SCAN_REF:
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

    # results: list[dict[str, Any]] = []
    for entry in tqdm(entries, desc=f"Calculating energies for {model_name}"):
        atoms = read(entry["xyz"])
        atoms.info["charge"] = entry["charge"]
        atoms.info["spin_multiplicity"] = entry["multiplicity"]
        atoms.info["spin"] = entry["multiplicity"]
        atoms.calc = calc
        energy_ev = float(atoms.get_potential_energy())
        energy_kcal = energy_ev * KCAL_PER_EV

        atoms.info["model"] = model_name
        atoms.info["energy_ev"] = energy_ev
        atoms.info["energy_kcal"] = energy_kcal
        atoms.info["system"] = entry["system"]
        atoms.info["isomer"] = entry["isomer"]

        atoms.info["ref_energy_kcal"] = R2SCAN_REF.get(entry["system"], {}).get(
            entry["isomer"]
        )

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{entry['system']}_{entry['isomer']}.xyz", atoms)
