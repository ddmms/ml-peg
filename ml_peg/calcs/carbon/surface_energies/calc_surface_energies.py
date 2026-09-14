"""Run calculations for carbon surface energies benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
import pytest

from ml_peg.calcs.carbon.utils.carbon_utils import (
    energy_at,
    get_dispersion_variants,
    load_carbon_systems,
    write_variant_frame,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"

AMORPHOUS_SYSTEM = "Amorphous_Bulk_Unrelaxed_2"
EXCLUDED_SYSTEMS = ("Diamond_110",)

EV_ANGSTROM2_TO_J_M2 = 16.0218


def surface_area(atoms: Atoms) -> float:
    """
    Get the cross-sectional area spanned by the first two cell vectors.

    Parameters
    ----------
    atoms
        Structure to measure the cell of.

    Returns
    -------
    float
        Magnitude of the cross product of the first two cell vectors, in Angstrom^2.
    """
    return float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))


def group_amorphous_frames(frames: list[Atoms]) -> dict[int, dict[str, Any]]:
    """
    Group amorphous reference frames by config, separating bulk from slab roles.

    Parameters
    ----------
    frames
        All frames read from the amorphous system's reference.xyz.

    Returns
    -------
    dict[int, dict[str, Any]]
        Mapping of amorphous_config to its bulk frame and list of slab frames.
    """
    by_config: dict[int, dict[str, Any]] = {}
    for atoms in frames:
        entry = by_config.setdefault(
            atoms.info["amorphous_config"], {"bulk": None, "slabs": []}
        )
        if atoms.info["structure_role"] == "bulk":
            entry["bulk"] = atoms
        else:
            entry["slabs"].append(atoms)
    return by_config


def amorphous_reference_surface_energy(by_config: dict[int, dict[str, Any]]) -> float:
    """
    Get the DFT ensemble-average as-cut surface energy across amorphous configs.

    Parameters
    ----------
    by_config
        Amorphous reference frames grouped by config, from `group_amorphous_frames`.

    Returns
    -------
    float
        Mean surface energy in J/m^2 across every (config, slab) pair.
    """
    energies = []
    for entry in by_config.values():
        bulk = entry["bulk"]
        if bulk is None or not entry["slabs"]:
            warn(
                "Skipping amorphous config with missing bulk or slab frames",
                stacklevel=2,
            )
            continue
        e_bulk = bulk.info["REF_energy"]
        area = surface_area(bulk)
        for slab in entry["slabs"]:
            energies.append(
                0.5 * (slab.info["REF_energy"] - e_bulk) / area * EV_ANGSTROM2_TO_J_M2
            )
    return float(np.mean(energies)) if energies else np.nan


def prepare_reference(data_dir: Path, system: str) -> dict[str, Any]:
    """
    Read reference geometry and precompute fixed reference surface energies.

    Parameters
    ----------
    data_dir
        Path to the extracted surface_energies benchmark data.
    system
        Name of the surface system.

    Returns
    -------
    dict[str, Any]
        Reference values and structures needed to compute `system`'s surface energy.
    """
    frames = read(data_dir / system / "reference.xyz", index=":")
    if system == AMORPHOUS_SYSTEM:
        by_config = group_amorphous_frames(frames)
        return {
            "is_amorphous": True,
            "by_config": by_config,
            "ref_as_cut_surface_energy_j_m2": amorphous_reference_surface_energy(
                by_config
            ),
            "ref_relaxed_surface_energy_j_m2": np.nan,
        }

    bulk, as_cut, relaxed = frames
    area = surface_area(bulk)
    return {
        "is_amorphous": False,
        "bulk": bulk,
        "as_cut": as_cut,
        "relaxed": relaxed,
        "area": area,
        "ref_as_cut_surface_energy_j_m2": (
            0.5
            * (as_cut.info["REF_energy"] - bulk.info["REF_energy"])
            / area
            * EV_ANGSTROM2_TO_J_M2
        ),
        "ref_relaxed_surface_energy_j_m2": (
            0.5
            * (relaxed.info["REF_energy"] - bulk.info["REF_energy"])
            / area
            * EV_ANGSTROM2_TO_J_M2
        ),
    }


def evaluate_facet(
    ref: dict[str, Any], calc: Calculator, system: str
) -> tuple[Atoms, float, float]:
    """
    Single-point evaluate a facet's bulk, as-cut and relaxed reference geometries.

    Parameters
    ----------
    ref
        Reference values and structures for `system`, from `prepare_reference`.
    calc
        ASE calculator to attach.
    system
        Name of the surface system, for warning messages.

    Returns
    -------
    tuple[Atoms, float, float]
        The relaxed reference structure, as-cut surface energy, and relaxed surface
        energy (J/m^2, `np.nan` on failure).
    """
    e_bulk = energy_at(ref["bulk"], calc, f"{system} bulk")
    e_as_cut = energy_at(ref["as_cut"], calc, f"{system} as-cut")
    e_relaxed = energy_at(ref["relaxed"], calc, f"{system} relaxed")

    as_cut_j_m2 = 0.5 * (e_as_cut - e_bulk) / ref["area"] * EV_ANGSTROM2_TO_J_M2
    relaxed_j_m2 = 0.5 * (e_relaxed - e_bulk) / ref["area"] * EV_ANGSTROM2_TO_J_M2

    return ref["relaxed"].copy(), as_cut_j_m2, relaxed_j_m2


def evaluate_amorphous(
    by_config: dict[int, dict[str, Any]], calc: Calculator
) -> tuple[Atoms, float]:
    """
    Single-point evaluate as-cut surface energies across the amorphous ensemble.

    Parameters
    ----------
    by_config
        Amorphous reference frames grouped by config, from `group_amorphous_frames`.
    calc
        ASE calculator to attach.

    Returns
    -------
    tuple[Atoms, float]
        A representative slab structure, and the ensemble-mean as-cut surface energy
        (J/m^2, `np.nan` if every evaluation failed).
    """
    representative = None
    energies = []
    for entry in by_config.values():
        bulk_reference = entry["bulk"]
        slab_references = entry["slabs"]
        if bulk_reference is None or not slab_references:
            warn(
                "Skipping amorphous config with missing bulk or slab frames",
                stacklevel=2,
            )
            continue

        area = surface_area(bulk_reference)
        e_bulk = energy_at(bulk_reference, calc, "amorphous bulk")
        for slab_reference in slab_references:
            if representative is None:
                representative = slab_reference.copy()
            e_slab = energy_at(slab_reference, calc, "amorphous slab")
            energy = 0.5 * (e_slab - e_bulk) / area * EV_ANGSTROM2_TO_J_M2
            if not np.isnan(energy):
                energies.append(energy)

    if representative is None:
        representative = Atoms(
            "C", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=True
        )
    as_cut_j_m2 = float(np.mean(energies)) if energies else np.nan
    return representative, as_cut_j_m2


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_surface_energies(mlip: tuple[str, Any]) -> None:
    """
    Run diamond, graphite and amorphous carbon surface energy benchmark.

    Parameters
    ----------
    mlip
        Name of model and model instance to get calculator from.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_dir, all_systems = load_carbon_systems("surface_energies")
    systems = [system for system in all_systems if system not in EXCLUDED_SYSTEMS]

    references = {system: prepare_reference(data_dir, system) for system in systems}

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    variant_calcs = get_dispersion_variants(model, calc)
    for variant_index, variant_calc in enumerate(variant_calcs):
        for system in systems:
            ref = references[system]
            if ref["is_amorphous"]:
                atoms, as_cut_j_m2 = evaluate_amorphous(ref["by_config"], variant_calc)
                relaxed_j_m2 = np.nan
            else:
                atoms, as_cut_j_m2, relaxed_j_m2 = evaluate_facet(
                    ref, variant_calc, system
                )

            atoms.info["as_cut_surface_energy_j_m2"] = as_cut_j_m2
            atoms.info["relaxed_surface_energy_j_m2"] = relaxed_j_m2
            atoms.info["ref_as_cut_surface_energy_j_m2"] = ref[
                "ref_as_cut_surface_energy_j_m2"
            ]
            atoms.info["ref_relaxed_surface_energy_j_m2"] = ref[
                "ref_relaxed_surface_energy_j_m2"
            ]

            write_variant_frame(
                write_dir / f"{system}.extxyz", atoms, variant_index, len(variant_calcs)
            )
