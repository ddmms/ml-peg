"""Analyse carbon nanotube formation energies benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.carbon.nanotube_formation_energies import (
    calc_nanotube_formation_energies as calc_nanotubes,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "carbon" / "nanotube_formation_energies" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "nanotube_formation_energies"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.extxyz",
    index=0,
    info_keys=["diameter_angstrom"],
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
    include_filenames=True,
)
SYSTEMS = INFO["filenames"]
CHIRALITIES = [calc_nanotubes.chirality_of(system) for system in SYSTEMS]
DIAMETERS = INFO["diameter_angstrom"]


def gather_strain_energies(frame_index: int) -> dict[str, list]:
    """
    Gather reference and predicted strain energies for every nanotube.

    Parameters
    ----------
    frame_index
        Extxyz frame to read: 0 for the plain calculator, 1 for D3-corrected (same
        as 0 for models already trained on dispersion).

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted strain energy in eV per atom, ordered
        as `SYSTEMS`. A missing file appends `np.nan`, so it reaches the
        aggregate and voids that model's score, same as any other failed
        evaluation.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        for system in SYSTEMS:
            struct_file = model_dir / f"{system}.extxyz"
            if not struct_file.is_file():
                results[model_name].append(np.nan)
                continue

            atoms = read(struct_file, index=frame_index)
            results[model_name].append(
                atoms.info.get("strain_energy_ev_per_atom", np.nan)
            )
            if not ref_stored:
                results["ref"].append(atoms.info["reference_strain_energy_ev_per_atom"])

        if not ref_stored:
            if len(results["ref"]) == len(SYSTEMS):
                ref_stored = True
            else:
                results["ref"] = []

    return results


def mae_for_chirality(
    results: dict[str, list], model_name: str, chirality: str
) -> float:
    """
    Get mean absolute error for one model on one chirality, or `np.nan`.

    Parameters
    ----------
    results
        Reference and per-model predicted strain energies, from
        `gather_strain_energies`.
    model_name
        Name of the model.
    chirality
        Chirality to select systems for, "Armchair" or "Zigzag".

    Returns
    -------
    float
        Mean absolute error over the systems belonging to `chirality`, or
        `np.nan` if `results["ref"]` is empty.
    """
    if not results["ref"]:
        return np.nan
    indices = [
        index
        for index, system_chirality in enumerate(CHIRALITIES)
        if system_chirality == chirality
    ]
    ref = [results["ref"][index] for index in indices]
    prediction = [results[model_name][index] for index in indices]
    return mae(ref, prediction)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_nanotube_formation_energies.json",
    title="Nanotube strain energy (D3-corrected)",
    x_label="Predicted strain energy / eV per atom",
    y_label="Reference strain energy / eV per atom",
    hoverdata={"Tube": SYSTEMS, "Diameter / Å": DIAMETERS},
    symbol_by=CHIRALITIES,
)
def strain_energy() -> dict[str, list]:
    """
    Get reference and D3-corrected predicted nanotube strain energy.

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted strain energy, in eV per atom, for
        all 20 nanotubes.
    """
    results = gather_strain_energies(frame_index=1)

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        structs_dir = OUT_PATH / model_name
        for system in SYSTEMS:
            struct_file = model_dir / f"{system}.extxyz"
            if not struct_file.is_file():
                continue
            atoms = read(struct_file, index=1)
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", atoms)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "nanotube_formation_energies_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(strain_energy: dict[str, list]) -> dict[str, dict]:
    """
    Get armchair and zigzag strain energy metrics, D3-corrected and uncorrected.

    Parameters
    ----------
    strain_energy
        D3-corrected reference and predicted strain energy.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    plain = gather_strain_energies(frame_index=0)

    return {
        "Armchair strain energy MAE (D3)": {
            model_name: 1000 * mae_for_chirality(strain_energy, model_name, "Armchair")
            for model_name in MODELS
        },
        "Armchair strain energy MAE": {
            model_name: 1000 * mae_for_chirality(plain, model_name, "Armchair")
            for model_name in MODELS
        },
        "Zigzag strain energy MAE (D3)": {
            model_name: 1000 * mae_for_chirality(strain_energy, model_name, "Zigzag")
            for model_name in MODELS
        },
        "Zigzag strain energy MAE": {
            model_name: 1000 * mae_for_chirality(plain, model_name, "Zigzag")
            for model_name in MODELS
        },
    }


def test_nanotube_formation_energies(metrics: dict[str, dict]) -> None:
    """
    Run nanotube formation energies test.

    Parameters
    ----------
    metrics
        All nanotube formation energies metrics.
    """
    return
