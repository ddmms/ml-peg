"""Analyse carbon surface energies benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.carbon.surface_energies.calc_surface_energies import (
    AMORPHOUS_SYSTEM,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "carbon" / "surface_energies" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "surface_energies"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.extxyz",
    index=0,
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
    include_filenames=True,
)
SYSTEMS = INFO["filenames"]
RELAXED_SYSTEMS = [system for system in SYSTEMS if system != AMORPHOUS_SYSTEM]

AS_CUT_ENTRIES = [(system, "as_cut_surface_energy_j_m2") for system in SYSTEMS]
RELAXED_ENTRIES = [
    (system, "relaxed_surface_energy_j_m2") for system in RELAXED_SYSTEMS
]


def gather_metric_values(
    entries: list[tuple[str, str]], frame_index: int
) -> dict[str, list]:
    """
    Gather reference and predicted values for a set of (system, info key) entries.

    Parameters
    ----------
    entries
        System and info-key pairs identifying which structures and fields to read.
    frame_index
        Extxyz frame to read: 0 for the plain calculator, 1 for D3-corrected (same
        as 0 for models already trained on dispersion).

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted values, ordered as `entries`. A missing
        file appends `np.nan`, so it reaches the aggregate and voids that model's
        score, same as any other failed evaluation.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        for system, key in entries:
            struct_file = model_dir / f"{system}.extxyz"
            if not struct_file.is_file():
                results[model_name].append(np.nan)
                continue

            atoms = read(struct_file, index=frame_index)
            results[model_name].append(atoms.info.get(key, np.nan))
            if not ref_stored:
                results["ref"].append(atoms.info[f"ref_{key}"])

        if not ref_stored:
            if len(results["ref"]) == len(entries):
                ref_stored = True
            else:
                results["ref"] = []

    return results


def mae_or_nan(results: dict[str, list], model_name: str) -> float:
    """
    Get mean absolute error for one model, or `np.nan` if no reference is available.

    Parameters
    ----------
    results
        Reference and per-model predicted values, from `gather_metric_values`.
    model_name
        Name of the model.

    Returns
    -------
    float
        Mean absolute error, or `np.nan` if `results["ref"]` is empty.
    """
    if not results["ref"]:
        return np.nan
    return mae(results["ref"], results[model_name])


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_surface_energies_relaxed.json",
    title="Relaxed surface energy (D3-corrected)",
    x_label="Predicted relaxed surface energy / J per m^2",
    y_label="Reference relaxed surface energy / J per m^2",
    hoverdata={"System": RELAXED_SYSTEMS},
)
def relaxed_surface_energy() -> dict[str, list]:
    """
    Get reference and D3-corrected predicted relaxed surface energy.

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted relaxed surface energy, in J/m^2, for the
        two surfaces with a relaxed reference.
    """
    results = gather_metric_values(RELAXED_ENTRIES, frame_index=1)

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        structs_dir = OUT_PATH / model_name
        for system in RELAXED_SYSTEMS:
            struct_file = model_dir / f"{system}.extxyz"
            if not struct_file.is_file():
                continue
            atoms = read(struct_file, index=1)
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", atoms)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "surface_energies_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(relaxed_surface_energy: dict[str, list]) -> dict[str, dict]:
    """
    Get all surface energies metrics, D3-corrected and uncorrected.

    Parameters
    ----------
    relaxed_surface_energy
        D3-corrected reference and predicted relaxed surface energy.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    as_cut_d3 = gather_metric_values(AS_CUT_ENTRIES, frame_index=1)
    as_cut_plain = gather_metric_values(AS_CUT_ENTRIES, frame_index=0)
    relaxed_plain = gather_metric_values(RELAXED_ENTRIES, frame_index=0)

    return {
        "As-cut surface energy MAE (D3)": {m: mae_or_nan(as_cut_d3, m) for m in MODELS},
        "As-cut surface energy MAE": {m: mae_or_nan(as_cut_plain, m) for m in MODELS},
        "Relaxed surface energy MAE (D3)": {
            m: mae_or_nan(relaxed_surface_energy, m) for m in MODELS
        },
        "Relaxed surface energy MAE": {m: mae_or_nan(relaxed_plain, m) for m in MODELS},
    }


def test_surface_energies(metrics: dict[str, dict]) -> None:
    """
    Run surface energies test.

    Parameters
    ----------
    metrics
        All surface energies metrics.
    """
    return
