"""Analyse carbon lattice parameters benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae, mape
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.carbon.lattice_parameters.calc_lattice_parameters import (
    MOLECULES,
    REPEAT_FACTORS,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "carbon" / "lattice_parameters" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "lattice_parameters"

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
SYSTEMS_NO_GRAPHITE = [system for system in SYSTEMS if system != "Graphite"]

BOND_LENGTH_KEYS = ("bond_length_1_angstrom", "bond_length_2_angstrom")

LATTICE_PARAMETER_ENTRIES = [
    (system, f"lattice_{axis}_angstrom")
    for system, factors in REPEAT_FACTORS.items()
    for axis in factors
]
BOND_LENGTH_ENTRIES = [
    (system, key) for system in MOLECULES for key in BOND_LENGTH_KEYS
]
ENERGY_ABOVE_GRAPHITE_ENTRIES = [
    (system, "energy_above_graphite_ev_per_atom") for system in SYSTEMS_NO_GRAPHITE
]


def gather_metric_values(
    entries: list[tuple[str, str]], frame_index: int
) -> tuple[dict[str, list], dict[str, list]]:
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
    tuple[dict[str, list], dict[str, list]]
        Reference and per-model predicted values, and per-model flags marking which
        entries belong in an aggregate, both ordered as `entries`. A missing file is
        flagged True so its NaN reaches the aggregate and voids the score.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    keep = {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        for system, key in entries:
            struct_file = model_dir / f"{system}.extxyz"
            if not struct_file.is_file():
                results[model_name].append(np.nan)
                keep[model_name].append(True)
                continue

            atoms = read(struct_file, index=frame_index)
            is_converged = atoms.info.get("converged", True)
            results[model_name].append(
                atoms.info.get(key, np.nan) if is_converged else np.nan
            )
            keep[model_name].append(is_converged)
            if not ref_stored:
                results["ref"].append(atoms.info[f"ref_{key}"])

        if not ref_stored:
            if len(results["ref"]) == len(entries):
                ref_stored = True
            else:
                results["ref"] = []

    return results, keep


def drop_unconverged(ref: list, prediction: list, keep: list) -> tuple[list, list]:
    """
    Drop entries not marked for inclusion from a paired ref/prediction list.

    Parameters
    ----------
    ref
        Reference values, empty if no model produced a complete set.
    prediction
        Predicted values, same order as `ref`.
    keep
        Flag for each entry, same order as `ref`.

    Returns
    -------
    tuple[list, list]
        `ref` and `prediction` with excluded entries removed, or a single NaN pair
        if nothing remains.
    """
    if not ref:
        return [np.nan], [np.nan]

    pairs = [
        (r, p) for r, p, include in zip(ref, prediction, keep, strict=True) if include
    ]
    if not pairs:
        return [np.nan], [np.nan]
    ref_kept, pred_kept = zip(*pairs, strict=True)
    return list(ref_kept), list(pred_kept)


def get_convergence_rate(model_name: str) -> float:
    """
    Get the percentage of relaxed frames that converged for one model.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    float
        Percentage of frames with `converged=True`, or `np.nan` if none exist.
    """
    model_dir = CALC_PATH / model_name
    flags = [
        read(struct_file, index=frame_index).info.get("converged", True)
        for system in SYSTEMS
        if (struct_file := model_dir / f"{system}.extxyz").is_file()
        for frame_index in (0, 1)
    ]
    return 100 * sum(flags) / len(flags) if flags else np.nan


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_parameters_energy_above_graphite.json",
    title="Energy above graphite (D3-corrected)",
    x_label="Predicted energy above graphite / eV per atom",
    y_label="Reference energy above graphite / eV per atom",
    hoverdata={"System": SYSTEMS_NO_GRAPHITE},
)
def energy_above_graphite() -> dict[str, list]:
    """
    Get reference and D3-corrected predicted energy above graphite.

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted energy above graphite, in eV per atom.
    """
    results, _ = gather_metric_values(ENERGY_ABOVE_GRAPHITE_ENTRIES, frame_index=1)

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        structs_dir = OUT_PATH / model_name
        for system in SYSTEMS_NO_GRAPHITE:
            struct_file = model_dir / f"{system}.extxyz"
            if not struct_file.is_file():
                continue
            atoms = read(struct_file, index=1)
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", atoms)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "lattice_parameters_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(energy_above_graphite: dict[str, list]) -> dict[str, dict]:
    """
    Get all lattice parameters metrics, D3-corrected and uncorrected.

    Parameters
    ----------
    energy_above_graphite
        D3-corrected reference and predicted energy above graphite.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    lattice_d3, lattice_d3_ok = gather_metric_values(
        LATTICE_PARAMETER_ENTRIES, frame_index=1
    )
    lattice_plain, lattice_plain_ok = gather_metric_values(
        LATTICE_PARAMETER_ENTRIES, frame_index=0
    )
    bond_d3, bond_d3_ok = gather_metric_values(BOND_LENGTH_ENTRIES, frame_index=1)
    bond_plain, bond_plain_ok = gather_metric_values(BOND_LENGTH_ENTRIES, frame_index=0)
    energy_d3, energy_d3_ok = gather_metric_values(
        ENERGY_ABOVE_GRAPHITE_ENTRIES, frame_index=1
    )
    energy_plain, energy_plain_ok = gather_metric_values(
        ENERGY_ABOVE_GRAPHITE_ENTRIES, frame_index=0
    )

    return {
        "Lattice parameter MAPE (D3)": {
            m: mape(
                *drop_unconverged(lattice_d3["ref"], lattice_d3[m], lattice_d3_ok[m])
            )
            for m in MODELS
        },
        "Lattice parameter MAPE": {
            m: mape(
                *drop_unconverged(
                    lattice_plain["ref"], lattice_plain[m], lattice_plain_ok[m]
                )
            )
            for m in MODELS
        },
        "Bond length MAPE (D3)": {
            m: mape(*drop_unconverged(bond_d3["ref"], bond_d3[m], bond_d3_ok[m]))
            for m in MODELS
        },
        "Bond length MAPE": {
            m: mape(
                *drop_unconverged(bond_plain["ref"], bond_plain[m], bond_plain_ok[m])
            )
            for m in MODELS
        },
        "Energy above graphite MAE (D3)": {
            m: 1000
            * mae(*drop_unconverged(energy_d3["ref"], energy_d3[m], energy_d3_ok[m]))
            for m in MODELS
        },
        "Energy above graphite MAE": {
            m: 1000
            * mae(
                *drop_unconverged(
                    energy_plain["ref"], energy_plain[m], energy_plain_ok[m]
                )
            )
            for m in MODELS
        },
        "Convergence": {m: get_convergence_rate(m) for m in MODELS},
    }


def test_lattice_parameters(metrics: dict[str, dict]) -> None:
    """
    Run lattice parameters test.

    Parameters
    ----------
    metrics
        All lattice parameters metrics.
    """
    return
