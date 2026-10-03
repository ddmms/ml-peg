"""
Analyse carbon lattice parameters benchmark.

A relaxation that does not reach its force threshold is reported as NaN for every
quantity measured from it, and excluded from the error metrics rather than voiding
them: a warning is raised for each such structure, and the Convergence metric
penalises the model instead.
"""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_scatter
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
    write_structs=True,
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
    entries: list[tuple[str, str]],
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Gather reference and predicted values for a set of (system, info key) entries.

    Unconverged relaxations are reported as NaN and flagged for exclusion from the
    aggregate metrics.

    Parameters
    ----------
    entries
        System and info-key pairs identifying which structures and fields to read.

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

            atoms = read(struct_file)
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
    Get the percentage of relaxed structures that converged for one model.

    Warns for each unconverged structure, whose values are reported as NaN.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    float
        Percentage of structures with `converged=True`, or `np.nan` if none exist.
    """
    model_dir = CALC_PATH / model_name
    flags = []
    for system in SYSTEMS:
        struct_file = model_dir / f"{system}.extxyz"
        if not struct_file.is_file():
            continue
        converged = read(struct_file).info.get("converged", True)
        if not converged:
            warn(
                f"{model_name}: {system} relaxation did not converge, so its values"
                " are reported as NaN and excluded from the error metrics",
                stacklevel=2,
            )
        flags.append(converged)
    return 100 * sum(flags) / len(flags) if flags else np.nan


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_parameters_lattice_parameter.json",
    title="Lattice parameters",
    x_label="Predicted lattice parameter / Å",
    y_label="Reference lattice parameter / Å",
    hoverdata={
        "System": [system for system, _ in LATTICE_PARAMETER_ENTRIES],
        "Parameter": [key for _, key in LATTICE_PARAMETER_ENTRIES],
    },
)
def lattice_parameter() -> dict[str, list]:
    """
    Get reference and predicted lattice parameters.

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted lattice parameters, in Å.
    """
    results, _ = gather_metric_values(LATTICE_PARAMETER_ENTRIES)
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_parameters_bond_length.json",
    title="Bond lengths",
    x_label="Predicted mean bond length / Å",
    y_label="Reference mean bond length / Å",
    hoverdata={
        "System": [system for system, _ in BOND_LENGTH_ENTRIES],
        "Shell": [key for _, key in BOND_LENGTH_ENTRIES],
    },
)
def bond_length() -> dict[str, list]:
    """
    Get reference and predicted first and second shell bond lengths.

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted mean bond lengths, in Å.
    """
    results, _ = gather_metric_values(BOND_LENGTH_ENTRIES)
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_parameters_energy_above_graphite.json",
    title="Energy above graphite",
    x_label="Predicted energy above graphite / eV per atom",
    y_label="Reference energy above graphite / eV per atom",
    hoverdata={"System": SYSTEMS_NO_GRAPHITE},
)
def energy_above_graphite() -> dict[str, list]:
    """
    Get reference and predicted energy above graphite.

    Returns
    -------
    dict[str, list]
        Reference and per-model predicted energy above graphite, in eV per atom.
    """
    results, _ = gather_metric_values(ENERGY_ABOVE_GRAPHITE_ENTRIES)
    return results


@pytest.fixture
@plot_scatter(
    filename=OUT_PATH / "figure_lattice_parameters_convergence.json",
    title="Residual force after relaxation",
    x_label="System",
    y_label="Max residual force / force threshold",
    hoverdata={"System": SYSTEMS},
    horizontal_lines=[{"y": 1.0, "name": "Converged below", "dash": "dash"}],
)
def convergence() -> dict[str, list]:
    """
    Get each model's max residual force relative to its convergence threshold.

    Returns
    -------
    dict[str, list]
        Per-model system names and residual force ratios, ordered as `SYSTEMS`.
    """
    results = {}
    for model_name in MODELS:
        ratios = []
        for system in SYSTEMS:
            struct_file = CALC_PATH / model_name / f"{system}.extxyz"
            if not struct_file.is_file():
                ratios.append(np.nan)
                continue
            info = read(struct_file).info
            ratios.append(
                info.get("max_force_ev_per_angstrom", np.nan)
                / info.get("fmax_ev_per_angstrom", np.nan)
            )
        results[model_name] = [SYSTEMS, ratios]
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "lattice_parameters_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    lattice_parameter: dict[str, list],
    bond_length: dict[str, list],
    energy_above_graphite: dict[str, list],
    convergence: dict[str, list],
) -> dict[str, dict]:
    """
    Get all lattice parameters metrics.

    Unconverged entries are excluded from the error metrics, and counted against
    the model in Convergence.

    Parameters
    ----------
    lattice_parameter
        Reference and predicted lattice parameters.
    bond_length
        Reference and predicted bond lengths.
    energy_above_graphite
        Reference and predicted energy above graphite.
    convergence
        Residual force ratios.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    _, lattice_ok = gather_metric_values(LATTICE_PARAMETER_ENTRIES)
    _, bond_ok = gather_metric_values(BOND_LENGTH_ENTRIES)
    _, energy_ok = gather_metric_values(ENERGY_ABOVE_GRAPHITE_ENTRIES)

    return {
        "Lattice parameter MAPE": {
            m: mape(
                *drop_unconverged(
                    lattice_parameter["ref"], lattice_parameter[m], lattice_ok[m]
                )
            )
            for m in MODELS
        },
        "Bond length MAPE": {
            m: mape(*drop_unconverged(bond_length["ref"], bond_length[m], bond_ok[m]))
            for m in MODELS
        },
        "Energy above graphite MAE": {
            m: 1000
            * mae(
                *drop_unconverged(
                    energy_above_graphite["ref"], energy_above_graphite[m], energy_ok[m]
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
