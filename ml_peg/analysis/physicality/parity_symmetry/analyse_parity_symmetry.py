"""Analyse parity symmetry benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.physicality.parity_symmetry.calc_parity_symmetry import (
    STRUCTURES,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "parity_symmetry" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "parity_symmetry"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

EV_TO_MEV = 1000

# Must match the metric names in metrics.yml and the column names in
# app_parity_symmetry.
LABEL = "parity transformation"

# Scores require complete results: a model only receives scores if every
# structure was evaluated successfully, rather than being scored on the
# structures that survived.
EXPECTED_STRUCTS = sorted(STRUCTURES)
N_STRUCTURES = len(EXPECTED_STRUCTS)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


def _mean_max(values: list[float]) -> tuple[float | None, float | None]:
    """
    Get the mean and max of a list of values, requiring a complete set.

    A model must evaluate every structure successfully to be scored: an
    incomplete set of values (any structure missing) or a failed (NaN)
    evaluation returns (None, None), which renders as a blank table cell rather
    than a score over a subset.

    Parameters
    ----------
    values
        Values to aggregate, one per expected structure.

    Returns
    -------
    tuple[float | None, float | None]
        Mean and max value, or (None, None) if any structure is missing or
        failed.
    """
    if len(values) != N_STRUCTURES or any(np.isnan(v) for v in values):
        return None, None
    return float(np.mean(values)), float(np.max(values))


def get_deltas(model_name: str, struct_name: str) -> tuple[float, float] | None:
    """
    Get the energy and force changes under inversion for one model/structure pair.

    Under parity (r -> -r), the energy is invariant, while forces are polar
    vectors and so change sign: atom i of the inverted structure should feel
    force -F_i, where F_i is its force in the original. Atom order is unchanged
    by the transformation, so forces are compared atom by atom.

    Parameters
    ----------
    model_name
        Name of model to compute changes for.
    struct_name
        Name of structure to compute changes for.

    Returns
    -------
    tuple[float, float] | None
        Absolute energy change (meV/atom) and largest absolute force component
        change (meV/Å), or None if outputs are missing. NaN if either
        evaluation failed.
    """
    xyz_path = CALC_PATH / model_name / f"{struct_name}.xyz"
    if not xyz_path.exists():
        return None

    frames = read(xyz_path, index=":")
    original = next((f for f in frames if not f.info["inverted"]), None)
    inverted = next((f for f in frames if f.info["inverted"]), None)
    if original is None or inverted is None:
        return None

    delta_energy = abs(
        inverted.get_potential_energy() - original.get_potential_energy()
    ) / len(original)
    # Inverted forces should be the negated original forces, so their sum
    # vanishes for an exactly parity-equivariant model
    delta_forces = np.abs(inverted.get_forces() + original.get_forces()).max()

    # NaN from a failed evaluation propagates, so _mean_max later blanks the model
    return float(delta_energy * EV_TO_MEV), float(delta_forces * EV_TO_MEV)


@pytest.fixture
def deltas_by_structure() -> dict[str, dict[str, tuple[float, float]]]:
    """
    Get energy and force changes for every (model, structure) pair.

    Returns
    -------
    dict[str, dict[str, tuple[float, float]]]
        Energy and force changes under inversion for all structures, for all
        models.
    """
    results: dict[str, dict[str, tuple[float, float]]] = {}
    for model_name in MODELS:
        results[model_name] = {}
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        for xyz_path in sorted(model_dir.glob("*.xyz")):
            struct_name = xyz_path.stem
            deltas = get_deltas(model_name, struct_name)
            if deltas is not None:
                results[model_name][struct_name] = deltas

    return results


@pytest.fixture
def parity_metrics(
    deltas_by_structure: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, dict[str, float | None]]:
    """
    Get mean/max energy and force changes over all structures, for all models.

    Parameters
    ----------
    deltas_by_structure
        Energy and force changes under inversion for all structures, for all
        models.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Mean/max energy and force changes for all models.
    """
    results: dict[str, dict[str, float | None]] = {
        "mean_e": {},
        "max_e": {},
        "mean_f": {},
        "max_f": {},
    }

    for model_name, per_struct in deltas_by_structure.items():
        # Checked by name, not count, so outputs for an unexpected or
        # renamed structure cannot stand in for a missing one.
        deltas = [per_struct[name] for name in EXPECTED_STRUCTS if name in per_struct]
        mean_e, max_e = _mean_max([d[0] for d in deltas])
        mean_f, max_f = _mean_max([d[1] for d in deltas])
        results["mean_e"][model_name] = mean_e
        results["max_e"][model_name] = max_e
        results["mean_f"][model_name] = mean_f
        results["max_f"][model_name] = max_f

    return results


def plot_structure_breakdown(
    model_name: str,
    per_struct: dict[str, tuple[float, float]],
) -> None:
    """
    Plot per-structure energy and force changes for one model.

    Parameters
    ----------
    model_name
        Name of model to plot changes for.
    per_struct
        Energy and force changes under inversion for all structures, for this
        model.
    """
    names = sorted(per_struct)

    @plot_scatter(
        filename=OUT_PATH / f"{model_name}_energy_by_structure.json",
        title=f"<b>{model_name}</b> energy change by structure",
        x_label="Structure",
        y_label="|ΔE| / meV per atom",
        show_line=False,
        show_markers=True,
    )
    def plot_energy() -> dict[str, tuple[list[str], list[float]]]:
        """
        Plot per-structure energy changes for this model.

        Returns
        -------
        dict[str, tuple[list[str], list[float]]]
            Structure names and energy changes under inversion.
        """
        return {"Inversion": (names, [per_struct[n][0] for n in names])}

    @plot_scatter(
        filename=OUT_PATH / f"{model_name}_force_by_structure.json",
        title=f"<b>{model_name}</b> force change by structure",
        x_label="Structure",
        y_label="max |ΔF| / meV/Å",
        show_line=False,
        show_markers=True,
    )
    def plot_force() -> dict[str, tuple[list[str], list[float]]]:
        """
        Plot per-structure force changes for this model.

        Returns
        -------
        dict[str, tuple[list[str], list[float]]]
            Structure names and largest force component changes under inversion.
        """
        return {"Inversion": (names, [per_struct[n][1] for n in names])}

    plot_energy()
    plot_force()


@pytest.fixture
def structure_breakdown(
    deltas_by_structure: dict[str, dict[str, tuple[float, float]]],
) -> None:
    """
    Write per-structure energy and force plots for all models.

    Parameters
    ----------
    deltas_by_structure
        Energy and force changes under inversion for all structures, for all
        models.
    """
    for model_name, per_struct in deltas_by_structure.items():
        if per_struct:
            plot_structure_breakdown(model_name, per_struct)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "parity_symmetry_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    parity_metrics: dict[str, dict[str, float | None]],
) -> dict[str, dict]:
    """
    Get all parity symmetry metrics.

    Parameters
    ----------
    parity_metrics
        Mean/max energy and force changes for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        f"Mean ΔE ({LABEL})": parity_metrics["mean_e"],
        f"Max ΔE ({LABEL})": parity_metrics["max_e"],
        f"Mean ΔF ({LABEL})": parity_metrics["mean_f"],
        f"Max ΔF ({LABEL})": parity_metrics["max_f"],
    }


def test_parity_symmetry(
    metrics: dict[str, dict],
    structure_breakdown: None,
) -> None:
    """
    Run parity symmetry analysis.

    Parameters
    ----------
    metrics
        All parity symmetry metrics.
    structure_breakdown
        Triggers per-structure energy and force plot generation.
    """
    return
