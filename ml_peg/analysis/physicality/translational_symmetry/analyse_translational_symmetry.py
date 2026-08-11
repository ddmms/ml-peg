"""Analyse translational symmetry benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "translational_symmetry" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "translational_symmetry"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_MEV = 1000
# Must match calc_translational_symmetry.MAGNITUDES.
MAGNITUDES = (1, 40, 1000)
# Must match len(calc_translational_symmetry.STRUCTURES). A model only receives
# scores for a magnitude if every structure was evaluated successfully at it.
N_STRUCTURES = 10

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


def struct_deltas(
    model_name: str, struct_name: str
) -> dict[float, tuple[float, float]] | None:
    """
    Get the energy and force changes for one model/structure pair.

    Parameters
    ----------
    model_name
        Name of model to compute changes for.
    struct_name
        Name of structure to compute changes for.

    Returns
    -------
    dict[float, tuple[float, float]] | None
        Per-magnitude (absolute energy change (meV/atom), max absolute force
        component change (meV/Å)), or None if outputs are missing.
    """
    xyz_path = CALC_PATH / model_name / f"{struct_name}.xyz"
    if not xyz_path.exists():
        return None

    frames = read(xyz_path, index=":")
    original = next(f for f in frames if f.info["shift_magnitude"] == 0)
    n_atoms = len(original)
    e0 = original.get_potential_energy()
    f0 = original.get_forces()

    deltas = {}
    for frame in frames:
        magnitude = frame.info["shift_magnitude"]
        if magnitude == 0:
            continue
        delta_energy = abs(frame.get_potential_energy() - e0) / n_atoms
        delta_forces = np.abs(frame.get_forces() - f0).max()
        deltas[magnitude] = (delta_energy * EV_TO_MEV, delta_forces * EV_TO_MEV)

    return deltas


@pytest.fixture
def deltas_by_structure() -> dict[str, dict[str, dict[float, tuple[float, float]]]]:
    """
    Get energy and force changes for every (model, structure) pair.

    Returns
    -------
    dict[str, dict[str, dict[float, tuple[float, float]]]]
        Per-magnitude energy/force changes for all structures, for all models.
    """
    results: dict[str, dict[str, dict[float, tuple[float, float]]]] = {}
    for model_name in MODELS:
        results[model_name] = {}
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        for xyz_path in sorted(model_dir.glob("*.xyz")):
            struct_name = xyz_path.stem
            deltas = struct_deltas(model_name, struct_name)
            if deltas is not None:
                results[model_name][struct_name] = deltas

    return results


def _mean_max(values: list[float]) -> tuple[float | None, float | None]:
    """
    Get the mean and max of a list of values, requiring a complete set.

    A model must evaluate every structure successfully to be scored: any
    missing structure or failed (NaN) evaluation returns (None, None), which
    renders as a blank table cell rather than a score over a subset.

    Parameters
    ----------
    values
        Values to aggregate, one per successfully read structure.

    Returns
    -------
    tuple[float | None, float | None]
        Mean and max value, or (None, None) if any structure is missing or
        failed.
    """
    if len(values) != N_STRUCTURES or any(np.isnan(v) for v in values):
        return None, None
    return float(np.mean(values)), float(np.max(values))


@pytest.fixture
def magnitude_metrics(
    deltas_by_structure: dict[str, dict[str, dict[float, tuple[float, float]]]],
) -> dict[float, dict[str, dict[str, float | None]]]:
    """
    Get mean/max energy and force changes per magnitude, for all models.

    Parameters
    ----------
    deltas_by_structure
        Per-magnitude energy/force changes for all structures, for all models.

    Returns
    -------
    dict[float, dict[str, dict[str, float | None]]]
        For each magnitude, mean/max energy and force changes for all models.
    """
    results: dict[float, dict[str, dict[str, float | None]]] = {
        magnitude: {"mean_e": {}, "max_e": {}, "mean_f": {}, "max_f": {}}
        for magnitude in MAGNITUDES
    }

    for model_name, per_struct in deltas_by_structure.items():
        for magnitude in MAGNITUDES:
            pairs = [
                deltas[magnitude]
                for deltas in per_struct.values()
                if magnitude in deltas
            ]
            energies = [pair[0] for pair in pairs]
            forces = [pair[1] for pair in pairs]
            mean_e, max_e = _mean_max(energies)
            mean_f, max_f = _mean_max(forces)
            results[magnitude]["mean_e"][model_name] = mean_e
            results[magnitude]["max_e"][model_name] = max_e
            results[magnitude]["mean_f"][model_name] = mean_f
            results[magnitude]["max_f"][model_name] = max_f

    return results


def plot_structure_breakdown(
    model_name: str,
    per_struct: dict[str, dict[float, tuple[float, float]]],
) -> None:
    """
    Plot per-structure energy and force changes for one model, all magnitudes.

    Parameters
    ----------
    model_name
        Name of model to plot changes for.
    per_struct
        Per-magnitude energy/force changes for all structures, for this model.
    """
    struct_names = sorted(per_struct)
    # Outputs written before a change to MAGNITUDES lack the newer magnitudes, so
    # plot only the structures actually evaluated at each one.
    names_by_magnitude = {
        magnitude: [name for name in struct_names if magnitude in per_struct[name]]
        for magnitude in MAGNITUDES
    }

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
            Structure names and energy changes, for each magnitude.
        """
        return {
            f"{magnitude} Å": (
                names,
                [per_struct[name][magnitude][0] for name in names],
            )
            for magnitude, names in names_by_magnitude.items()
            if names
        }

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
            Structure names and force changes, for each magnitude.
        """
        return {
            f"{magnitude} Å": (
                names,
                [per_struct[name][magnitude][1] for name in names],
            )
            for magnitude, names in names_by_magnitude.items()
            if names
        }

    plot_energy()
    plot_force()


@pytest.fixture
def structure_breakdown(
    deltas_by_structure: dict[str, dict[str, dict[float, tuple[float, float]]]],
) -> None:
    """
    Write per-structure energy and force plots for all models.

    Parameters
    ----------
    deltas_by_structure
        Per-magnitude energy/force changes for all structures, for all models.
    """
    for model_name, per_struct in deltas_by_structure.items():
        if per_struct:
            plot_structure_breakdown(model_name, per_struct)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "translational_symmetry_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    magnitude_metrics: dict[float, dict[str, dict[str, float | None]]],
) -> dict[str, dict]:
    """
    Get all translational symmetry metrics.

    Parameters
    ----------
    magnitude_metrics
        For each magnitude, mean/max energy and force changes for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    result = {}
    for magnitude in MAGNITUDES:
        result[f"Mean ΔE ({magnitude} Å)"] = magnitude_metrics[magnitude]["mean_e"]
        result[f"Max ΔE ({magnitude} Å)"] = magnitude_metrics[magnitude]["max_e"]
        result[f"Mean ΔF ({magnitude} Å)"] = magnitude_metrics[magnitude]["mean_f"]
        result[f"Max ΔF ({magnitude} Å)"] = magnitude_metrics[magnitude]["max_f"]
    return result


def test_translational_symmetry(
    metrics: dict[str, dict],
    structure_breakdown: None,
) -> None:
    """
    Run translational symmetry analysis.

    Parameters
    ----------
    metrics
        All translational symmetry metrics.
    structure_breakdown
        Triggers per-structure energy and force plot generation.
    """
    return
