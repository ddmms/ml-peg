"""Analyse permutation symmetry benchmark."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.physicality.permutation_symmetry.calc_permutation_symmetry import (  # noqa: E501
    STRUCTURES,
    generate_permutations,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "permutation_symmetry" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "permutation_symmetry"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

EV_TO_MEV = 1000

# Must match the metric names in metrics.yml and the column names in
# app_permutation_symmetry.
LABEL = "all permutations"

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

    A model must evaluate every structure successfully to be scored: an empty
    set of values (any structure missing) or a failed (NaN) evaluation returns
    (None, None), which renders as a blank table cell rather than a score over
    a subset.

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


def list_deltas(model_name: str, struct_name: str) -> list[float] | None:
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
    list[float] | None
        Mean and max absolute energy change (meV/atom), then mean and max of the
        largest absolute force component change (meV/Å), or None if outputs are
        missing. Mean and max are taken over all permutations of the structure,
        relative to the original. NaN if any evaluation failed, or if the
        outputs are unreadable or incomplete.
    """
    xyz_path = CALC_PATH / model_name / f"{struct_name}.xyz"
    if not xyz_path.exists():
        return None

    # Unreadable, truncated or incomplete outputs are a failed datum (NaN), so
    # _mean_max blanks the model rather than the whole analysis aborting, or
    # the model being scored on only the surviving permutations
    try:
        frames = read(xyz_path, index=":")
        indices = sorted(f.info["permutation_index"] for f in frames)
        if struct_name in STRUCTURES:
            n_expected = len(generate_permutations(STRUCTURES[struct_name])) + 1
            if indices != list(range(n_expected)):
                raise ValueError(
                    f"expected permutation indices 0-{n_expected - 1}, "
                    f"found {indices}"
                )
        original = next(f for f in frames if f.info["permutation_index"] == 0)
        n_atoms = len(original)
        e0 = original.get_potential_energy()
        f0 = original.get_forces()

        energy_deltas = []
        force_deltas = []
        for frame in frames:
            if frame.info["permutation_index"] == 0:
                continue
            # Atom i of the permuted frame is atom perm[i] of the original, so
            # scatter its forces back into the original atom order before
            # comparing
            perm = frame.arrays["permutation"]
            forces = np.empty_like(f0)
            forces[perm] = frame.get_forces()

            delta_energy = abs(frame.get_potential_energy() - e0) / n_atoms
            delta_forces = np.abs(forces - f0).max()
            energy_deltas.append(delta_energy * EV_TO_MEV)
            force_deltas.append(delta_forces * EV_TO_MEV)

        if not energy_deltas:
            raise ValueError("no permuted frames found")
    except Exception as exc:
        warn(
            f"Failed to read permutation symmetry outputs for model {model_name}, "
            f"structure {struct_name}: {exc}",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return [np.nan] * 4

    # NaN from a failed evaluation propagates, so _mean_max later blanks the model
    return [
        float(np.mean(energy_deltas)),
        float(np.max(energy_deltas)),
        float(np.mean(force_deltas)),
        float(np.max(force_deltas)),
    ]


@pytest.fixture
def deltas_by_structure() -> dict[str, dict[str, list[float]]]:
    """
    Get energy and force changes for every (model, structure) pair.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Mean and max (over permutations) energy/force changes for all
        structures, for all models.
    """
    results: dict[str, dict[str, list[float]]] = {}
    for model_name in MODELS:
        results[model_name] = {}
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        for xyz_path in sorted(model_dir.glob("*.xyz")):
            struct_name = xyz_path.stem
            deltas = list_deltas(model_name, struct_name)
            if deltas is not None:
                results[model_name][struct_name] = deltas

    return results


@pytest.fixture
def permutation_metrics(
    deltas_by_structure: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float | None]]:
    """
    Get mean/max energy and force changes over all structures, for all models.

    Parameters
    ----------
    deltas_by_structure
        Mean and max (over permutations) energy/force changes for all
        structures, for all models.

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

    # Means are taken over permutations per structure first, then over
    # structures with equal weight, so structures with many permutations do not
    # dilute the errors of those with few. Maxima are over all permutations of
    # all structures.
    for model_name, per_struct in deltas_by_structure.items():
        # Checked by name, not count, so outputs for an unexpected or
        # renamed structure cannot stand in for a missing one.
        deltas = [per_struct[name] for name in EXPECTED_STRUCTS if name in per_struct]
        results["mean_e"][model_name] = _mean_max([d[0] for d in deltas])[0]
        results["max_e"][model_name] = _mean_max([d[1] for d in deltas])[1]
        results["mean_f"][model_name] = _mean_max([d[2] for d in deltas])[0]
        results["max_f"][model_name] = _mean_max([d[3] for d in deltas])[1]

    return results


def plot_structure_breakdown(
    model_name: str,
    per_struct: dict[str, list[float]],
) -> None:
    """
    Plot per-structure mean and max energy and force changes for one model.

    Parameters
    ----------
    model_name
        Name of model to plot changes for.
    per_struct
        Mean and max (over permutations) energy/force changes for all
        structures, for this model.
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
            Structure names and mean/max energy changes over permutations.
        """
        return {
            "Mean over permutations": (names, [per_struct[n][0] for n in names]),
            "Max over permutations": (names, [per_struct[n][1] for n in names]),
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
            Structure names and mean/max force changes over permutations.
        """
        return {
            "Mean over permutations": (names, [per_struct[n][2] for n in names]),
            "Max over permutations": (names, [per_struct[n][3] for n in names]),
        }

    plot_energy()
    plot_force()


@pytest.fixture
def structure_breakdown(
    deltas_by_structure: dict[str, dict[str, list[float]]],
) -> None:
    """
    Write per-structure energy and force plots for all models.

    Parameters
    ----------
    deltas_by_structure
        Mean and max (over permutations) energy/force changes for all
        structures, for all models.
    """
    for model_name, per_struct in deltas_by_structure.items():
        if per_struct:
            plot_structure_breakdown(model_name, per_struct)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "permutation_symmetry_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    permutation_metrics: dict[str, dict[str, float | None]],
) -> dict[str, dict]:
    """
    Get all permutation symmetry metrics.

    Parameters
    ----------
    permutation_metrics
        Mean/max energy and force changes for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        f"Mean ΔE ({LABEL})": permutation_metrics["mean_e"],
        f"Max ΔE ({LABEL})": permutation_metrics["max_e"],
        f"Mean ΔF ({LABEL})": permutation_metrics["mean_f"],
        f"Max ΔF ({LABEL})": permutation_metrics["max_f"],
    }


def test_permutation_symmetry(
    metrics: dict[str, dict],
    structure_breakdown: None,
) -> None:
    """
    Run permutation symmetry analysis.

    Parameters
    ----------
    metrics
        All permutation symmetry metrics.
    structure_breakdown
        Triggers per-structure energy and force plot generation.
    """
    return
