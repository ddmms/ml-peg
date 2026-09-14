"""Analyse rotational symmetry benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "rotational_symmetry" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "rotational_symmetry"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_MEV = 1000
# Must match the rotations built in calc_rotational_symmetry: each output file
# holds the unrotated reference frame followed by these rotations in order (a
# single cumulative walk whose first steps cover small and intermediate
# relative angles, followed by uniformly random steps). They are rebuilt here
# to compute the relative rotation angle between successive frames, which
# successive-pair comparisons are plotted against.
N_RANDOM = 100
STEP_ANGLES = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    14.0,
    20.0,
    28.0,
    40.0,
)


def _build_rotations() -> Rotation:
    """
    Rebuild the seeded walk of orientations used by the calc.

    Each step composes a rotation onto the previous orientation: STEP_ANGLES
    rotations about random axes, then N_RANDOM uniformly random rotations.

    Returns
    -------
    Rotation
        The walk's orientations, in order.
    """
    rng = np.random.default_rng(42)
    axes = rng.normal(size=(len(STEP_ANGLES), 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    steps = Rotation.concatenate(
        [
            Rotation.from_rotvec(np.radians(STEP_ANGLES)[:, None] * axes),
            Rotation.random(N_RANDOM, random_state=42),
        ]
    )

    orientations = []
    current = Rotation.identity()
    for step in steps:
        current = step * current
        orientations.append(current)

    return Rotation.concatenate(orientations)


ROTATIONS = _build_rotations()
N_ROTATIONS = len(STEP_ANGLES) + N_RANDOM


def _frame_and_relative_angles() -> tuple[np.ndarray, list[float]]:
    """
    Get the rotation angle of each frame and between successive frames.

    Frame 0 is the unrotated reference; frame k is ROTATIONS[k - 1].

    Returns
    -------
    tuple[np.ndarray, list[float]]
        The rotation angle of each frame, and the relative rotation angle
        between each pair of successive frames, both in degrees.
    """
    matrices = np.concatenate([np.eye(3)[None], ROTATIONS.as_matrix()])
    frame_angles = np.concatenate([[0.0], np.degrees(ROTATIONS.magnitude())])
    relative_angles = [
        float(
            np.degrees(
                Rotation.from_matrix(matrices[k + 1] @ matrices[k].T).magnitude()
            )
        )
        for k in range(N_ROTATIONS)
    ]
    return frame_angles, relative_angles


FRAME_ANGLES, RELATIVE_ANGLES = _frame_and_relative_angles()

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)

# Scores require complete results: a model that fails, or lacks outputs, for
# any structure receives no score, rather than being scored on the structures
# that survived. The mock outputs (which must exist for the analysis to run at
# all) define the full structure set.
EXPECTED_STRUCTS = sorted(path.stem for path in (CALC_PATH / "mock").glob("*.xyz"))


def struct_deltas(
    model_name: str, struct_name: str
) -> list[tuple[float, float, float]] | None:
    """
    Get the energy and force changes for one model/structure pair.

    Successive frames (the unrotated reference, then each rotation in turn)
    are compared pairwise, so no single orientation serves as a privileged
    reference. Energy is invariant under rotation, so energies are compared
    directly. Forces are equivariant, so the comparison uses each frame's
    "forces_original_frame" array (the forces rotated back into a common
    frame, stored by the calc), which a perfect model leaves identical for
    every frame.

    Parameters
    ----------
    model_name
        Name of model to compute changes for.
    struct_name
        Name of structure to compute changes for.

    Returns
    -------
    list[tuple[float, float, float]] | None
        Per successive frame pair (relative rotation angle (degrees),
        absolute energy change (meV/atom), force MAE (meV/Å)), or None if
        outputs are missing or do not match the expected rotations.
    """
    xyz_path = CALC_PATH / model_name / f"{struct_name}.xyz"
    if not xyz_path.exists():
        return None

    frames = read(xyz_path, index=":")
    # Guard against stale outputs written with a different rotation set: the
    # frames must line up with the rotations rebuilt at module level.
    angles = [frame.info["angle"] for frame in frames]
    if len(frames) != N_ROTATIONS + 1 or not np.allclose(
        angles, FRAME_ANGLES, atol=1e-6
    ):
        return None
    n_atoms = len(frames[0])

    deltas = []
    for k in range(N_ROTATIONS):
        prev, this = frames[k], frames[k + 1]
        delta_energy = (
            abs(this.get_potential_energy() - prev.get_potential_energy()) / n_atoms
        )
        # Force MAE between the two orientations, adapted from MLIP Arena
        # (Chiang et al., arXiv:2509.20630): mean absolute difference between
        # the two frames' force predictions, compared component-wise in a
        # common frame.
        delta_forces = np.abs(
            this.arrays["forces_original_frame"] - prev.arrays["forces_original_frame"]
        ).mean()
        deltas.append(
            (RELATIVE_ANGLES[k], delta_energy * EV_TO_MEV, delta_forces * EV_TO_MEV)
        )

    return deltas


@pytest.fixture
def deltas_by_structure() -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    """
    Get energy and force changes for every (model, structure) pair.

    Returns
    -------
    dict[str, dict[str, list[tuple[float, float, float]]]]
        Per successive-pair relative angles and energy/force changes for all
        structures, for all models.
    """
    results: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
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
    Get the mean and max of a list of values.

    Parameters
    ----------
    values
        Values to aggregate.

    Returns
    -------
    tuple[float | None, float | None]
        Mean and max value, or (None, None) if there are no values or any
        value is NaN: a model is not scored on partial results.
    """
    if not values or any(np.isnan(v) for v in values):
        return None, None
    return float(np.mean(values)), float(np.max(values))


@pytest.fixture
def rotation_metrics(
    deltas_by_structure: dict[str, dict[str, list[tuple[float, float, float]]]],
) -> dict[str, dict[str, float | None]]:
    """
    Get mean/max energy and force changes over all rotations, for all models.

    Parameters
    ----------
    deltas_by_structure
        Per successive-pair relative angles and energy/force changes for all
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

    for model_name, per_struct in deltas_by_structure.items():
        # Only score complete results: every structure must be present with
        # every rotation, else the metrics are left blank.
        complete = all(
            len(per_struct.get(name, [])) == N_ROTATIONS for name in EXPECTED_STRUCTS
        )
        if complete:
            deltas = [d for name in EXPECTED_STRUCTS for d in per_struct[name]]
        else:
            deltas = []
        mean_e, max_e = _mean_max([d[1] for d in deltas])
        mean_f, max_f = _mean_max([d[2] for d in deltas])
        results["mean_e"][model_name] = mean_e
        results["max_e"][model_name] = max_e
        results["mean_f"][model_name] = mean_f
        results["max_f"][model_name] = max_f

    return results


def plot_angle_breakdown(
    model_name: str,
    per_struct: dict[str, list[tuple[float, float, float]]],
) -> None:
    """
    Plot energy and force changes against rotation angle for one model.

    Parameters
    ----------
    model_name
        Name of model to plot changes for.
    per_struct
        Per successive-pair relative angles and energy/force changes for all
        structures, for this model.
    """
    struct_names = sorted(per_struct)

    @plot_scatter(
        filename=OUT_PATH / f"{model_name}_energy_by_angle.json",
        title=f"<b>{model_name}</b> energy change by relative rotation angle",
        x_label="Relative rotation angle / °",
        y_label="|ΔE| / meV per atom",
        show_line=False,
        show_markers=True,
    )
    def plot_energy() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot successive-pair energy changes for this model.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Relative rotation angles and energy changes, for each structure.
        """
        return {
            name: (
                [d[0] for d in per_struct[name]],
                [d[1] for d in per_struct[name]],
            )
            for name in struct_names
        }

    @plot_scatter(
        filename=OUT_PATH / f"{model_name}_force_by_angle.json",
        title=f"<b>{model_name}</b> force change by relative rotation angle",
        x_label="Relative rotation angle / °",
        y_label="ΔF (force MAE) / meV/Å",
        show_line=False,
        show_markers=True,
    )
    def plot_force() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot successive-pair force changes for this model.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Relative rotation angles and force changes, for each structure.
        """
        return {
            name: (
                [d[0] for d in per_struct[name]],
                [d[2] for d in per_struct[name]],
            )
            for name in struct_names
        }

    plot_energy()
    plot_force()


@pytest.fixture
def angle_breakdown(
    deltas_by_structure: dict[str, dict[str, list[tuple[float, float, float]]]],
) -> None:
    """
    Write violation-against-angle energy and force plots for all models.

    Parameters
    ----------
    deltas_by_structure
        Per successive-pair relative angles and energy/force changes for all
        structures, for all models.
    """
    for model_name, per_struct in deltas_by_structure.items():
        if per_struct:
            plot_angle_breakdown(model_name, per_struct)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "rotational_symmetry_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    rotation_metrics: dict[str, dict[str, float | None]],
) -> dict[str, dict]:
    """
    Get all rotational symmetry metrics.

    Parameters
    ----------
    rotation_metrics
        Mean/max energy and force changes for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Mean ΔE": rotation_metrics["mean_e"],
        "Max ΔE": rotation_metrics["max_e"],
        "Mean ΔF": rotation_metrics["mean_f"],
        "Max ΔF": rotation_metrics["max_f"],
    }


def test_rotational_symmetry(
    metrics: dict[str, dict],
    angle_breakdown: None,
) -> None:
    """
    Run rotational symmetry analysis.

    Parameters
    ----------
    metrics
        All rotational symmetry metrics.
    angle_breakdown
        Triggers violation-against-angle energy and force plot generation.
    """
    return
