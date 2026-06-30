"""Analyse the polymer-density benchmark."""

from __future__ import annotations

import logging
import pathlib

import ase.io
import ase.io.trajectory as ase_traj
import ase.units as ase_units
import numpy as np
import pandas as pd
import pytest

from ml_peg import app, calcs
from ml_peg.analysis.utils import decorators
from ml_peg.analysis.utils import utils as analysis_utils
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

LOG = logging.getLogger(__name__)

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = analysis_utils.build_dispersion_name_map(MODELS)
PRECOMPUTED_DENSITY_MODELS: tuple[str, ...] = (
    "MACE-OFF23-S",
    "PCFF",
    "uma-s-1p1-omol",
    "vivance",
)

AU_TO_G_CM3 = 1e24 / ase_units.mol
PRODUCTION_STAGE = "23_step22_final_npt"

CALC_PATH = calcs.CALCS_ROOT / "molecular_dynamics" / "polymers" / "outputs"
DATA_CSV = (
    calcs.CALCS_ROOT / "molecular_dynamics" / "polymers" / "resources" / "data.csv"
)
POLYMER_SETS_DIR = (
    calcs.CALCS_ROOT / "molecular_dynamics" / "polymers" / "resources" / "polymer_sets"
)
POLYMER_SETS: dict[str, frozenset[str]] = {
    "MAE (small)": frozenset((POLYMER_SETS_DIR / "small.txt").read_text().splitlines()),
    "MAE (medium)": frozenset(
        (POLYMER_SETS_DIR / "medium.txt").read_text().splitlines()
    ),
    "MAE (large)": frozenset((POLYMER_SETS_DIR / "large.txt").read_text().splitlines()),
    "MAE (X-large)": frozenset(
        (POLYMER_SETS_DIR / "x-large.txt").read_text().splitlines()
    ),
}
OUT_PATH = app.APP_ROOT / "data" / "molecular_dynamics" / "polymers"
LOCAL_STRUCTURES_DIR = calcs.CALCS_ROOT / "molecular_dynamics" / "polymers" / "polymers"
TRAJECTORY_MODEL_NAMES = tuple(
    model_name
    for model_name in MODELS
    if any((CALC_PATH / model_name).glob(f"*/{PRODUCTION_STAGE}.traj"))
)
DENSITY_MODEL_NAMES = list(
    dict.fromkeys((*PRECOMPUTED_DENSITY_MODELS, *TRAJECTORY_MODEL_NAMES))
)

METRICS_CONFIG_PATH = pathlib.Path(__file__).with_name("metrics.yml")
# `load_metrics_config` actually returns a 3-tuple (thresholds, tooltips,
# weights), but its declared signature is a 2-tuple — silence the upstream
# stub mismatch.
(  # ty: ignore[invalid-assignment]
    DEFAULT_THRESHOLDS,
    DEFAULT_TOOLTIPS,
    DEFAULT_WEIGHTS,
) = analysis_utils.load_metrics_config(METRICS_CONFIG_PATH)  # type: ignore[misc]


def _load_polymer_table() -> pd.DataFrame:
    """
    Load the experimental polymer table indexed by polymer id.

    Returns
    -------
    pd.DataFrame
        The polymer table indexed by ``id``, sorted alphabetically.
    """
    df = pd.read_csv(DATA_CSV, na_values=["NaN"], encoding="utf-8", comment="%")
    return df.set_index("id").sort_index()


POLYMER_TABLE = _load_polymer_table()


def labels() -> list[str]:
    """
    Return the list of polymer ids covered by the benchmark.

    Returns
    -------
    list[str]
        Polymer ids from ``data.csv``, in sorted order.
    """
    return [str(poly_id) for poly_id in POLYMER_TABLE.index]


def _load_density_csv(model_name: str) -> dict[str, float]:
    """
    Load precomputed model densities from ``outputs/<model>/densities.csv``.

    Parameters
    ----------
    model_name
        Model/output directory name.

    Returns
    -------
    dict[str, float]
        Mapping from polymer id to density in g/cm³.
    """
    csv_path = CALC_PATH / model_name / "densities.csv"
    if not csv_path.exists():
        LOG.warning(f"Missing precomputed density CSV: {csv_path}")
        return {}
    df = pd.read_csv(csv_path, comment="#")
    return dict(
        zip(df["poly_id"].astype(str), df["density"].astype(float), strict=False)
    )


def _write_polymer_structure(
    poly_id: str,
    structs_dir: pathlib.Path,
    atoms=None,
) -> None:
    """
    Write a polymer structure into a model-specific app data directory.

    Parameters
    ----------
    poly_id
        Polymer identifier.
    structs_dir
        Model-specific app data directory.
    atoms
        Final trajectory frame to write. If None, write the input structure.
    """
    if atoms is None:
        input_structure = LOCAL_STRUCTURES_DIR / f"{poly_id}.xyz"
        if not input_structure.exists():
            return
        atoms = ase.io.read(input_structure)
    ase.io.write(str(structs_dir / f"{poly_id}.xyz"), atoms)


def _open_trajectory(traj_path: pathlib.Path) -> ase_traj.TrajectoryReader | None:
    """
    Open an ASE trajectory for reading; ``None`` if missing or unreadable.

    Parameters
    ----------
    traj_path
        Path to an ASE trajectory file.

    Returns
    -------
    TrajectoryReader | None
        Open reader, or ``None`` if the file is missing or unreadable.
    """
    if not traj_path.exists():
        return None
    try:
        return ase_traj.Trajectory(str(traj_path))
    except (OSError, ValueError, RuntimeError) as err:
        LOG.warning(f"Could not read {traj_path}: {err}")
        return None


def _mean_density_g_cm3(traj: ase_traj.TrajectoryReader) -> float:
    """
    Average the density (g/cm³) over every frame of an ASE trajectory.

    Parameters
    ----------
    traj
        Open ASE trajectory reader (already known to be non-empty).

    Returns
    -------
    float
        Frame-averaged density in g/cm³.
    """
    densities = [
        AU_TO_G_CM3 * float(frame.get_masses().sum()) / float(frame.get_volume())
        for frame in traj
    ]
    return float(np.mean(densities))


@pytest.fixture
@decorators.plot_parity(
    filename=str(OUT_PATH / "figure_polymers.json"),
    title="Polymer densities",
    x_label="Predicted density / g/cm^3",
    y_label="Reference density / g/cm^3",
    hoverdata={"Labels": labels()},
)
def polymer_densities() -> dict[str, list[float]]:
    """
    Collect reference and predicted densities for all polymers and models.

    Returns
    -------
    dict[str, list[float]]
        ``{"ref": [...], "<model>": [...], ...}`` aligned to ``labels()``.
        Missing, unreadable, or empty trajectories contribute ``NaN``.
    """
    poly_ids = labels()

    results: dict[str, list[float]] = {"ref": []}
    for name in DENSITY_MODEL_NAMES:
        results[name] = []

    for poly_id in poly_ids:
        ref_density = float(POLYMER_TABLE.loc[poly_id, "density"])
        results["ref"].append(ref_density)

    for model_name in DENSITY_MODEL_NAMES:
        # Per project convention, the dashboard/app must not read from the
        # calc outputs directory: anything the app needs (here, the final
        # production-stage frame for visualisation) is copied here, into the
        # app's data tree, during analysis.
        csv_densities = (
            _load_density_csv(model_name)
            if model_name in PRECOMPUTED_DENSITY_MODELS
            else {}
        )
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        for poly_id in poly_ids:
            traj_path = CALC_PATH / model_name / poly_id / f"{PRODUCTION_STAGE}.traj"
            traj = _open_trajectory(traj_path)
            if traj is None or len(traj) == 0:
                _write_polymer_structure(poly_id, structs_dir)
            else:
                _write_polymer_structure(poly_id, structs_dir, traj[-1])

            if poly_id in csv_densities:
                results[model_name].append(csv_densities[poly_id])
                continue

            # Average over the full production-NPT stage; the preceding
            # `npt_equilibration` stage is the warm-up window.
            if traj is None or len(traj) == 0:
                results[model_name].append(float("nan"))
                continue
            results[model_name].append(_mean_density_g_cm3(traj))

    return results


@pytest.fixture
def get_mae_by_subset(
    polymer_densities: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """
    Compute per-subset MAE for each model, ignoring NaN entries.

    Parameters
    ----------
    polymer_densities
        Output of :func:`polymer_densities`.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{"MAE (small)": {model: mae, ...}, "MAE (medium)": ..., ...}``.
        A subset MAE is reported only when all polymers in that subset are available.
    """
    poly_ids = labels()
    ref = np.asarray(polymer_densities["ref"], dtype=float)
    result: dict[str, dict[str, float]] = {}
    for set_name, poly_set in POLYMER_SETS.items():
        subset_mask = np.array([pid in poly_set for pid in poly_ids])
        model_maes: dict[str, float] = {}
        for model_name in DENSITY_MODEL_NAMES:
            pred = np.asarray(polymer_densities[model_name], dtype=float)
            mask = subset_mask & np.isfinite(ref) & np.isfinite(pred)
            model_maes[model_name] = (
                float(analysis_utils.mae(ref[mask].tolist(), pred[mask].tolist()))
                if mask.sum() == subset_mask.sum()
                else None
            )
        result[set_name] = model_maes
    return result


@pytest.fixture
@decorators.build_table(
    filename=str(OUT_PATH / "polymers_metrics_table.json"),
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    get_mae_by_subset: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Assemble the metric table consumed by the dashboard.

    Parameters
    ----------
    get_mae_by_subset
        Per-subset, per-model MAE in g/cm³.

    Returns
    -------
    dict[str, dict]
        ``{"MAE (small)": {...}, "MAE (medium)": {...}, "MAE (large)": {...}}``.
    """
    return get_mae_by_subset


def test_polymers(metrics: dict[str, dict[str, float]]) -> None:  # noqa: PT019
    """
    Materialize the polymer metric table (no assertions).

    Parameters
    ----------
    metrics
        Per-metric, per-model values produced by :func:`metrics`.
    """
    return
