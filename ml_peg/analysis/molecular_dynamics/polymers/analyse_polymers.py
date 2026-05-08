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
from ml_peg.models import get_models, models

LOG = logging.getLogger(__name__)

MODELS = get_models.load_models(models.current_models)
D3_MODEL_NAMES = analysis_utils.build_dispersion_name_map(MODELS)

AU_TO_G_CM3 = 1e24 / ase_units.mol
PRODUCTION_STAGE = "23_step22_final_npt"

CALC_PATH = calcs.CALCS_ROOT / "molecular_dynamics" / "polymers" / "outputs"
DATA_CSV = (
    calcs.CALCS_ROOT / "molecular_dynamics" / "polymers" / "resources" / "data.csv"
)
OUT_PATH = app.APP_ROOT / "data" / "molecular_dynamics" / "polymers"

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
    for name in MODELS:
        results[name] = []

    for poly_id in poly_ids:
        ref_density = float(POLYMER_TABLE.loc[poly_id, "density"])
        results["ref"].append(ref_density)

    for model_name in MODELS:
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        for poly_id in poly_ids:
            # Average over the full production-NPT stage; the preceding
            # `npt_equilibration` stage is the warm-up window.
            traj_path = CALC_PATH / model_name / poly_id / f"{PRODUCTION_STAGE}.traj"
            traj = _open_trajectory(traj_path)
            if traj is None or len(traj) == 0:
                results[model_name].append(float("nan"))
                continue
            results[model_name].append(_mean_density_g_cm3(traj))
            ase.io.write(str(structs_dir / f"{poly_id}.xyz"), traj[-1])

    return results


@pytest.fixture
def get_mae(polymer_densities: dict[str, list[float]]) -> dict[str, float]:
    """
    Compute the mean absolute density error per model, ignoring NaN entries.

    Parameters
    ----------
    polymer_densities
        Output of :func:`polymer_densities`.

    Returns
    -------
    dict[str, float]
        Per-model MAE in g/cm³.
    """
    ref = np.asarray(polymer_densities["ref"], dtype=float)
    out: dict[str, float] = {}
    for model_name in MODELS:
        pred = np.asarray(polymer_densities[model_name], dtype=float)
        mask = np.isfinite(ref) & np.isfinite(pred)
        if not mask.any():
            out[model_name] = float("nan")
        else:
            out[model_name] = float(
                analysis_utils.mae(ref[mask].tolist(), pred[mask].tolist())
            )
    return out


@pytest.fixture
@decorators.build_table(
    filename=str(OUT_PATH / "polymers_metrics_table.json"),
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_mae: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    Assemble the metric table consumed by the dashboard.

    Parameters
    ----------
    get_mae
        Per-model MAE in g/cm³.

    Returns
    -------
    dict[str, dict]
        ``{"MAE": {<model>: <value>, ...}}``.
    """
    return {"MAE": get_mae}


def test_polymers(metrics: dict[str, dict[str, float]]) -> None:
    """
    Materialize the polymer metric table (no assertions).

    Parameters
    ----------
    metrics
        Per-metric, per-model values produced by :func:`metrics`.
    """
    return
