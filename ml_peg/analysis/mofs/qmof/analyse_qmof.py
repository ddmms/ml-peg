"""Analyse qmof benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
    sample_density_grid,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "mofs" / "qmof" / "outputs"
OUT_PATH = APP_ROOT / "data" / "mofs" / "qmof"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="qmof_valid_structures.traj",
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
)


def write_density_trajs(
    mofs: list, ref_vals: list[float], pred_vals: list[float], traj_dir: Path
) -> None:
    """
    Write one trajectory per density scatter point for WEAS structure viewing.

    Parameters
    ----------
    mofs
        Structures in the same order as `ref_vals` and `pred_vals`.
    ref_vals
        Reference energies passed to the density scatter.
    pred_vals
        Predicted energies passed to the density scatter.
    traj_dir
        Directory to write sampled trajectories to.
    """
    _, _, sampled_mapping = sample_density_grid(ref_vals, pred_vals)
    traj_dir.mkdir(parents=True, exist_ok=True)
    for point_idx, source_indices in enumerate(sampled_mapping):
        write(traj_dir / f"{point_idx}.extxyz", [mofs[idx] for idx in source_indices])


@pytest.fixture
def qmof_energies() -> dict[str, list]:
    """
    Get energies per atom for all qmof systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted energies per atom.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        struct_file = "qmof_valid_structures.traj"
        mofs = read(model_dir / struct_file, index=":")
        for mof in mofs:
            mof_energy = mof.get_potential_energy() / len(mof)

            results[model_name].append(mof_energy)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(mof.info["dft_energy"] / len(mof))

        # Write structure trajectories for density scatter points
        write_density_trajs(
            mofs,
            results["ref"],
            results[model_name],
            OUT_PATH / model_name / "density_traj",
        )

        ref_stored = True

    return results


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_qmof_density.json",
    title="QMOF energy density plot",
    x_label="Reference energy / eV/atom",
    y_label="Predicted energy / eV/atom",
    annotation_metadata={"system_count": "Systems"},
)
def qmof_density(qmof_energies: dict[str, list]) -> dict[str, dict]:
    """
    Build density scatter inputs for qmof energies.

    Parameters
    ----------
    qmof_energies
        Dictionary of reference and predicted energies per atom.

    Returns
    -------
    dict[str, dict]
        Mapping of model names to density-plot payloads.
    """
    return {
        model_name: {
            "ref": qmof_energies["ref"],
            "pred": qmof_energies[model_name],
            "meta": {"system_count": len(qmof_energies[model_name])},
        }
        for model_name in MODELS
    }


@pytest.fixture
def qmof_errors(qmof_energies) -> dict[str, float]:
    """
    Get mean absolute error for energies per atom.

    Parameters
    ----------
    qmof_energies
        Dictionary of reference and predicted energies per atom.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if qmof_energies[model_name]:
            results[model_name] = mae(qmof_energies["ref"], qmof_energies[model_name])
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "qmof_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(qmof_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all qmof metrics.

    Parameters
    ----------
    qmof_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": qmof_errors,
    }


def test_qmof(metrics: dict[str, dict], qmof_density: dict[str, dict]) -> None:
    """
    Run qmof test.

    Parameters
    ----------
    metrics
        All qmof metrics.
    qmof_density
        Density scatter data for qmof energies.
    """
    return
