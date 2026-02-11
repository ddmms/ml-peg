"""
Analyse the BH2O-36 benchmark for hydrolysis reaction barriers.

Journal of Chemical Theory and Computation 2023 19 (11), 3159-3171
DOI: 10.1021/acs.jctc.3c00176
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "BH2O_36" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "BH2O_36"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of reaction system names from the first available model.

    Returns
    -------
    list[str]
        List of base system names (without suffixes).
    """
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            system_names = []
            for system_path in sorted(model_dir.glob("*_ts.xyz")):
                system_names.append(system_path.stem.replace("_ts", ""))
            if system_names:
                return system_names
    return []


def get_barrier_labels() -> list[str]:
    """
    Get list of barrier labels for plotting (two per reaction system).

    Returns
    -------
    list[str]
        List of barrier labels with reaction context.
    """
    return [
        label
        for system in get_system_names()
        for label in [f"{system} (TS-Reactants)", f"{system} (TS-Products)"]
    ]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_bh2o_36_barriers.json",
    title="Reaction barriers",
    x_label="Predicted barrier / eV",
    y_label="Reference barrier / eV",
    hoverdata={
        "System": [
            system_name
            for system_name in get_system_names()
            for _ in range(2)  # Duplicate each system name for both barriers
        ],
        "Barrier Type": [
            barrier_type
            for _ in get_system_names()
            for barrier_type in ["TS-Reactants", "TS-Products"]
        ],
    },
)
def barrier_heights() -> dict[str, list]:
    """
    Get barrier heights for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted barrier heights.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    system_names = get_system_names()
    for model_name in MODELS:
        for system_name in system_names:
            atoms_rct = read(CALC_PATH / model_name / f"{system_name}_rct.xyz")
            atoms_pro = read(CALC_PATH / model_name / f"{system_name}_pro.xyz")
            atoms_ts = read(CALC_PATH / model_name / f"{system_name}_ts.xyz")

            # TS - Reactants barrier
            results[model_name].append(
                atoms_ts.info["pred_energy"] - atoms_rct.info["pred_energy"]
            )
            # TS - Products barrier
            results[model_name].append(
                atoms_ts.info["pred_energy"] - atoms_pro.info["pred_energy"]
            )

            if not ref_stored:
                results["ref"].append(
                    atoms_ts.info["ref_energy"] - atoms_rct.info["ref_energy"]
                )
                results["ref"].append(
                    atoms_ts.info["ref_energy"] - atoms_pro.info["ref_energy"]
                )

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)

            # Write individual structures
            write(structs_dir / f"{system_name}_rct.xyz", atoms_rct)
            write(structs_dir / f"{system_name}_pro.xyz", atoms_pro)
            write(structs_dir / f"{system_name}_ts.xyz", atoms_ts)

            # Write trajectory files for each barrier type
            # TS-Reactants barrier: reactants -> TS
            write(
                structs_dir / f"{system_name}_rct_to_ts.xyz",
                [atoms_rct, atoms_ts],
                append=False,
            )
            # TS-Products barrier: products -> TS
            write(
                structs_dir / f"{system_name}_pro_to_ts.xyz",
                [atoms_pro, atoms_ts],
                append=False,
            )
        ref_stored = True
    return results


@pytest.fixture
def get_mae(barrier_heights) -> dict[str, float]:
    """
    Get mean absolute error for barrier heights.

    Parameters
    ----------
    barrier_heights
        Dictionary of reference and predicted barrier heights.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier height errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(barrier_heights["ref"], barrier_heights[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "bh2o_36_barriers_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
    }


def test_bh2o_36_barriers(metrics: dict[str, dict]) -> None:
    """
    Run BH2O-36 test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
