"""
Analyse the BH9 reaction barriers dataset.

Journal of Chemical Theory and Computation 2022 18 (1), 151-166
DOI: 10.1021/acs.jctc.1c00694
"""

from __future__ import annotations

from pathlib import Path

from ase import units
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

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "BH9" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "BH9"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


def get_system_names() -> list[str]:
    """
    Get list of reaction system names from the first available model.

    Returns
    -------
    list[str]
        List of system names (reaction identifiers).
    """
    for model_name in sorted(CALC_PATH.glob("*")):
        if model_name.is_dir():
            # Note: sorting different to rxn_count order in calc
            xyz_paths = sorted((CALC_PATH / model_name).glob("*.xyz"))
            return [path.stem for path in xyz_paths]
    return []


def get_reaction_numbers() -> list[int]:
    """
    Get reaction numbers extracted from system names.

    Returns
    -------
    list[int]
        List of reaction numbers (e.g., [1, 2, 3, ...]).
    """
    system_names = get_system_names()
    reaction_nums = []
    for name in system_names:
        # Extract reaction number from format like "01_1" -> 1
        parts = name.split("_")
        if len(parts) == 2:
            reaction_nums.append(int(parts[0]))
    return reaction_nums


def get_structure_numbers() -> list[int]:
    """
    Get structure numbers (different geometries for same reaction).

    Returns
    -------
    list[int]
        List of structure numbers for each reaction.
    """
    system_names = get_system_names()
    struct_nums = []
    for name in system_names:
        # Extract structure number from format like "01_1" -> 1
        parts = name.split("_")
        if len(parts) == 2:
            struct_nums.append(int(parts[1]))
    return struct_nums


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_bh9_barriers.json",
    title="Reaction barriers",
    x_label="Predicted barrier / kcal/mol",
    y_label="Reference barrier / kcal/mol",
    hoverdata={
        "Reaction": get_reaction_numbers(),
        "Structure": get_structure_numbers(),
        "System ID": get_system_names(),
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
        model_barriers = []
        ref_barriers = []
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            results[model_name] = []
            continue
        for system_name in system_names:
            model_forward_barrier = 0
            ref_forward_barrier = 0

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)

            for fname in model_dir.glob(f"{system_name}*"):
                atoms = read(fname, index=":")
                model_forward_barrier += atoms[0].info["model_energy"]
                for struct in atoms[1:]:
                    model_forward_barrier -= struct.info["model_energy"]

                ref_forward_barrier = atoms[0].info["ref_forward_barrier"]

                write(structs_dir / f"{fname.stem}.xyz", atoms)

            model_barriers.append(model_forward_barrier * EV_TO_KCAL)
            ref_barriers.append(ref_forward_barrier * EV_TO_KCAL)

        results[model_name] = model_barriers
        if not ref_stored:
            results["ref"] = ref_barriers
            ref_stored = True
    return results


@pytest.fixture
def get_mae(barrier_heights: dict[str, list]) -> dict[str, float]:
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
    filename=OUT_PATH / "bh9_barriers_metrics_table.json",
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


def test_bh9_barriers(metrics: dict[str, dict]) -> None:
    """
    Run BH9 test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
