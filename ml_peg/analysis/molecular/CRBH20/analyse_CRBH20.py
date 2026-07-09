"""
Analyse CRBH20 Reaction Barriers benchmark.

Reference barriers from Appendix B.5 of arXiv:2401.00096.
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular" / "CRBH20" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "CRBH20"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal

RXN_IDS = tuple(range(1, 21))

# Reference barriers in eV (Appendix B.5 of arXiv:2401.00096), from DFT (r2SCAN)
REF_BARRIERS_EV = {
    1: 1.7194,
    2: 1.9241,
    3: 1.7499,
    4: 1.8238,
    5: 1.7237,
    6: 1.5653,
    7: 1.0911,
    8: 1.8983,
    9: 1.5477,
    10: 1.7115,
    11: 1.7379,
    12: 2.0361,
    13: 1.8739,
    14: 1.9760,
    15: 1.8865,
    16: 1.5741,
    17: 1.2587,
    18: 1.7497,
    19: 1.6989,
    20: 1.7654,
}
REF_BARRIERS_KCAL = {
    rxn_id: barrier * EV_TO_KCAL for rxn_id, barrier in REF_BARRIERS_EV.items()
}

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    sort_key=lambda path: int(path.stem.removeprefix("crbh20_")),
    index=0,
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_reaction_barriers.json",
    title="CRBH20 Reaction Barriers",
    x_label="Predicted Barrier (kcal/mol)",
    y_label="Reference Barrier (kcal/mol)",
    hoverdata={
        "Reaction ID": [str(rxn_id) for rxn_id in RXN_IDS],
    },
)
def reaction_barriers() -> dict[str, list]:
    """
    Get barriers for all CRBH20 systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted barriers in kcal/mol.
    """
    results = {"ref": [REF_BARRIERS_KCAL[rxn_id] for rxn_id in RXN_IDS]}
    results |= {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        for rxn_id in RXN_IDS:
            xyz_file = model_dir / f"crbh20_{rxn_id}.xyz"

            if not xyz_file.exists():
                results[model_name].append(np.nan)
                continue

            # Read the combined XYZ (reactant is index 0, TS is index 1)
            # The barrier is stored in the info tag of both structures
            structs = read(xyz_file, index=":")
            reactant = structs[0]
            results[model_name].append(reactant.info.get("barrier_kcal", np.nan))

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"crbh20_{rxn_id}.xyz", structs)

    return results


@pytest.fixture
def crbh20_errors(reaction_barriers) -> dict[str, float]:
    """
    Compute Mean Absolute Error (MAE) for reaction barriers.

    Parameters
    ----------
    reaction_barriers
        Dictionary of reference and predicted barriers in kcal/mol.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier errors for all models.
    """
    return {
        model_name: mae(reaction_barriers["ref"], reaction_barriers[model_name])
        for model_name in MODELS
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "crbh20_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(crbh20_errors: dict[str, float]) -> dict[str, dict]:
    """
    Compile all metrics for the table.

    Parameters
    ----------
    crbh20_errors
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": crbh20_errors,
    }


def test_crbh20_analysis(metrics: dict[str, dict]) -> None:
    """
    Trigger the analysis pipeline.

    The decorators on the fixtures above (@plot_parity, @build_table)
    do the heavy lifting of saving the JSON files when this test runs.

    Parameters
    ----------
    metrics
        All benchmark metric names and dictionary of values for each model.
    """
    assert metrics is not None
    assert "MAE" in metrics
