"""
Analyse the glucose dataset for pyrolysis.

Zhao, Qiyuan, and Brett M. Savoie.
Proceedings of the National Academy of Sciences
120.34 (2023): e2305884120
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map()

EV_TO_KCAL = units.mol / units.kcal

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "glucose" / "outputs"

OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "glucose"

OUTPUT_FILENAME = "glucose_calculated.xyz"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


# ---------------------------------------------------------------------
# Structure information
# ---------------------------------------------------------------------
def get_glucose_info(model_name: str) -> dict:
    """
    Build the info.json for plot.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    dict
       A dictionary with two keys of elements and filenames.
    """
    data = {"elements": [], "filenames": []}
    reactions = load_reactions(model_name)
    keys = list(reactions.keys())
    keys = sorted(keys)
    for key in keys:
        data["filenames"].append(key)
        data["elements"].append(reactions[key]["ts"].get_chemical_symbols())

    return data


# ---------------------------------------------------------------------
# Read and organize reaction structures
# ---------------------------------------------------------------------


def load_reactions(model_name: str) -> dict:
    """
    Read the multi-frame XYZ file and organize structures by reaction ID.

    Each reaction must contain exactly one reactant and one TS.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    dict
        Dictionary containing reactant and TS structures for each reaction.
    """
    xyz_path = CALC_PATH / model_name / OUTPUT_FILENAME

    structures = read(xyz_path, index=":")

    reactions = {}

    for atoms in structures:
        info = atoms.info

        required_keys = [
            "reaction_id",
            "type",
            "ref_energy",
            "model_energy",
        ]

        missing_keys = [key for key in required_keys if key not in info]

        if missing_keys:
            raise ValueError(f"{xyz_path}: missing metadata {missing_keys}")

        reaction_id = str(atoms.info["reaction_id"]).strip().strip('"')
        structure_type = str(info["type"]).lower()

        if structure_type not in ("reactant", "ts"):
            raise ValueError(
                f"Unexpected structure type '{structure_type}' "
                f"for reaction {reaction_id}"
            )

        if reaction_id not in reactions:
            reactions[reaction_id] = {}

        if structure_type in reactions[reaction_id]:
            raise ValueError(
                f"Duplicate {structure_type} for reaction {reaction_id} in {model_name}"
            )

        reactions[reaction_id][structure_type] = atoms

    # Validate reaction pairs
    for reaction_id, pair in reactions.items():
        if set(pair) != {"reactant", "ts"}:
            raise ValueError(f"Incomplete reaction pair: {reaction_id} in {model_name}")

    return reactions


# ---------------------------------------------------------------------
# Barrier calculations
# ---------------------------------------------------------------------


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "glucose_barriers.json",
    title="Glucose reaction barriers",
    x_label="Predicted barrier / kcal/mol",
    y_label="Reference barrier / kcal/mol",
    hoverdata={
        "Labels": [],
    },
)
def barrier_heights() -> dict[str, list]:
    """
    Calculate reference and predicted activation barriers.

    Reactions are paired using reaction_id rather than frame order.

    Returns
    -------
    dict
        A reaction index with barrier hieght in kcal/mol.
    """
    models = [
        model_name
        for model_name in MODELS
        if os.path.exists(f"{CALC_PATH}/{model_name}")
    ]

    all_reactions = {model_name: load_reactions(model_name) for model_name in models}

    # Use the first model as the reference source.
    reference_model = models[0]
    reference_reactions = all_reactions[reference_model]
    info = get_glucose_info(reference_model)
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH / "info.json", "w") as f:
        json.dump(info, f, indent=1)
    reaction_ids = sorted(reference_reactions)

    results = {
        "ref": [],
        **{model_name: [] for model_name in models},
    }

    # Save reaction labels for parity plot.
    barrier_heights.labels = reaction_ids

    for reaction_id in reaction_ids:
        ref_pair = reference_reactions[reaction_id]

        ref_r = ref_pair["reactant"].info["ref_energy"]
        ref_ts = ref_pair["ts"].info["ref_energy"]

        reference_barrier = (ref_ts - ref_r) * EV_TO_KCAL

        results["ref"].append(reference_barrier)

        for model_name in models:
            reactions = all_reactions[model_name]

            if reaction_id not in reactions:
                raise ValueError(f"Reaction {reaction_id} missing from {model_name}")

            pair = reactions[reaction_id]

            reactant = pair["reactant"]
            ts = pair["ts"]

            # Check reference energy consistency.
            model_ref_r = reactant.info["ref_energy"]
            model_ref_ts = ts.info["ref_energy"]

            if abs(model_ref_r - ref_r) > 1e-6:
                raise ValueError(
                    f"Reference reactant energy mismatch: {reaction_id}, {model_name}"
                )

            if abs(model_ref_ts - ref_ts) > 1e-6:
                raise ValueError(
                    f"Reference TS energy mismatch: {reaction_id}, {model_name}"
                )

            # Calculate MLIP activation barrier.
            model_r = reactant.info["model_energy"]
            model_ts = ts.info["model_energy"]

            predicted_barrier = (model_ts - model_r) * EV_TO_KCAL

            results[model_name].append(predicted_barrier)

            # Write individual reaction structures for the app.
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)

            write(
                structs_dir / f"{reaction_id}.xyz",
                [reactant, ts],
            )

    return results


# ---------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------


@pytest.fixture
def get_mae(barrier_heights) -> dict[str, float]:
    """
    Calculate mean absolute error for reaction barriers.

    Will add more statistic in the future.

    Parameters
    ----------
    barrier_heights : dict
        Reference and predicted reaction barrier heights for each model.

    Returns
    -------
    dict[str, float]
        Mean absolute error for each model.
    """
    results = {}
    models = [
        model_name
        for model_name in MODELS
        if os.path.exists(f"{CALC_PATH}/{model_name}")
    ]
    for model_name in models:
        results[model_name] = mae(
            barrier_heights["ref"],
            barrier_heights[model_name],
        )

    return results


# ---------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------


@pytest.fixture
@build_table(
    filename=OUT_PATH / "glucose_barriers_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(get_mae: dict[str, float]) -> dict[str, dict]:
    """
    Calculate benchmark metrics.

    Parameters
    ----------
    get_mae : callable
        Function used to calculate mean absolute error.

    Returns
    -------
    dict
        Calculated benchmark metrics.
    """
    return {
        "MAE": get_mae,
    }


# ---------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------


@pytest.mark.framework("mace-polar-1")
def test_glucose_barriers(metrics: dict[str, dict]) -> None:
    """
    Test glucose reaction barrier metrics.

    Parameters
    ----------
    metrics : dict
        Calculated glucose benchmark metrics.
    """
    return
