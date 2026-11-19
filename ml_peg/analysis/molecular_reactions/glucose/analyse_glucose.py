"""Analyse the glucose reaction benchmark."""

from __future__ import annotations

from pathlib import Path
import warnings

from ase import units
from ase.io import read, write
import numpy as np
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
from ml_peg.models.get_models import get_model_names

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "glucose" / "outputs"

DATASET_PATH = CALCS_ROOT / "molecular_reactions" / "glucose" / "glucose_dataset.xyz"

OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "glucose"

OUTPUT_FILENAME = "glucose_calculated.xyz"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

EV_TO_KCAL = units.mol / units.kcal

# This list is populated by barrier_heights().
# plot_parity reads it after the decorated function has returned.
REACTION_LABELS: list[str] = []


# ---------------------------------------------------------------------------
# Reference dataset
# ---------------------------------------------------------------------------
def load_reference_reactions() -> dict[str, dict]:
    """
    Load and validate all reference reactant-TS pairs.

    Returns
    -------
    dict[str, dict]
        Reaction ID mapped to its reactant and transition-state structures.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Glucose reference dataset not found: {DATASET_PATH}")

    structures = read(DATASET_PATH, index=":")

    reactions: dict[str, dict] = {}

    for atoms in structures:
        reaction_id = str(atoms.info["reaction_id"]).strip().strip('"')
        structure_type = atoms.info.get("type")

        if reaction_id is None:
            raise ValueError("A structure in the reference dataset has no reaction_id.")

        if structure_type not in ("reactant", "ts"):
            raise ValueError(
                f"Invalid structure type {structure_type!r} for reaction {reaction_id}."
            )

        if "ref_energy" not in atoms.info:
            raise ValueError(
                f"Missing ref_energy for reaction {reaction_id} ({structure_type})."
            )

        reaction_id = str(reaction_id)

        if reaction_id not in reactions:
            reactions[reaction_id] = {}

        if structure_type in reactions[reaction_id]:
            raise ValueError(
                f"Duplicate {structure_type} structure "
                f"for reaction {reaction_id} in reference dataset."
            )

        reactions[reaction_id][structure_type] = atoms

    # Reference data must contain complete reaction pairs.
    incomplete = [
        reaction_id
        for reaction_id, pair in reactions.items()
        if set(pair) != {"reactant", "ts"}
    ]

    if incomplete:
        raise ValueError(
            "Incomplete reference reaction pairs: " + ", ".join(incomplete[:10])
        )

    return dict(sorted(reactions.items()))


# ---------------------------------------------------------------------------
# Mock structures for the application
# ---------------------------------------------------------------------------


def write_glucose_structures(
    reference_reactions: dict[str, dict],
) -> None:
    """
    Write reference structures for the application.

    Parameters
    ----------
    reference_reactions
        A dictionary with reaction index as key with information of reaction.
    """
    structures_dir = OUT_PATH / "mock"
    structures_dir.mkdir(parents=True, exist_ok=True)

    # Remove old files to avoid retaining reactions from an older dataset.
    for old_file in structures_dir.glob("*_rct.xyz"):
        old_file.unlink()

    for reaction_id, pair in reference_reactions.items():
        write(
            structures_dir / f"{reaction_id}_rct.xyz",
            [pair["reactant"], pair["ts"]],
            format="extxyz",
        )


def write_glucose_info(
    reference_reactions: dict[str, dict],
) -> None:
    """
    Write metadata used by the application.

    Parameters
    ----------
    reference_reactions
        A dictionary with reaction index as key with information of reaction.
    """
    info = {
        "filenames": [],
        "elements": [],
    }

    for reaction_id, pair in reference_reactions.items():
        reactant = pair["reactant"]

        info["filenames"].append(reaction_id)
        info["elements"].append(sorted(set(reactant.get_chemical_symbols())))

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    import json

    with (OUT_PATH / "info.json").open("w", encoding="utf8") as file:
        json.dump(info, file, indent=2)


# ---------------------------------------------------------------------------
# Model calculation outputs
# ---------------------------------------------------------------------------


def load_reactions(model_name: str) -> dict[str, dict]:
    """
    Load calculated reactant-TS pairs for one model.

    Parameters
    ----------
    model_name
        A str for matching calculators in ML-PEG.

    Returns
    -------
    dict[str, dict]
        Reaction ID mapped to its reactant and transition-state structures.
    """
    output_file = CALC_PATH / model_name / OUTPUT_FILENAME

    if not output_file.exists():
        warnings.warn(
            f"No glucose calculation output found for {model_name}: "
            f"{output_file}. All predictions will be NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    structures = read(output_file, index=":")

    reactions: dict[str, dict] = {}

    for atoms in structures:
        reaction_id = str(atoms.info["reaction_id"]).strip().strip('"')
        structure_type = atoms.info.get("type")

        if reaction_id is None:
            warnings.warn(
                f"{model_name}: structure without reaction_id was skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if structure_type not in ("reactant", "ts"):
            warnings.warn(
                f"{model_name}: invalid structure type {structure_type!r} "
                f"for reaction {reaction_id}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        reaction_id = str(reaction_id)

        if reaction_id not in reactions:
            reactions[reaction_id] = {}

        if structure_type in reactions[reaction_id]:
            warnings.warn(
                f"{model_name}: duplicate {structure_type} structure "
                f"for reaction {reaction_id}. Keeping the first one.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        reactions[reaction_id][structure_type] = atoms

    # Keep only complete pairs.
    complete_reactions = {}

    for reaction_id, pair in reactions.items():
        if set(pair) != {"reactant", "ts"}:
            warnings.warn(
                f"{model_name}: incomplete reaction {reaction_id}. "
                "Its predicted barrier will be NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        complete_reactions[reaction_id] = pair

    return complete_reactions


# ---------------------------------------------------------------------------
# Reaction barriers and parity plot
# ---------------------------------------------------------------------------


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_glucose_barriers.json",
    title="Glucose reaction barriers",
    x_label="Predicted barrier / kcal/mol",
    y_label="Reference barrier / kcal/mol",
    hoverdata={"Reaction ID": REACTION_LABELS},
)
def barrier_heights() -> dict[str, list]:
    """
    Calculate reference and predicted reaction barriers.

    Returns
    -------
    dict[str, list]
        A dictionary with keys of model name and a list to store barriers.
    """
    reference_reactions = load_reference_reactions()
    reaction_ids = list(reference_reactions)

    # Populate labels before plot_parity constructs the hover data.
    REACTION_LABELS[:] = reaction_ids

    # Generate all application data from the reference dataset.
    write_glucose_structures(reference_reactions)
    write_glucose_info(reference_reactions)

    # Load outputs for all models selected in this analysis run.
    model_reactions = {model_name: load_reactions(model_name) for model_name in MODELS}

    results = {
        "ref": [],
        **{model_name: [] for model_name in MODELS},
    }

    for reaction_id in reaction_ids:
        reference_pair = reference_reactions[reaction_id]

        reference_reactant_energy = float(reference_pair["reactant"].info["ref_energy"])
        reference_ts_energy = float(reference_pair["ts"].info["ref_energy"])

        reference_barrier = (
            reference_ts_energy - reference_reactant_energy
        ) * EV_TO_KCAL

        results["ref"].append(reference_barrier)

        for model_name in MODELS:
            pair = model_reactions[model_name].get(reaction_id)

            # Missing reaction: preserve its position and assign NaN.
            if pair is None:
                results[model_name].append(float("nan"))
                continue

            reactant_energy = pair["reactant"].info.get("model_energy")
            ts_energy = pair["ts"].info.get("model_energy")

            # Missing or failed energy calculation.
            if reactant_energy is None or ts_energy is None:
                results[model_name].append(float("nan"))
                continue

            try:
                reactant_energy = float(reactant_energy)
                ts_energy = float(ts_energy)
            except (TypeError, ValueError):
                results[model_name].append(float("nan"))
                continue

            if not (np.isfinite(reactant_energy) and np.isfinite(ts_energy)):
                results[model_name].append(float("nan"))
                continue

            predicted_barrier = (ts_energy - reactant_energy) * EV_TO_KCAL

            results[model_name].append(predicted_barrier)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def glucose_mae(
    barrier_heights: dict[str, list],
) -> dict[str, float]:
    """
    Calculate MAE for each model.

    Parameters
    ----------
    barrier_heights
        A dictionary with model names and the barrier heights.

    Returns
    -------
    dict[str, float]
        A dictionary with keys of MLIP name and a float of MAE.
    """
    reference = barrier_heights["ref"]

    return {
        model_name: mae(reference, barrier_heights[model_name]) for model_name in MODELS
    }


@pytest.fixture
@build_table(
    thresholds=DEFAULT_THRESHOLDS,
    filename=OUT_PATH / "glucose_barriers_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    glucose_mae: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Build the benchmark metrics table.

    Parameters
    ----------
    glucose_mae
        A dictionary of calculated MAE.

    Returns
    -------
    dict[str, dict[str, float]]
        A dictioanry as a key of model name and a dict of MAE.
    """
    return {
        "MAE": glucose_mae,
    }


def test_glucose(metrics: dict) -> None:
    """
    Run glucose benchmark analysis.

    Parameters
    ----------
    metrics
        A dictionary of available metrics.
    """
