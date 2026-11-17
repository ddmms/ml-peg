"""
Analyse NABP30 base pair interaction and binding energy benchmark.

J. Sponer, P. Jurecka, P. Hobza, J. Am. Chem. Soc. 2004, 126, 10142-10151.
"""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase import Atoms, units
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
from ml_peg.calcs.non_covalent_interactions.NABP30.calc_NABP30 import FRAMES
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map()

EV_TO_KCAL = units.mol / units.kcal

CALC_PATH = CALCS_ROOT / "non_covalent_interactions" / "NABP30" / "outputs"
OUT_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "NABP30"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Reference values and the input structure (shown in the app) are taken from the base
# pair at the input geometry (first frame) of the mock calculation, which is always
# complete.
INFO = get_struct_info(
    calc_path=CALC_PATH,
    index=0,
    info_keys=["ref_interaction_energy", "ref_binding_energy"],
    include_filenames=True,
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)

LABELS = INFO["filenames"]
REF_INTERACTION_ENERGIES = [
    energy * EV_TO_KCAL for energy in INFO["ref_interaction_energy"]
]
REF_BINDING_ENERGIES = [energy * EV_TO_KCAL for energy in INFO["ref_binding_energy"]]


def read_frames(model_name: str, label: str) -> dict[str, list[Atoms]] | None:
    """
    Read the structures calculated for a base pair.

    Parameters
    ----------
    model_name
        Name of the model.
    label
        Label of the base pair.

    Returns
    -------
    dict[str, list[Atoms]] | None
        Base pair and its two bases at the ``"input"`` geometry, and ``"relaxed"``,
        or None if the file is missing, cannot be read, or its frames are not as
        written by the calculation (``FRAMES``).
    """
    struct_path = CALC_PATH / model_name / f"{label}.xyz"
    if not struct_path.exists():
        return None
    try:
        frames = read(struct_path, index=":")
    except Exception as exc:
        warn(f"{model_name}, {label}: cannot read {struct_path}: {exc}", stacklevel=1)
        return None
    if [(f.info.get("geometry"), f.info.get("fragment")) for f in frames] != FRAMES:
        warn(
            f"{model_name}, {label}: unexpected frames in {struct_path}; "
            "please rerun the calculation",
            stacklevel=1,
        )
        return None
    return {"input": frames[:3], "relaxed": frames[3:]}


def energy_difference(frames: list[Atoms]) -> float:
    """
    Get the energy of a base pair relative to its two bases.

    Parameters
    ----------
    frames
        Base pair and its two bases.

    Returns
    -------
    float
        E(AB) - E(A) - E(B) in kcal/mol, or NaN if an energy is missing.
    """
    pair, base_a, base_b = (f.info.get("model_energy", np.nan) for f in frames)
    return (pair - base_a - base_b) * EV_TO_KCAL


def get_interaction_energy(frames: dict[str, list[Atoms]] | None) -> float:
    """
    Get the interaction energy of a base pair at its input geometry.

    Parameters
    ----------
    frames
        Structures calculated for the base pair, from ``read_frames``.

    Returns
    -------
    float
        Interaction energy in kcal/mol, or NaN if the calculation failed.
    """
    if frames is None:
        return np.nan
    return energy_difference(frames["input"])


def get_binding_energy(frames: dict[str, list[Atoms]] | None) -> float:
    """
    Get the binding energy of a base pair from its relaxed structures.

    Parameters
    ----------
    frames
        Structures calculated for the base pair, from ``read_frames``.

    Returns
    -------
    float
        Binding energy in kcal/mol, or NaN if any relaxation did not converge or
        changed covalent bonds.
    """
    if frames is None:
        return np.nan
    relaxed = frames["relaxed"]
    if not all(
        f.info.get("converged") and f.info.get("bonds_unchanged") for f in relaxed
    ):
        return np.nan
    return energy_difference(relaxed)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="Interaction energies at the reference geometries",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "Labels": LABELS,
    },
)
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all base pairs.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted interaction energies.
    """
    results = {"ref": list(REF_INTERACTION_ENERGIES)} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        for label in LABELS:
            frames = read_frames(model_name, label)
            results[model_name].append(get_interaction_energy(frames))

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_binding_energies.json",
    title="Binding energies, structures relaxed by the model",
    x_label="Predicted binding energy / kcal/mol",
    y_label="Reference binding energy / kcal/mol",
    hoverdata={
        "Labels": LABELS,
    },
)
def binding_energies() -> dict[str, list]:
    """
    Get binding energies for all base pairs.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted binding energies.
    """
    results = {"ref": list(REF_BINDING_ENERGIES)} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        for label in LABELS:
            frames = read_frames(model_name, label)
            results[model_name].append(get_binding_energy(frames))
            if frames is not None:
                # Relaxed base pair, shown in the app
                (OUT_PATH / model_name).mkdir(parents=True, exist_ok=True)
                write(OUT_PATH / model_name / f"{label}.xyz", frames["relaxed"][0])

    return results


@pytest.fixture
def interaction_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for interaction energies.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            interaction_energies["ref"], interaction_energies[model_name]
        )
    return results


@pytest.fixture
def binding_mae(binding_energies) -> dict[str, float]:
    """
    Get mean absolute error for binding energies.

    Parameters
    ----------
    binding_energies
        Dictionary of reference and predicted binding energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted binding energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(binding_energies["ref"], binding_energies[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "nabp30_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    interaction_mae: dict[str, float], binding_mae: dict[str, float]
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    interaction_mae
        Mean absolute errors of interaction energies for all models.
    binding_mae
        Mean absolute errors of binding energies for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Interaction energy MAE": interaction_mae,
        "Binding energy MAE": binding_mae,
    }


def test_nabp30(metrics: dict[str, dict]) -> None:
    """
    Run NABP30 test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
