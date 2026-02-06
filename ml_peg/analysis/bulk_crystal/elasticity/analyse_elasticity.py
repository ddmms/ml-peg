"""Analyse elasticity benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import (
    build_table,
    plot_density_scatter,
)
from ml_peg.analysis.utils.utils import (
    build_density_inputs,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "elasticity" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "elasticity"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

K_COLUMN = "K_vrh"
G_COLUMN = "G_vrh"
E_TENSOR_COLUMN = "elastic_tensor"
SYMMETRY_COLUMN = "crystal_system"

# Sources:
# Physical Properties of Crystals: An Introduction (pp 215)
# https://ocean-jh.github.io/elastic-mechanics/

VOIGT_SYMMETRIES = {
    "triclinic": {
        "indices": [
            (0, 0),
            (1, 1),
            (2, 2),
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 3),
            (4, 4),
            (5, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        "labels": [
            "C11",
            "C22",
            "C33",
            "C12",
            "C13",
            "C23",
            "C14",
            "C15",
            "C16",
            "C24",
            "C25",
            "C26",
            "C34",
            "C35",
            "C36",
            "C44",
            "C55",
            "C66",
            "C45",
            "C46",
            "C56",
        ],
    },
    "monoclinic": {
        "indices": [
            (0, 0),
            (1, 1),
            (2, 2),
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (0, 3),
            (1, 3),
            (2, 3),
            (4, 5),
        ],
        "labels": [
            "C11",
            "C22",
            "C33",
            "C12",
            "C13",
            "C23",
            "C44",
            "C55",
            "C66",
            "C14",
            "C24",
            "C34",
            "C56",
        ],
    },
    "orthorhombic": {
        "indices": [
            (0, 0),
            (1, 1),
            (2, 2),
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ],
        "labels": ["C11", "C22", "C33", "C12", "C13", "C23", "C44", "C55", "C66"],
    },
    "tetragonal_7": {
        "indices": [(0, 0), (0, 1), (0, 2), (2, 2), (3, 3), (5, 5), (0, 5)],
        "labels": ["C11", "C12", "C13", "C33", "C44", "C66", "C16"],
    },
    "tetragonal_6": {
        "indices": [(0, 0), (0, 1), (0, 2), (2, 2), (3, 3), (5, 5)],
        "labels": ["C11", "C12", "C13", "C33", "C44", "C66"],
    },
    "trigonal_7": {
        "indices": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (2, 2), (3, 3)],
        "labels": ["C11", "C12", "C13", "C14", "C15", "C33", "C44"],
    },
    "trigonal_6": {
        "indices": [(0, 0), (0, 1), (0, 2), (0, 3), (2, 2), (3, 3)],
        "labels": ["C11", "C12", "C13", "C14", "C33", "C44"],
    },
    "hexagonal": {
        "indices": [(0, 0), (0, 1), (0, 2), (2, 2), (3, 3)],
        "labels": ["C11", "C12", "C13", "C33", "C44"],
    },
    "cubic": {
        "indices": [(0, 0), (0, 1), (3, 3)],
        "labels": ["C11", "C12", "C44"],
    },
}


def get_independent_cs(c_ref, c_arr, crystal_symmetry, rtol=0.10):
    """
    Extract symmetry-independent elastic constants.

    Parameters
    ----------
    c_ref : (6, 6) array_like
        Reference Voigt tensor.
    c_arr : (6, 6) array_like
        Comparison Voigt tensor.
    crystal_symmetry : str
        Crystal symmetry label.
    rtol : float, optional
        Relative tolerance for consistency checks (default: 0.10).

    Returns
    -------
    tuple of ndarray
        Independent elastic constants for reference and comparison tensors.
    """

    def _allclose(values):
        """
        Check whether all values in an array are close to the first element.

        Parameters
        ----------
        values : array_like
            Sequence of numeric values to compare. The first value is taken
            as the reference.

        Returns
        -------
        bool
            True if all values are close to the reference value within the
            specified relative tolerance; False otherwise.

        Notes
        -----
        If the reference value is zero, all elements must be exactly zero to
        return True.
        """
        values = np.asarray(values)
        ref = values[0]
        if ref == 0.0:
            return np.all(values == 0.0)
        return np.all(np.abs(values - ref) <= rtol * np.abs(ref))

    pass_symm_check = True
    for c in [c_ref, c_arr]:
        if c.shape != (6, 6):
            raise ValueError("C must be a 6x6 matrix")

        if crystal_symmetry == "cubic":
            if not _allclose([c[0, 0], c[1, 1], c[2, 2]]):
                pass_symm_check = False
            if not _allclose([c[3, 3], c[4, 4], c[5, 5]]):
                pass_symm_check = False

        elif crystal_symmetry in (
            "hexagonal",
            "trigonal_6",
            "trigonal_7",
            "tetragonal_6",
            "tetragonal_7",
        ):
            if not _allclose([c[0, 0], c[1, 1]]):
                pass_symm_check = False
            if not _allclose([c[3, 3], c[4, 4]]):
                pass_symm_check = False

        elif crystal_symmetry in ("orthorhombic", "monoclinic", "triclinic"):
            pass_symm_check = False

    if pass_symm_check:
        voigt = VOIGT_SYMMETRIES[crystal_symmetry]
        rows, cols = zip(*voigt["indices"], strict=False)
        return c_ref[rows, cols], c_arr[rows, cols]

    voigt = VOIGT_SYMMETRIES["triclinic"]
    rows, cols = zip(*voigt["indices"], strict=False)
    vals_ref = c_ref[rows, cols]
    vals_arr = c_arr[rows, cols]
    tol = 1e-3
    mask = (np.abs(vals_arr) > tol) | (np.abs(vals_ref) > tol)
    return vals_ref[mask], vals_arr[mask]


def _filter_results(df: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, int]:
    """
    Filter outlier predictions and return remaining data with exclusion count.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing raw benchmark results.
    model_name : str
        Model whose columns should be filtered..

    Returns
    -------
    tuple[pd.DataFrame, int]
        Filtered dataframe and number of excluded entries.
    """
    mask_bulk = df[f"{K_COLUMN}_{model_name}"].between(-50, 600)
    mask_shear = df[f"{G_COLUMN}_{model_name}"].between(-50, 600)
    valid = df[mask_bulk & mask_shear].copy()
    excluded = len(df) - len(valid)
    return valid, excluded


@pytest.fixture
def elasticity_stats() -> dict[str, dict[str, Any]]:
    """
    Load and cache processed benchmark statistics per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Aggregated statistics including bulk, shear, elastic tensors, and exclusions.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        results_path = CALC_PATH / model_name / "moduli_results.csv"
        df = pd.read_csv(results_path)
        filtered, excluded = _filter_results(df, model_name)

        stats[model_name] = {
            "bulk": {
                "ref": filtered[f"{K_COLUMN}_DFT"].tolist(),
                "pred": filtered[f"{K_COLUMN}_{model_name}"].tolist(),
            },
            "shear": {
                "ref": filtered[f"{G_COLUMN}_DFT"].tolist(),
                "pred": filtered[f"{G_COLUMN}_{model_name}"].tolist(),
            },
            "elastic_tensor": {
                "ref": filtered[f"{E_TENSOR_COLUMN}_DFT"].tolist(),
                "pred": filtered[f"{E_TENSOR_COLUMN}_{model_name}"].tolist(),
            },
            "crystal_system": {
                "ref": filtered[f"{SYMMETRY_COLUMN}_DFT"].tolist(),
                "pred": filtered[f"{SYMMETRY_COLUMN}_{model_name}"].tolist(),
            },
            "excluded": excluded,
        }

    return stats


@pytest.fixture
def bulk_mae(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """
    Compute MAE for bulk modulus predictions.

    Parameters
    ----------
    elasticity_stats : dict
        Aggregated benchmark statistics.

    Returns
    -------
    dict[str, float | None]
        Bulk modulus MAE per model.
    """
    results: dict[str, float | None] = {}
    for model_name in MODELS:
        prop = elasticity_stats.get(model_name, {}).get("bulk")
        results[model_name] = mae(prop["ref"], prop["pred"])
    return results


@pytest.fixture
def shear_mae(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """
    Compute MAE for shear modulus predictions.

    Parameters
    ----------
    elasticity_stats : dict
        Aggregated benchmark statistics.

    Returns
    -------
    dict[str, float | None]
        Shear modulus MAE per model.
    """
    results: dict[str, float | None] = {}
    for model_name in MODELS:
        prop = elasticity_stats.get(model_name, {}).get("shear")
        results[model_name] = mae(prop["ref"], prop["pred"])
    return results


@pytest.fixture
def elastic_tensor_mae(
    elasticity_stats: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    """
    Compute MAE for elastic tensor predictions.

    Parameters
    ----------
    elasticity_stats : dict
        Aggregated benchmark statistics.

    Returns
    -------
    dict[str, float | None]
        Elastic tensor MAE per model.
    """

    def _str_to_array(s: str) -> np.ndarray:
        """
        Convert a string representation of a 6x6 elastic tensor into a NumPy array.

        Parameters
        ----------
        s : str
            String representation of a 6x6 elastic tensor. The string may
            contain square brackets and whitespace-separated numbers.

        Returns
        -------
        np.ndarray
            A 6x6 NumPy array containing the numeric values from the input string.
        """
        s_clean = s.replace("[", "").replace("]", "")
        return np.fromstring(s_clean, sep=" ").reshape(6, 6)

    results: dict[str, float | None] = {}
    for model_name in MODELS:
        prop = elasticity_stats.get(model_name, {}).get("elastic_tensor")
        crystal_system = elasticity_stats.get(model_name, {}).get("crystal_system")
        if prop is None or not prop["ref"]:
            results[model_name] = None
            continue

        tensor_maes = []
        for r, p, cs in zip(
            prop["ref"], prop["pred"], crystal_system["ref"], strict=False
        ):
            r_arr = np.asarray(r) if isinstance(r, np.ndarray) else _str_to_array(r)
            p_arr = np.asarray(p) if isinstance(p, np.ndarray) else _str_to_array(p)
            r_arr[np.abs(r_arr) < 0.01] = 0.0
            p_arr[np.abs(p_arr) < 0.01] = 0.0
            r_vals, p_vals = get_independent_cs(r_arr, p_arr, cs)
            tensor_maes.append(mae(r_vals, p_vals))
        results[model_name] = float(np.mean(tensor_maes))
    return results


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_bulk_density.json",
    title="Bulk modulus density plot",
    x_label="Reference bulk modulus / GPa",
    y_label="Predicted bulk modulus / GPa",
    annotation_metadata={"excluded": "Excluded"},
)
def bulk_density(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Prepare density scatter data for bulk modulus.

    Parameters
    ----------
    elasticity_stats : dict
        Aggregated benchmark statistics.

    Returns
    -------
    dict[str, dict]
        Density-scatter data per model.
    """
    return build_density_inputs(MODELS, elasticity_stats, "bulk", metric_fn=mae)


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_shear_density.json",
    title="Shear modulus density plot",
    x_label="Reference shear modulus / GPa",
    y_label="Predicted shear modulus / GPa",
    annotation_metadata={"excluded": "Excluded"},
)
def shear_density(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Prepare density scatter data for shear modulus.

    Parameters
    ----------
    elasticity_stats : dict
        Aggregated benchmark statistics.

    Returns
    -------
    dict[str, dict]
        Density-scatter data per model.
    """
    return build_density_inputs(MODELS, elasticity_stats, "shear", metric_fn=mae)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "elasticity_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    bulk_mae: dict[str, float | None],
    shear_mae: dict[str, float | None],
    elastic_tensor_mae: dict[str, float | None],
) -> dict[str, dict]:
    """
    Aggregate all elasticity metrics.

    Parameters
    ----------
    bulk_mae : dict
        Bulk modulus MAE per model.
    shear_mae : dict
        Shear modulus MAE per model.
    elastic_tensor_mae : dict
        Elastic tensor MAE per model.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping metric names to model values.
    """
    return {
        "Bulk modulus MAE": bulk_mae,
        "Shear modulus MAE": shear_mae,
        "Elasticity tensor MAE": elastic_tensor_mae,
    }


def test_elasticity(
    metrics: dict[str, dict],
    bulk_density: dict[str, dict],
    shear_density: dict[str, dict],
) -> None:
    """
    Run elasticity analysis.

    Parameters
    ----------
    metrics : dict
        Benchmark metric values.
    bulk_density : dict
        Density scatter data for bulk modulus.
    shear_density : dict
        Density scatter data for shear modulus.
    """
    return
