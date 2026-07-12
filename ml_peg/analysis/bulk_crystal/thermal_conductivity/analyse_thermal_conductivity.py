"""Analyse thermal conductivity benchmark."""

from __future__ import annotations

from pathlib import Path
import traceback
import warnings

from ase.io import read
import h5py
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.bulk_crystal.thermal_conductivity import thermal_conductivity as tc
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "thermal_conductivity" / "outputs"
REF_PATH = CALCS_ROOT / "bulk_crystal" / "thermal_conductivity" / "data"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "thermal_conductivity"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

STRUCTURE_FILE = REF_PATH / "phononDB-PBE-structures.extxyz"


def get_system_names() -> list[str]:
    """
    Get list of thermal conductivity system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    system_ids = []

    atoms_list = read(STRUCTURE_FILE, index=":")

    for atoms in atoms_list:
        system_names.append(f"{atoms.info['name']}")
        system_ids.append(f"{atoms.info[tc.TCKeys.mat_id]}")
    return [x for _, x in sorted(zip(system_ids, system_names, strict=False))]


def get_system_ids() -> list[str]:
    """
    Get list of thermal conductivity system IDs.

    Returns
    -------
    list[str]
        List of system IDs from structure files.
    """
    system_ids = []
    atoms_list = read(STRUCTURE_FILE, index=":")
    for atoms in atoms_list:
        system_ids.append(f"{atoms.info[tc.TCKeys.mat_id]}")
    return sorted(system_ids)


def calc_kappa_metrics_from_dfs(
    df_pred: pd.DataFrame, df_true: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-material thermal-conductivity prediction metrics from two dataframes.

    This function takes raw ML predictions and DFT reference results and computes
    benchmark metrics. It handles array-type columns (e.g., stress tensors and
    mode-resolved properties), calculates averaged quantities, and computes SRD,
    SRE, and SRME.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        DataFrame containing ML model predictions with columns for thermal
        conductivity tensors, mode-resolved properties, and other structural
        information.
    df_true : pandas.DataFrame
        DataFrame containing DFT reference calculations with the same structure
        as `df_pred`.

    Returns
    -------
    pandas.DataFrame
        A copy of `df_pred` with additional columns for benchmark metrics, e.g.:

        - SRD: Symmetric Relative Difference between ML and DFT conductivities
        - SRE: Absolute value of SRD
        - SRME: Mode-resolved error
        - DFT_kappa_tot_avg: Reference DFT conductivity values
    """
    # Remove precomputed columns
    df_pred[tc.TCKeys.kappa_tot_avg] = df_pred[tc.TCKeys.kappa_tot_rta].map(
        tc.calculate_kappa_avg
    )

    df_pred[tc.TCKeys.srd] = (
        2
        * (df_pred[tc.TCKeys.kappa_tot_avg] - df_true[tc.TCKeys.kappa_tot_avg])
        / (df_pred[tc.TCKeys.kappa_tot_avg] + df_true[tc.TCKeys.kappa_tot_avg])
    )

    # turn temperature list to the first temperature (300K) TODO: allow multiple
    # temperatures to be tested
    df_pred[tc.TCKeys.srd] = df_pred[tc.TCKeys.srd].map(
        lambda x: x if isinstance(x, float) else x[0]
    )

    # We substitute NaN values with 0 predicted conductivity, yielding -2 for SRD
    df_pred[tc.TCKeys.srd] = df_pred[tc.TCKeys.srd].fillna(-2)

    df_pred[tc.TCKeys.sre] = df_pred[tc.TCKeys.srd].abs()

    df_pred[tc.TCKeys.srme] = calc_kappa_srme_dataframes(df_pred, df_true)

    df_pred[tc.TCKeys.true_kappa_tot_avg] = df_true[tc.TCKeys.kappa_tot_avg]

    return df_pred


def calc_kappa_srme_dataframes(
    df_pred: pd.DataFrame, df_true: pd.DataFrame
) -> list[float]:
    """
    Calculate the Symmetric Relative Mean Error (SRME) for each material.

    SRME is a comprehensive metric that evaluates both the overall accuracy of thermal
    conductivity predictions and the accuracy of individual phonon mode contributions.
    It is symmetric (like SRD) to treat over- and under-predictions equally, and
    accounts for the mean error across all phonon modes weighted by their contributions.

    The function handles various edge cases:
    - Returns 2.0 for materials with imaginary frequencies (unphysical predictions)
    - Returns 2.0 for materials where symmetry is broken during relaxation
    - Returns 2.0 for failed calculations or missing data

    Parameters
    ----------
    df_pred : pd.DataFrame
        ML predictions including mode-resolved properties.
    df_true : pd.DataFrame
        DFT reference data including mode-resolved properties.

    Returns
    -------
    list[float]:
        SRME values for each material. Values are between 0 and 2, where:
        - 0 indicates perfect agreement in both total k and mode-resolved properties
        - 2 indicates complete failure (imaginary frequencies, broken symmetry, etc.)
        - Values in between indicate partial agreement, with lower being better
    """
    srme_list: list[float] = []
    for idx, row_pred in df_pred.iterrows():
        row_true = df_true.loc[idx]

        # NOTE code below just until before return used to be wrapped in try/except in
        # which case SRME=2 was set for the failing material
        has_imag_ph_modes = row_pred.get(tc.TCKeys.has_imag_ph_modes, False)
        if pd.notna(has_imag_ph_modes) and bool(has_imag_ph_modes):
            srme_list.append(2)
            continue
        if relaxed_space_group_number := row_pred.get(tc.TCKeys.final_spg_num):
            if initial_space_group_number := row_pred.get(tc.TCKeys.init_spg_num):
                if relaxed_space_group_number != initial_space_group_number:
                    srme_list.append(2)
                    continue
            elif relaxed_space_group_number != row_true.get(tc.TCKeys.spg_num):
                srme_list.append(2)
                continue
        result = calc_kappa_srme(row_pred, row_true)
        srme_list.append(float(result[0]))  # append the first temperature's SRME

    return srme_list


def calc_kappa_srme(kappas_pred: pd.Series, kappas_true: pd.Series) -> np.ndarray:
    """
    Calculate the Symmetric Relative Mean Error (SRME) for a single material.

        SRME = 2 * (sum|k_pred,i - k_true,i| * w_i) / (k_pred,tot + k_true,tot)

    where:
    - k_pred,i and k_true,i are mode-resolved conductivities for mode i
    - w_i are the mode weights
    - k_pred,tot and k_true,tot are total conductivities

    The calculation involves:
    1. Computing mode-resolved average conductivities if not pre-computed
    2. Calculating the weighted mean absolute error across all modes
    3. Normalizing by the sum of total conductivities to make it symmetric and relative

    Parameters
    ----------
    kappas_pred : Series
        Series containing ML predictions including:
            - kappa_tot_avg: Average total conductivity
            - mode_kappa_tot: Mode-resolved conductivities
            - mode_weights: Mode weights for averaging.
    kappas_true : Series
        Series containing DFT reference data with same structure.

    Returns
    -------
    np.ndarray
        SRME values per temperature, each between 0 and 2, where:
        - 0 indicates perfect agreement in both total k and mode-resolved properties
        - 2 indicates complete disagreement or invalid results
        On error conditions (missing data, NaN values), returns np.array([2.0]).
    """
    if np.any(np.isnan(kappas_true[tc.TCKeys.kappa_tot_avg])):
        raise ValueError("found NaNs in kappa_tot_avg reference values")
    if (  # return highest possible SRME=2 if any of these conditions are met:
        # only have NaN averaged kappa preds
        np.all(np.isnan(kappas_pred[tc.TCKeys.kappa_tot_avg]))
        # some mode-resolved kappa preds are NaN
        or np.any(np.isnan(kappas_pred[tc.TCKeys.kappa_tot_rta]))
        # some mode weights are NaN
        or np.any(np.isnan(kappas_pred[tc.TCKeys.mode_weights]))
    ):
        return np.array([2.0])

    mode_kappa_tot_avgs = {}  # store results for pred and true
    # Try different data sources in order of preference for both pred and true data
    for label, kappas in {"preds": kappas_pred, "true": kappas_true}.items():
        keys = set(kappas.keys())
        if tc.TCKeys.mode_kappa_tot_avg in kappas:
            kappas = kappas[tc.TCKeys.mode_kappa_tot_avg]
        elif tc.TCKeys.mode_kappa_tot_rta in kappas:
            kappas = tc.calculate_kappa_avg(kappas[tc.TCKeys.mode_kappa_tot_rta])
        elif {
            tc.TCKeys.kappa_p_rta,
            tc.TCKeys.kappa_c,
            tc.TCKeys.heat_capacity,
        } <= keys:
            kappas = tc.calculate_kappa_avg(
                tc.calc_mode_kappa_tot(
                    kappas[tc.TCKeys.kappa_p_rta],
                    kappas[tc.TCKeys.kappa_c],
                    kappas[tc.TCKeys.heat_capacity],
                )
            )
        else:
            raise ValueError(
                f"Neither mode_kappa_tot_avg, mode_kappa_tot nor individual kappa\n"
                f"components found in {label}, got\n{keys}"
            )
        mode_kappa_tot_avgs[label] = np.asarray(kappas)

    # calculating microscopic error for all temperatures
    microscopic_error = (
        np.abs(mode_kappa_tot_avgs["preds"] - mode_kappa_tot_avgs["true"]).sum(
            axis=tuple(range(1, np.asarray(mode_kappa_tot_avgs["preds"]).ndim))
        )
        / np.asarray(kappas_pred[tc.TCKeys.mode_weights]).sum()
    )

    denominator = (
        kappas_pred[tc.TCKeys.kappa_tot_avg] + kappas_true[tc.TCKeys.kappa_tot_avg]
    )
    return 2 * microscopic_error / denominator


def _add_missing_error_rows(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    model_name: str,
    result_type: str = "",
) -> pd.DataFrame:
    """
    Warn and add default-error rows for missing result files.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction results.
    ref_df : pd.DataFrame
        Reference results defining the required index.
    model_name : str
        Model name used in the warning.
    result_type : str
        Optional result-type prefix used in the warning.

    Returns
    -------
    pd.DataFrame
        Prediction results reindexed to the reference when rows are missing.
    """
    missing = ref_df.index.difference(df.index)
    if len(missing):
        warnings.warn(
            f"Missing {len(missing)} {result_type}thermal-conductivity result files "
            f"for {model_name}; using default error rows.",
            stacklevel=2,
        )
        return df.reindex(ref_df.index)
    return df


@pytest.fixture
def kappa_stats() -> dict[str, pd.DataFrame]:
    """
    Get thermal conductivity statistics for all models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.
    """
    results: dict[str, pd.DataFrame] = {}
    # ref_df = pd.read_json(REF_PATH / "FT_stats.json", orient="index")

    ref_df = pd.DataFrame.from_dict(
        tc.hdf5_to_dict(h5py.File(REF_PATH / "PBE" / "kappas.hdf5", "r")),
        orient="index",
    )
    ref_df.sort_index(inplace=True)

    fast_ref_df = pd.DataFrame.from_dict(
        tc.hdf5_to_dict(h5py.File(REF_PATH / "PBE" / "fast_kappas.hdf5", "r")),
        orient="index",
    )
    fast_ref_df.sort_index(inplace=True)

    results["ref"] = ref_df

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        pred_df = None
        fast_pred_df = None

        if not model_dir.exists():
            continue

        if (model_dir / "kappas.hdf5").exists():
            current_pred_dict = tc.hdf5_to_dict(
                h5py.File(model_dir / "kappas.hdf5", "r")
            )
            pred_df = pd.DataFrame.from_dict(current_pred_dict, orient="index")
        elif (model_dir / "kappas.json.gz").exists():
            pred_df = pd.read_json(model_dir / "kappas.json.gz").set_index(
                tc.TCKeys.mat_id
            )
        else:
            try:
                current_pred_dict = tc.load_hdf5_subdir_dicts(model_dir, "kappa.hdf5")
                if current_pred_dict:
                    pred_df = pd.DataFrame.from_dict(current_pred_dict, orient="index")
            except Exception:
                print(f"Error loading kappas for {model_name}...")
                traceback.print_exc()

        if (model_dir / "fast_kappas.hdf5").exists():
            current_pred_dict = tc.hdf5_to_dict(
                h5py.File(model_dir / "fast_kappas.hdf5", "r")
            )
            fast_pred_df = pd.DataFrame.from_dict(current_pred_dict, orient="index")
        elif (model_dir / "fast_kappas.json.gz").exists():
            fast_pred_df = pd.read_json(model_dir / "fast_kappas.json.gz").set_index(
                tc.TCKeys.mat_id
            )
        else:
            try:
                current_pred_dict = tc.load_hdf5_subdir_dicts(
                    model_dir, "fast_kappa.hdf5"
                )
                if current_pred_dict:
                    fast_pred_df = pd.DataFrame.from_dict(
                        current_pred_dict, orient="index"
                    )
            except Exception:
                print(f"Error loading fast_kappas for {model_name}...")
                traceback.print_exc()

        if pred_df is not None:
            pred_df = _add_missing_error_rows(pred_df, ref_df, model_name)
        if fast_pred_df is not None:
            fast_pred_df = _add_missing_error_rows(
                fast_pred_df, fast_ref_df, model_name, "fast "
            )

        if pred_df is None and fast_pred_df is None:
            continue

        if pred_df is not None and fast_pred_df is None:
            pred_df.sort_index(inplace=True)
            pred_df = calc_kappa_metrics_from_dfs(pred_df, ref_df)
        elif pred_df is None:
            fast_pred_df.sort_index(inplace=True)
            pred_df = calc_kappa_metrics_from_dfs(fast_pred_df, fast_ref_df)
            pred_df["fast_sre"] = pred_df[tc.TCKeys.sre]
            pred_df["fast_srme"] = pred_df[tc.TCKeys.srme]
            pred_df.drop(columns=[tc.TCKeys.sre, tc.TCKeys.srme], inplace=True)
        elif fast_pred_df is not None:
            pred_df.sort_index(inplace=True)
            fast_pred_df.sort_index(inplace=True)
            pred_df = calc_kappa_metrics_from_dfs(pred_df, ref_df)
            fast_pred_df = calc_kappa_metrics_from_dfs(fast_pred_df, fast_ref_df)
            pred_df["fast_sre"] = fast_pred_df[tc.TCKeys.sre]
            pred_df["fast_srme"] = fast_pred_df[tc.TCKeys.srme]
        else:
            print(
                f"Unexpected case for model {model_name} with pred_df is not None: "
                f"{pred_df is not None} and fast_pred_df is not None: "
                f"{fast_pred_df is not None}"
            )
            continue

        results[model_name] = pred_df

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_thermal_conductivity.json",
    title="Thermal Conductivity Parity Plot",
    x_label="Predicted k (W/mK)",
    y_label="Reference k (W/mK)",
    hoverdata={
        "System": get_system_names(),
        "System ID": get_system_ids(),
    },
)
def conductivity(kappa_stats: dict[str, pd.DataFrame]) -> dict[str, list]:
    """
    Get thermal conductivity parity plot data.

    Parameters
    ----------
    kappa_stats : dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.

    Returns
    -------
    dict[str, list]
        Dictionary mapping model names to lists of thermal conductivity values.
    """
    conductivity_data = {}
    for model_name in MODELS:
        if model_name not in kappa_stats:
            continue
        df = kappa_stats[model_name]
        if tc.TCKeys.kappa_tot_avg not in df:
            continue
        df = df.reindex(kappa_stats["ref"].index)
        conductivity_data[model_name] = df[tc.TCKeys.kappa_tot_avg].tolist()
    conductivity_data["ref"] = kappa_stats["ref"][tc.TCKeys.kappa_tot_avg].tolist()
    return conductivity_data


@pytest.fixture
def mean_sre(kappa_stats: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Calculate mean Symmetric Relative Error (SRE) for each model.

    Parameters
    ----------
    kappa_stats : dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to their mean SRE values.
    """
    sre_values = {}
    for model_name in MODELS:
        if model_name not in kappa_stats:
            continue
        df = kappa_stats[model_name]
        if tc.TCKeys.sre not in df:
            continue
        sre_values[model_name] = df[tc.TCKeys.sre].mean()
    return sre_values


@pytest.fixture
def mean_srme(kappa_stats: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Calculate mean Symmetric Relative Mean Error (SRME) for each model.

    Parameters
    ----------
    kappa_stats : dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to their mean SRME values.
    """
    srme_values = {}
    for model_name in MODELS:
        if model_name not in kappa_stats:
            continue
        df = kappa_stats[model_name]
        if tc.TCKeys.srme not in df:
            continue
        srme_values[model_name] = df[tc.TCKeys.srme].mean()
    return srme_values


@pytest.fixture
def instability(kappa_stats: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Calculate instability metric for each model.

    Parameters
    ----------
    kappa_stats : dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to their instability values
        (fraction of unstable materials).
    """
    instability_values = {}
    ref_length = len(kappa_stats["ref"])
    for model_name in MODELS:
        if model_name not in kappa_stats:
            continue
        df = kappa_stats[model_name]
        if tc.TCKeys.has_imag_ph_modes not in df:
            continue
        has_imag_ph_modes = df[tc.TCKeys.has_imag_ph_modes].values
        has_imag_ph_modes_filtered = has_imag_ph_modes[pd.notna(has_imag_ph_modes)]
        instability_values[model_name] = has_imag_ph_modes_filtered.sum() / ref_length

    return instability_values


@pytest.fixture
def failure(kappa_stats: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Calculate failure metric for each model.

    Parameters
    ----------
    kappa_stats : dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to their failure values
        (fraction of failed calculations).
    """
    failure_values = {}
    ref_length = len(kappa_stats["ref"])
    for model_name in MODELS:
        if model_name not in kappa_stats:
            continue
        df = kappa_stats[model_name]
        if tc.TCKeys.has_imag_ph_modes not in df:
            continue
        has_nan_frequency = df[tc.TCKeys.has_imag_ph_modes].isna().values
        failure_values[model_name] = has_nan_frequency.sum() / ref_length

    return failure_values


@pytest.fixture
def mean_fast_sre(kappa_stats: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Calculate mean Symmetric Relative Error (SRE) for each model with "fast" settings.

    In "fast" settings the qpoint mesh is reduced in the phonon interaction strength c
    alculation to allow quicker benchmarking.

    Parameters
    ----------
    kappa_stats : dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to their mean SRE values.
    """
    sre_values = {}
    for model_name in MODELS:
        if model_name not in kappa_stats:
            continue
        df = kappa_stats[model_name]
        if "fast_sre" not in df:
            continue
        sre_values[model_name] = df["fast_sre"].mean()
    return sre_values


@pytest.fixture
def mean_fast_srme(kappa_stats: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Calculate mean Symmetric Relative Mean Error (SRME) with "fast" settings.

    Calculation processes each model. In "fast" settings the qpoint mesh is reduced
    in the phonon interaction strength calculation.

    Parameters
    ----------
    kappa_stats : dict[str, pd.DataFrame]
        Dictionary of DataFrames containing thermal conductivity statistics for
        each model.

    Returns
    -------
    dict[str, float]
        Dictionary mapping model names to their mean SRME values.
    """
    srme_values = {}
    for model_name in MODELS:
        if model_name not in kappa_stats:
            continue
        df = kappa_stats[model_name]
        if "fast_srme" not in df:
            continue
        srme_values[model_name] = df["fast_srme"].mean()
    return srme_values


@pytest.fixture
@build_table(
    filename=OUT_PATH / "thermal_conductivity.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    mean_srme: dict[str, float],
    mean_sre: dict[str, float],
    instability: dict[str, float],
    failure: dict[str, float],
    mean_fast_sre: dict[str, float],
    mean_fast_srme: dict[str, float],
) -> dict[str, dict]:
    """
    Get all thermal conductivity metrics.

    Parameters
    ----------
    mean_srme : dict[str, float]
        Mean Symmetric Relative Mean Error for each model.
    mean_sre : dict[str, float]
        Mean Symmetric Relative Error for each model.
    instability : dict[str, float]
        Fraction of unstable materials (has imaginary frequencies) for each model.
    failure : dict[str, float]
        Fraction of failed calculations (NaN frequencies) for each model.
    mean_fast_sre : dict[str, float]
        Mean Symmetric Relative Error for each model (fast version).
    mean_fast_srme : dict[str, float]
        Mean Symmetric Relative Mean Error for each model (fast version).

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "kSRE": mean_sre,
        "kSRME": mean_srme,
        "Instability": instability,
        "Failure": failure,
        "Fast kSRE": mean_fast_sre,
        "Fast kSRME": mean_fast_srme,
    }


def test_thermal_conducticity(metrics: dict[str, dict]):
    """
    Run thermal conductivity benchmark tests.

    Parameters
    ----------
    metrics : dict[str, dict]
        Metric names and values for all models.
    """
    return
