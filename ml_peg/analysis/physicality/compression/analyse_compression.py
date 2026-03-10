"""Analyse compression benchmark."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from ase.formula import Formula
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.signal import find_peaks

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "compression" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "compression"
CURVE_PATH = OUT_PATH / "curves"
FIGURE_PATH = OUT_PATH / "figures"

# Palette for overlaid structure curves
_PALETTE = [
    "royalblue",
    "firebrick",
    "seagreen",
    "darkorange",
    "mediumpurple",
    "deeppink",
    "teal",
    "goldenrod",
    "slategray",
    "crimson",
]

# Symlog scaling parameters
LINEAR_FRAC: float = 0.6
LINTHRESH_ENERGY: float = 10.0  # eV/atom
SYMLOG_DECADES: int = 5
DEFAULT_WEIGHTS = {
    "Holes": 1.0,
    "Energy minima": 0.1,
    "Deep Energy minima": 1.0,
    "Energy inflections": 0.1,
    "Big Pressure sign flips": 1.0,
    "Pressure sign flips": 0.1,
    "ρ(-E,Vsmall)": 1.0,
    "ρ(E,Vlarge)": 0.1,
}


def _symlog(
    values: list[float] | np.ndarray,
    linthresh: float = LINTHRESH_ENERGY,
    linear_frac: float = LINEAR_FRAC,
    decades: int = SYMLOG_DECADES,
) -> list[float]:
    """
    Apply a symmetric-log transform to *values*.

    Linear within ``[-linthresh, linthresh]``, logarithmic outside.
    The linear region is scaled so that it occupies *linear_frac* of the
    total axis range (assuming *decades* log decades on each side).

    Parameters
    ----------
    values
        Raw data values.
    linthresh
        Linear threshold.
    linear_frac
        Fraction of the full axis height reserved for the linear region.
    decades
        Number of log decades shown on each side.

    Returns
    -------
    list[float]
        Transformed values.
    """
    lin_half = linear_frac / 2.0
    log_scale = (1.0 - linear_frac) / (2.0 * max(decades, 1))

    arr = np.asarray(values, dtype=float)
    out = np.where(
        np.abs(arr) <= linthresh,
        lin_half * arr / linthresh,
        np.sign(arr) * (lin_half + log_scale * np.log10(np.abs(arr) / linthresh)),
    )
    return out.tolist()


def _symlog_ticks(
    linthresh: float = LINTHRESH_ENERGY,
    decades: int = SYMLOG_DECADES,
    linear_frac: float = LINEAR_FRAC,
) -> tuple[list[float], list[str]]:
    """
    Generate tick positions and labels for a symlog axis.

    Parameters
    ----------
    linthresh
        Linear threshold matching ``_symlog``.
    decades
        Number of decades to show on each side of zero.
    linear_frac
        Must match the value passed to ``_symlog``.

    Returns
    -------
    tuple[list[float], list[str]]
        Tick values (in transformed space) and their display labels.
    """
    ticks_val: list[float] = []
    ticks_txt: list[str] = []

    kw = {"linthresh": linthresh, "linear_frac": linear_frac, "decades": decades}

    # Negative decades
    for d in range(decades, 0, -1):
        raw = -(linthresh * 10**d)
        ticks_val.append(_symlog([raw], **kw)[0])
        ticks_txt.append(f"{raw:.0e}")
    # Linear region
    for v in [-linthresh, -linthresh / 2, 0, linthresh / 2, linthresh]:
        ticks_val.append(_symlog([v], **kw)[0])
        ticks_txt.append(f"{v:g}")
    # Positive decades
    for d in range(1, decades + 1):
        raw = linthresh * 10**d
        ticks_val.append(_symlog([raw], **kw)[0])
        ticks_txt.append(f"{raw:.0e}")

    return ticks_val, ticks_txt


METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)


def load_model_data(model_name: str) -> pd.DataFrame:
    """
    Load compression curve data for a model.

    Parameters
    ----------
    model_name
        Model identifier.

    Returns
    -------
    pd.DataFrame
        Dataframe containing structure, scale, volume, energy, and pressure columns.
    """
    csv_path = CALC_PATH / model_name / "compression.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def prepare_structure_series(
    struct_dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort and align energy/pressure series for a crystal structure.

    Parameters
    ----------
    struct_dataframe
        Structure-specific dataframe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Volume per atom, shifted energy per atom, pressure, and scale factors
        sorted by increasing volume.
    """
    df_sorted = struct_dataframe.sort_values("volume_per_atom").drop_duplicates(
        "volume_per_atom"
    )
    if len(df_sorted) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    volumes = df_sorted["volume_per_atom"].to_numpy()
    energies = df_sorted["energy_per_atom"].to_numpy()
    pressures = df_sorted["pressure"].to_numpy()
    scales = df_sorted["scale"].to_numpy()

    # Shift energies so the equilibrium value (scale closest to 1.0) is zero
    # eq_idx = int(np.argmin(np.abs(scales - 1.0)))
    # shifted_energies = energies - energies[eq_idx]
    shifted_energies = (
        energies - energies[-1]
    )  # Shift by the last energy value (largest volume)

    return volumes, shifted_energies, pressures, scales


def count_sign_changes(array: np.ndarray, tol: float) -> int:
    """
    Count sign changes in a sequence while ignoring small magnitudes.

    Parameters
    ----------
    array
        Input values.
    tol
        Absolute tolerance below which values are treated as zero.

    Returns
    -------
    int
        Number of sign changes exceeding the specified tolerance.
    """
    if array.size < 3:
        return 0
    clipped = array[np.abs(array) > tol]
    if clipped.size < 2:
        return 0
    signs = np.sign(clipped)
    sign_flips = signs[:-1] != signs[1:]
    return int(np.sum(sign_flips))


def compute_structure_metrics(
    df_struct: pd.DataFrame,
) -> dict[str, float] | None:
    """
    Compute diagnostics for a single crystal structure compression curve.

    Parameters
    ----------
    df_struct
        Structure-specific dataframe.

    Returns
    -------
    dict[str, float] | None
        Dictionary of metrics, or None if insufficient data.
    """
    volumes, shifted_energies, pressures, scales = prepare_structure_series(df_struct)
    if volumes.size < 3:
        return None

    # Energy minima: find peaks in inverted energy
    minima = 0
    if shifted_energies.size >= 3:
        minima_indices, _ = find_peaks(
            -shifted_energies,
            prominence=0.001,
            width=1,
        )
        minima = len(minima_indices)

        minima_indices, _ = find_peaks(
            -shifted_energies,
            prominence=0.1,
            width=1,
        )
        deep_minima = len(minima_indices)

    # Is there a hole present or not
    # Relative to the largest volume energy
    relative_energies = shifted_energies - shifted_energies[-1]
    holes = 0
    # If any energy is 100 eV/atom lower than the largest volume
    # energy, we safely consider it a hole
    if np.any(relative_energies < -100):
        holes = 1

    # Pressure sign flips
    pressure_flips = count_sign_changes(pressures, tol=1e-3)
    big_pressure_flips = count_sign_changes(pressures, tol=1)

    # Spearman correlations in compressed and expanded regimes
    spearman_compression = np.nan
    spearman_expansion = np.nan

    try:
        from scipy import stats

        # NOTE: this isn't perfect, minimum will be in the
        # wrong place if holes are present.
        # BUT for models without holes, this is another way
        # of finding spurious minima.
        eq_idx = np.argmin(shifted_energies)

        # Compressed regime
        if volumes[:eq_idx].size > 2:
            spearman_compression = float(
                stats.spearmanr(
                    volumes[: eq_idx + 1], shifted_energies[: eq_idx + 1]
                ).statistic
            )
        # Expanded regime
        if volumes[eq_idx:].size > 2:
            spearman_expansion = float(
                stats.spearmanr(volumes[eq_idx:], shifted_energies[eq_idx:]).statistic
            )
    except Exception:
        pass

    return {
        "Holes": float(holes),
        "Energy minima": float(minima),
        "Deep Energy minima": float(deep_minima),
        # "Energy inflections": float(inflections),
        "Big Pressure sign flips": float(big_pressure_flips),
        "Pressure sign flips": float(pressure_flips),
        "ρ(-E,Vsmall)": -float(spearman_compression),
        "ρ(E,Vlarge)": float(spearman_expansion),
    }


def aggregate_model_metrics(
    model_dataframe: pd.DataFrame,
) -> dict[str, float]:
    """
    Aggregate metrics across all crystal structures for a model.

    Parameters
    ----------
    model_dataframe
        Per-model compression dataset.

    Returns
    -------
    dict[str, float]
        Aggregated model metrics (averaged across all structures).
    """
    if model_dataframe.empty:
        return {}

    structure_metrics: list[dict[str, float]] = []

    for _struct, struct_dataframe in model_dataframe.groupby("structure"):
        metrics = compute_structure_metrics(struct_dataframe)
        if metrics is None:
            continue
        structure_metrics.append(metrics)

    if not structure_metrics:
        return {}

    return {
        key: float(np.nanmean([m.get(key, np.nan) for m in structure_metrics]))
        for key in DEFAULT_THRESHOLDS.keys()
    }


# Helper to load per-model data -----------------------------------------------


def _load_structure_data() -> dict[str, pd.DataFrame]:
    """
    Load per-model compression curve dataframes from calculator outputs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of model name to curve dataframe.
    """
    structure_data: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        model_dataframe = load_model_data(model_name)
        if not model_dataframe.empty:
            structure_data[model_name] = model_dataframe
    return structure_data


def _write_curve_payloads(
    curve_dir: Path, model_name: str, frame: pd.DataFrame
) -> None:
    """
    Serialise per-structure JSON payloads for the Dash app callbacks.

    Parameters
    ----------
    curve_dir
        Base directory for curve JSON files.
    model_name
        Name of the model.
    frame
        Dataframe with all structure data for this model.
    """
    model_curve_dir = curve_dir / model_name
    model_curve_dir.mkdir(parents=True, exist_ok=True)

    for struct_label, group in frame.groupby("structure"):
        ordered = group.sort_values("volume_per_atom").drop_duplicates(
            "volume_per_atom"
        )
        volumes = ordered["volume_per_atom"].tolist()
        energies = ordered["energy_per_atom"].tolist()
        pressures = ordered["pressure"].tolist()
        scales = ordered["scale"].tolist()

        # Compute dE/dV (numerical derivative of energy per atom w.r.t. volume)
        vol_arr = np.array(volumes)
        eng_arr = np.array(energies)
        if vol_arr.size >= 2:
            de_dv = np.gradient(eng_arr, vol_arr).tolist()
        else:
            de_dv = [0.0] * len(volumes)

        payload = {
            "structure": str(struct_label),
            "volume_per_atom": volumes,
            "energy_per_atom": energies,
            "pressure": pressures,
            "scale": scales,
            "dEdV": de_dv,
        }
        filepath = model_curve_dir / f"{struct_label}.json"
        with filepath.open("w", encoding="utf8") as fh:
            json.dump(payload, fh)


def _chemical_formula_from_label(label: str) -> str:
    """
    Convert a structure label to its reduced chemical formula.

    Parameters
    ----------
    label
        Structure label such as ``"C2H4_pyxtal_0"`` or ``"C_bcc"``.

    Returns
    -------
    str
        Reduced formula, e.g. ``"CH2"`` or ``"C"``.
    """
    formula_str = label.split("_")[0]
    f = Formula(formula_str)
    return str(f.reduce()[0])


def _build_formula_figures(model_name: str, frame: pd.DataFrame) -> None:
    """
    Build and save an energy-vs-scale scatter figure for each formula.

    All structures that share the same reduced chemical formula are overlaid
    on a single figure.  Figures are written to
    ``FIGURE_PATH / model_name / {formula}.json``.

    Parameters
    ----------
    model_name
        Model identifier.
    frame
        Dataframe with columns *structure*, *scale*, *energy_per_atom*
        (at minimum) for this model.
    """
    model_fig_dir = FIGURE_PATH / model_name
    model_fig_dir.mkdir(parents=True, exist_ok=True)

    # Group structure labels by reduced formula
    formula_groups: dict[str, list[str]] = defaultdict(list)
    for struct_label in frame["structure"].unique():
        formula = _chemical_formula_from_label(str(struct_label))
        formula_groups[formula].append(str(struct_label))

    # Pre-compute symlog ticks
    tick_vals, tick_text = _symlog_ticks(
        LINTHRESH_ENERGY,
        decades=SYMLOG_DECADES,
        linear_frac=LINEAR_FRAC,
    )

    for formula, struct_labels in formula_groups.items():
        fig = go.Figure()
        y_plot_range = [0, 0]
        for idx, struct_label in enumerate(sorted(struct_labels)):
            group = (
                frame[frame["structure"] == struct_label]
                .sort_values("volume_per_atom")
                .drop_duplicates("volume_per_atom")
            )
            scales = group["scale"].tolist()
            energies = group["energy_per_atom"].tolist()
            if not scales or not energies:
                continue

            color = _PALETTE[idx % len(_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=_symlog(energies),
                    mode="lines",
                    name=struct_label,
                    line={"color": color},
                    marker={"size": 5, "color": color},
                    customdata=energies,
                    hovertemplate=(
                        "scale: %{x:.3f}<br>E/atom: %{customdata:.4f} eV<extra></extra>"
                    ),
                )
            )
            y_plot_range[0] = min(y_plot_range[0], min(_symlog(energies)))
            y_plot_range[1] = max(y_plot_range[1], max(_symlog(energies)))

        default_range = [
            max(tick_vals[0], y_plot_range[0]),
            min(tick_vals[-1], y_plot_range[1]),
        ]

        default_range[0] -= abs(default_range[0]) * 0.1
        default_range[1] += abs(default_range[1]) * 0.1

        fig.update_layout(
            title=f"{model_name} — {formula}",
            xaxis_title="Scale factor",
            yaxis_title="Energy per atom (eV, symlog)",
            yaxis={
                "tickvals": tick_vals,
                "ticktext": tick_text,
                "range": default_range,
            },
            height=500,
            showlegend=True,
            template="plotly_white",
        )

        fig.write_json(model_fig_dir / f"{formula}.json")


def persist_compression_data() -> dict[str, pd.DataFrame]:
    """
    Persist curve payloads, build figures, and return per-model dataframes.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of model name to per-structure curve data.
    """
    data = _load_structure_data()
    CURVE_PATH.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    for model_name, frame in data.items():
        if frame is not None and not frame.empty:
            _write_curve_payloads(CURVE_PATH, model_name, frame)
            _build_formula_figures(model_name, frame)
    return data


@pytest.fixture
def compression_data_fixture() -> dict[str, pd.DataFrame]:
    """
    Load curve data and persist gallery assets for pytest use.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of model name to per-structure curve data.
    """
    return persist_compression_data()


def collect_metrics(
    structure_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Gather metrics for all models.

    Metrics are averaged across all crystal structures.

    Parameters
    ----------
    structure_data
        Optional mapping of model names to curve dataframes. When ``None``,
        the data is loaded via ``persist_compression_data``.

    Returns
    -------
    pd.DataFrame
        Aggregated metrics table (all structures).
    """
    metrics_rows: list[dict[str, float | str]] = []

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    data = structure_data if structure_data is not None else persist_compression_data()

    for model_name, model_dataframe in data.items():
        metrics = aggregate_model_metrics(model_dataframe)
        row = {"Model": model_name} | metrics
        metrics_rows.append(row)

    columns = ["Model"] + list(DEFAULT_THRESHOLDS.keys())

    return pd.DataFrame(metrics_rows).reindex(columns=columns)


@pytest.fixture
def compression_collection(
    compression_data_fixture: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Collect compression metrics across all models.

    Parameters
    ----------
    compression_data_fixture
        Mapping of model names to curve dataframes generated by the fixture.

    Returns
    -------
    pd.DataFrame
        Aggregated metrics dataframe.
    """
    return collect_metrics(compression_data_fixture)


@pytest.fixture
def compression_metrics_dataframe(
    compression_collection: pd.DataFrame,
) -> pd.DataFrame:
    """
    Provide the aggregated compression metrics dataframe.

    Parameters
    ----------
    compression_collection
        Metrics dataframe produced by ``collect_metrics``.

    Returns
    -------
    pd.DataFrame
        Aggregated compression metrics indexed by model.
    """
    return compression_collection


@pytest.fixture
@build_table(
    filename=OUT_PATH / "compression_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    compression_metrics_dataframe: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute compression metrics for all models.

    Parameters
    ----------
    compression_metrics_dataframe
        Aggregated per-model metrics produced by ``collect_metrics``.

    Returns
    -------
    dict[str, dict]
        Mapping of metric names to per-model results.
    """
    metrics_df = compression_metrics_dataframe
    metrics_dict: dict[str, dict[str, float | None]] = {}
    for column in metrics_df.columns:
        if column == "Model":
            continue
        values = [
            value if pd.notna(value) else None for value in metrics_df[column].tolist()
        ]
        metrics_dict[column] = dict(zip(metrics_df["Model"], values, strict=False))
    return metrics_dict


def test_compression(metrics: dict[str, dict]) -> None:
    """
    Run compression analysis.

    Parameters
    ----------
    metrics
        Benchmark metrics generated by fixtures.
    """
    return
