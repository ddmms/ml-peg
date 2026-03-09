"""Utility functions for analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ase.io import read, write
from matplotlib import cm
from matplotlib.colors import Colormap
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from yaml import safe_load

from ml_peg.app.utils.utils import (
    ThresholdEntry,
    Thresholds,
    clean_thresholds,
    clean_weights,
)
from ml_peg.models.get_models import load_model_configs

MetricRow = dict[str, float | int | str | None]
TableRow = dict[str, object]


def build_dispersion_name_map(
    models: Iterable[str],
    suffix: str = "-D3",
) -> dict[str, str]:
    """
    Return a suffix map for models requiring runtime dispersion corrections.

    Parameters
    ----------
    models
        Iterable of model identifiers to inspect.
    suffix
        String appended to model names that need the dispersion correction indicator.
        Defaults to "-D3" for D3 dispersion corrections.

    Returns
    -------
    dict[str, str]
        Mapping of model -> display name for models not trained with dispersion.
    """
    configs, _ = load_model_configs(tuple(models))
    name_map: dict[str, str] = {}

    for model in models:
        if configs[model]["trained_on_dispersion"]:
            continue
        dispersion_kwargs = configs[model].get("dispersion_kwargs") or {}
        label = dispersion_kwargs.get("label")
        suffix_to_use = (
            label.strip() if isinstance(label, str) and label.strip() else suffix
        )
        if not suffix_to_use.startswith("-"):
            suffix_to_use = f"-{suffix_to_use}"
        name_map[model] = f"{model}{suffix_to_use}"

    return name_map


def load_metrics_config(config_path: Path) -> tuple[Thresholds, dict[str, str]]:
    """
    Load metric thresholds and tooltips from a YAML configuration file.

    Parameters
    ----------
    config_path
        Path to the YAML file containing metric definitions.

    Returns
    -------
    tuple[Thresholds, dict[str, str]]
        Mapping of metric thresholds, tooltips, and weights.
    """
    if not config_path.exists():
        msg = f"Metrics configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf8") as stream:
        data = safe_load(stream)

    metrics_data = data.get("metrics", {})
    thresholds: Thresholds = {}
    tooltips: dict[str, str] = {}
    weights: dict[str, float] = {}

    for metric_name, metric_config in metrics_data.items():
        good_value = float(metric_config["good"])
        bad_value = float(metric_config["bad"])

        if "unit" not in metric_config:
            raise KeyError(
                f"Metric '{metric_name}' must define a 'unit' entry in {config_path}"
            )

        if metric_config["unit"] is not None:
            unit_value = str(metric_config["unit"]).strip()
        else:
            unit_value = "-"
        level_of_theory = metric_config.get("level_of_theory")

        metric_threshold: ThresholdEntry = {
            "good": good_value,
            "bad": bad_value,
            "unit": unit_value,
            "level_of_theory": level_of_theory,
        }

        thresholds[metric_name] = metric_threshold

        tooltip = metric_config.get("tooltip")

        if tooltip is None or tooltip == "":
            msg = (
                f"Metric '{metric_name}' in '{config_path}' must define a non-empty "
                "'tooltip' entry."
            )
            raise ValueError(msg)

        tooltip_text = tooltip.strip()
        tooltip_lines = [tooltip_text]
        if unit_value and unit_value not in ("", "-"):
            tooltip_lines[0] = f"{tooltip_lines[0]} [{unit_value}]"
        if level_of_theory:
            tooltip_lines.append(f"Level of theory: {level_of_theory}")
        markdown_value = "\n".join(tooltip_lines).replace("\n", "  \n")
        tooltips[metric_name] = {
            "value": markdown_value,
            "type": "markdown",
        }

        weights[metric_name] = float(metric_config.get("weight", 1.0))

    return thresholds, tooltips, weights


def mae(ref: list, prediction: list) -> float:
    """
    Get mean absolute error.

    Parameters
    ----------
    ref
        Reference data.
    prediction
        Predicted data.

    Returns
    -------
    float
        Mean absolute error.
    """
    return mean_absolute_error(ref, prediction)


def rmse(ref: list, prediction: list) -> float:
    """
    Get root mean squared error.

    Parameters
    ----------
    ref
        Reference data.
    prediction
        Predicted data.

    Returns
    -------
    float
        Root mean squared error.
    """
    return mean_squared_error(ref, prediction)


DENSITY_GRID_SIZE = 80
DENSITY_MAX_POINTS_PER_CELL = 5
DENSITY_SAMPLE_SEED = 0


def sample_density_grid(
    ref_vals: list[float] | np.ndarray,
    pred_vals: list[float] | np.ndarray,
    *,
    grid_size: int = DENSITY_GRID_SIZE,
    max_points_per_cell: int = DENSITY_MAX_POINTS_PER_CELL,
    seed: int = DENSITY_SAMPLE_SEED,
) -> tuple[list[int], list[int], list[list[int]]]:
    """
    Sample indices from a density grid, returning density and cell memberships.

    This is the shared implementation used by ``plot_density_scatter`` so the
    sampling order stays consistent across density-driven artifacts.

    Parameters
    ----------
    ref_vals
        Reference (x-axis) values for all systems, in the same order as the
        data passed to the density scatter fixture.
    pred_vals
        Predicted (y-axis) values, same order as ``ref_vals``.
    grid_size
        Number of bins per axis. Must match ``@plot_density_scatter`` default.
    max_points_per_cell
        Maximum sampled points per cell. Must match decorator default.
    seed
        RNG seed for deterministic sampling. Must match decorator default.

    Returns
    -------
    tuple[list[int], list[int], list[list[int]]]
        ``sampled_indices``: original indices kept in plotting order.
        ``sampled_density``: density value per sampled index.
        ``sampled_mapping``: list of source indices for each sampled point.
    """
    ref_arr = np.asarray(ref_vals, dtype=float)
    pred_arr = np.asarray(pred_vals, dtype=float)

    if ref_arr.size == 0 or pred_arr.size == 0:
        return [], [], []

    delta_x = ref_arr.max() - ref_arr.min()
    delta_y = pred_arr.max() - pred_arr.min()
    eps = 1e-9

    norm_x = np.clip((ref_arr - ref_arr.min()) / max(delta_x, eps), 0.0, 0.999999)
    norm_y = np.clip((pred_arr - pred_arr.min()) / max(delta_y, eps), 0.0, 0.999999)
    bins_x = (norm_x * grid_size).astype(int)
    bins_y = (norm_y * grid_size).astype(int)

    cell_points: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (cx, cy) in enumerate(zip(bins_x, bins_y, strict=True)):
        cell_points[(int(cx), int(cy))].append(idx)

    rng = np.random.default_rng(seed)
    sampled_indices: list[int] = []
    sampled_density: list[int] = []
    sampled_mapping: list[list[int]] = []
    for indices in cell_points.values():
        if len(indices) > max_points_per_cell:
            chosen = rng.choice(indices, size=max_points_per_cell, replace=False)
        else:
            chosen = indices
        density = len(indices)
        for idx in chosen:
            sampled_indices.append(int(idx))
            sampled_density.append(density)
            sampled_mapping.append(indices)

    return sampled_indices, sampled_density, sampled_mapping


def build_density_inputs(
    models: list[str],
    model_results: dict[str, dict[str, Any]],
    property_key: str,
    metric_fn: Callable[[list, list], float],
) -> dict[str, dict[str, Any]]:
    """
    Prepare a model->data mapping for density scatter plots.

    Parameters
    ----------
    models
        Ordered list of model names to include.
    model_results
        Mapping of model -> {"<property_key>": {"ref": [...], "pred": [...]},
        "excluded": int}. These per-model property arrays come from the analysis step
        (e.g. filtered bulk/shear values and metadata).
    property_key
        Key to extract from ``model_results`` for each model (e.g. ``"bulk"`` or
        ``"shear"``).
    metric_fn
        Function that turns the ``ref`` and ``pred`` lists into a single value (for
        example, MAE). This number is stored in the result so the plotting code can show
        it in hover text/annotations.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping ready for ``plot_density_scatter``.
    """
    inputs: dict[str, dict[str, Any]] = {}

    for model_name in models:
        stats = model_results.get(model_name, {})
        prop = stats.get(property_key)
        excluded = stats.get("excluded")

        ref_vals = prop.get("ref", [])
        pred_vals = prop.get("pred", [])
        inputs[model_name] = {
            "ref": ref_vals,
            "pred": pred_vals,
            "metric": metric_fn(ref_vals, pred_vals) if ref_vals else None,
            "meta": {"excluded": excluded} if excluded is not None else {},
        }

    return inputs


def write_density_trajectories(
    *,
    labels_list: list[str],
    ref_vals: list[float],
    pred_vals: list[float],
    struct_dir: Path,
    traj_dir: Path,
    struct_filename_builder: Callable[[str], str],
    grid_size: int = DENSITY_GRID_SIZE,
    max_points_per_cell: int = DENSITY_MAX_POINTS_PER_CELL,
    seed: int = DENSITY_SAMPLE_SEED,
) -> None:
    """
    Write one extxyz trajectory per sampled density point for WEAS display.

    Parameters
    ----------
    labels_list
        Ordered system labels matching ``ref_vals``/``pred_vals``.
    ref_vals
        Reference values passed to the density scatter.
    pred_vals
        Predicted values passed to the density scatter.
    struct_dir
        Directory containing per-system structure files.
    traj_dir
        Output directory for sampled density trajectories.
    struct_filename_builder
        Function to convert each label into the source structure filename.
    grid_size
        Number of bins per axis. Must match ``@plot_density_scatter``.
    max_points_per_cell
        Maximum sampled points per occupied density cell.
    seed
        RNG seed for deterministic sampling.
    """
    _, _, sampled_mapping = sample_density_grid(
        ref_vals,
        pred_vals,
        grid_size=grid_size,
        max_points_per_cell=max_points_per_cell,
        seed=seed,
    )

    traj_dir.mkdir(parents=True, exist_ok=True)

    for point_idx, source_indices in enumerate(sampled_mapping):
        frames = []
        for source_idx in source_indices:
            label = labels_list[source_idx]
            struct_path = struct_dir / struct_filename_builder(label)
            frames.append(read(struct_path))
        write(traj_dir / f"{point_idx}.extxyz", frames)


def calc_metric_scores(
    metrics_data: list[MetricRow],
    thresholds: Thresholds | None = None,
    normalizer: Callable[[float, float, float], float] | None = None,
) -> list[MetricRow]:
    """
    Calculate all normalised scores.

    Parameters
    ----------
    metrics_data
        Rows data containing model name and metric values.
    thresholds
        Normalisation thresholds keyed by metric name. Each value must be a mapping
        containing ``good``, ``bad``, and ``unit`` entries. Default is `None`.
    normalizer
        Optional function to map (value, good, bad) -> normalised score.
        If `None`, and thresholds are specified, uses `normalize_metric`.

    Returns
    -------
    list[MetricRow]
        Rows data with metric scores in place of values.
    """
    normalizer = normalizer if normalizer is not None else normalize_metric
    cleaned_thresholds = clean_thresholds(thresholds) if thresholds else None

    metrics_scores = [row.copy() for row in metrics_data]
    for row in metrics_scores:
        for key, value in row.items():
            # Value may be ``None`` if missing for a benchmark
            if key not in {"MLIP", "Score", "id"} and value is not None:
                if cleaned_thresholds is None or key not in cleaned_thresholds:
                    row[key] = value
                    continue

                entry = cleaned_thresholds[key]
                row[key] = normalizer(value, entry["good"], entry["bad"])

    return metrics_scores


def calc_table_scores(
    metrics_data: list[MetricRow],
    weights: dict[str, float] | None = None,
    thresholds: Thresholds | None = None,
    normalizer: Callable[[float, float, float], float] | None = None,
    require_all_metrics: bool = True,
) -> list[MetricRow]:
    """
    Calculate (normalised) score for each model and add to table data.

    If `thresholds` is not `None`, `normalizer` will be used to normalise the score.

    Parameters
    ----------
    metrics_data
        Rows data containing model name and metric values.
    weights
        Weight for each metric. Default is 1.0 for each metric.
    thresholds
        Normalisation thresholds keyed by metric name. Each value must be a mapping
        with ``good``, ``bad``, and ``unit`` entries. Defauls is `None`.
    normalizer
        Optional function to map (value, good, bad) -> normalised score.
        If `None`, and thresholds are specified, uses `normalize_metric`.
    require_all_metrics
        If True, score is set to None unless all metrics are present (not None).
        If False, score is calculated from available metrics only.
        Default is True.

    Returns
    -------
    list[MetricRow]
        Rows of data with combined score for each model added.
    """
    weights = weights if weights else {}

    metrics_scores = calc_metric_scores(metrics_data, thresholds, normalizer)

    for metrics_row, scores_row in zip(metrics_data, metrics_scores, strict=True):
        scores_list = []
        weights_list = []
        all_metrics_present = True

        for key, value in metrics_row.items():
            if key in {"MLIP", "Score", "id"}:
                continue

            weight = weights.get(key, 1.0)
            if weight == 0:
                # Weight of zero excludes the metric from scoring requirements
                continue

            if value is not None:
                scores_list.append(scores_row[key])
                weights_list.append(weight)
            else:
                # Track if any (weighted) metric is missing
                all_metrics_present = False

        # Calculate score only if conditions are met
        if require_all_metrics and not all_metrics_present:
            # Strict mode: require all metrics to be present
            metrics_row["Score"] = None
        elif scores_list:
            # Calculate weighted average of available metrics
            try:
                metrics_row["Score"] = np.average(scores_list, weights=weights_list)
            except ZeroDivisionError:
                metrics_row["Score"] = np.mean(scores_list)
        else:
            metrics_row["Score"] = None

    return metrics_data


def get_table_style(
    data: list[TableRow],
    *,
    scored_data: list[dict] | None = None,
    normalized: bool = True,
    all_cols: bool = True,
    col_names: list[str] | str | None = None,
) -> list[TableRow]:
    """
    Viridis-style colormap for Dash DataTable.

    Parameters
    ----------
    data
        Data from Dash table to be coloured.
    scored_data
        Data with metric values replaced with scores.
    normalized
        Whether metric/score columns have been normalized to between 0 and 1. Default is
        `True`.
    all_cols
        Whether to colour all numerical columns.
    col_names
        Column name or list of names to be coloured.

    Returns
    -------
    list[TableRow]
        Conditional style data to apply to table.
    """
    cmap = cm.get_cmap("viridis_r")

    def rgba_from_val(val: float, vmin: float, vmax: float, cmap: Colormap) -> str:
        """
        Get RGB values for a cell.

        Parameters
        ----------
        val
            Value to colour.
        vmin
            Minimum value in column.
        vmax
            Maximum value in column.
        cmap
            Colour map for cell colours.

        Returns
        -------
        str
            RGB colours in backgroundColor format.
        """
        norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0
        rgba = cmap(norm)
        r, g, b = [int(255 * x) for x in rgba[:3]]
        return f"rgb({r}, {g}, {b})"

    style_data_conditional: list[TableRow] = []

    # All columns other than MLIP and ID (not displayed) should be coloured
    if all_cols:
        cols = data[0].keys() - {"MLIP", "id"}
    elif col_names:
        if isinstance(col_names, str):
            cols = [col_names]
    else:
        raise ValueError("Specify either all_cols=True or provide col_name.")

    for col in cols:
        if col not in data[0]:
            raise ValueError(f"Column '{col}' not found in data.")

    for col in cols:
        numeric_entries: list[tuple[object, float, float]] = []
        none_row_indices: list[int] = []  # Track rows with None values
        for i, row in enumerate(data):
            if col not in row:
                continue
            raw_value = row[col]
            # Track None values separately for styling
            is_none = raw_value is None
            is_nan = isinstance(raw_value, float) and np.isnan(raw_value)
            if is_none or is_nan:
                none_row_indices.append(i)
                continue
            # Skip if unable to convert to float
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                none_row_indices.append(i)
                continue

            # Get scored value, if is exists
            try:
                scored_value = float(scored_data[i][col])
            except (TypeError, ValueError, IndexError):
                scored_value = raw_value

            numeric_entries.append((raw_value, numeric_value, scored_value))

        # Apply styling for None/missing values: gray hashed pattern
        for row_idx in none_row_indices:
            mlip_name = data[row_idx].get("MLIP", "")
            style_data_conditional.append(
                {
                    "if": {
                        "filter_query": f"{{MLIP}} = '{mlip_name}'",
                        "column_id": col,
                    },
                    "backgroundColor": "#e0e0e0",
                    "backgroundImage": (
                        "repeating-linear-gradient("
                        "45deg, "
                        "transparent, "
                        "transparent 5px, "
                        "#d0d0d0 5px, "
                        "#d0d0d0 10px"
                        ")"
                    ),
                    "color": "#888888",
                    "fontStyle": "italic",
                }
            )

        if not numeric_entries:
            continue

        numeric_values = [numeric for _, numeric, _ in numeric_entries]

        # Use thresholds
        if normalized:
            min_value, max_value = 1, 0
        else:
            min_value = min(numeric_values)
            max_value = max(numeric_values)

        for raw_value, _, scored_value in numeric_entries:
            # Determine direction of values
            mid = (min_value + max_value) / 2
            increasing = max_value >= min_value

            style_data_conditional.append(
                {
                    "if": {
                        "filter_query": f"{{{col}}} = {raw_value}",
                        "column_id": col,
                    },
                    "backgroundColor": rgba_from_val(
                        scored_value, min_value, max_value, cmap
                    ),
                    "color": "white"
                    if (scored_value > mid if increasing else scored_value < mid)
                    else "black",
                }
            )

    return style_data_conditional


def update_score_style(
    data: list[MetricRow],
    weights: dict[str, float] | None = None,
    thresholds: Thresholds | None = None,
) -> tuple[list[MetricRow], list[TableRow]]:
    """
    Update table scores and table styles.

    Parameters
    ----------
    data
        Rows data containing model name and metric values.
    weights
        Weight for each metric. Default is `None`.
    thresholds
        Normalisation thresholds keyed by metric name. Each value must be a mapping
        with ``good``, ``bad``, and ``unit`` entries. Default is `None`.

    Returns
    -------
    tuple[list[MetricRow], list[TableRow]]
        Updated table rows and style data.
    """
    weights = clean_weights(weights)
    data = calc_table_scores(data, weights, thresholds)
    scored_data = calc_metric_scores(data, thresholds)
    style = get_table_style(data, scored_data=scored_data)
    return data, style


def normalize_metric(
    value: float, good_threshold: float, bad_threshold: float
) -> float | None:
    """
    Normalize a metric value to [0.0, 1.0].

    `good_threshold` is mapped to 1.0 and `bad_threshold` mapped to 0.0. Values beyond
    these thresholds are clipped to 0.0 and 1.0.

    Works regardless of whether `good_threshold` > `bad_threshold` or
    `good_threshold` < `bad_threshold`.

    Parameters
    ----------
    value
        The metric value to normalise.
    good_threshold
        Threshold that maps to score 1.0.
    bad_threshold
        Threshold that maps to score 0.0.

    Returns
    -------
    float | None
        Normalized score between 0 and 1, or `None` if normalisation process
        raises an error.
    """
    if value is None or good_threshold is None or bad_threshold is None:
        return None

    try:
        # Handle NaNs robustly
        if np.isnan([value, good_threshold, bad_threshold]).any():
            return None
    except TypeError:
        return None

    if good_threshold == bad_threshold:
        return 1.0 if value == good_threshold else 0.0

    # Linear map: Y -> 0, X -> 1
    t = (value - bad_threshold) / (good_threshold - bad_threshold)

    # Clip to [0, 1]
    return max(min(1.0, float(t)), 0.0)
