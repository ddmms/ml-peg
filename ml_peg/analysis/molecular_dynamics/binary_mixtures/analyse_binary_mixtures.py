"""
Analyse the binary mixtures benchmark.

Scores ΔHmix, mixture density, excess molar volume (V^E) and pure-liquid
densities against experiment. The densities are shown at weight 0, as the
liquid densities benchmark already scores most of the pure liquids and mixture
density errors are mostly pure-density errors. Box means and their
uncertainties come from the autocorrelation estimator in
``ml_peg.analysis.utils.timeseries`` (see ``mixing.py``). Metric confidence
intervals are bootstrapped at the level of the 50 boxes, so the correlation
between compositions that share pure boxes is included. Every property is
scored by MAE and by MAZE, which divides each error by a fixed,
model-independent σ_ref from ``systems.csv`` or ``pure.csv`` (experimental
uncertainty and Sage replicate spread).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
import json
from pathlib import Path

from ase.io import Trajectory, read, write
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.molecular_dynamics.binary_mixtures.mixing import (
    MIN_N_EFF,
    BoxResult,
    box_result,
    combine,
    frame_checks,
    holm,
    mixing_matrix,
    mixture_box_id,
    observables,
    read_production,
)
from ml_peg.analysis.utils.decorators import (
    build_table,
    merge_saved_models,
    plot_parity,
    plot_scatter,
)
from ml_peg.analysis.utils.timeseries import (
    MetricEstimate,
    TimeSeriesEstimate,
    bootstrap_metric,
)
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.molecular_dynamics.binary_mixtures.calc_binary_mixtures import (
    DATA_KEY,
    DATA_ZIP,
    EQUILIBRATION_PS,
    PRODUCTION_PS,
    PS,
    TIMESTEP,
    TRAJ_INTERVAL,
)
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

CATEGORY = "molecular_dynamics"
BENCHMARK = "binary_mixtures"
CALC_PATH = CALCS_ROOT / CATEGORY / BENCHMARK / "outputs"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_dispersion_name_map()

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

TOTAL_PS = EQUILIBRATION_PS + PRODUCTION_PS
FRAME_TIMES_PS = np.arange(EQUILIBRATION_PS, TOTAL_PS + 1, 100.0)
TRAJ_EVERY_PS = TRAJ_INTERVAL * TIMESTEP / PS
FLAG_P = 0.05  # family-wise, Holm-adjusted, as for the Sage reference runs
DEMIX_SD = 5.0  # like-fraction above Sage by this many Sage SDs...
DEMIX_MIN = 0.05  # ...and by at least this much
CONVERGENCE_BLOCK_PS = 100.0
RUNNING_MEAN_SKIP_PS = 50.0
SAGE_LABEL = "Sage 2.3.0 (classical)"


@cache
def load_data() -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download the benchmark data and read its tables.

    Returns
    -------
    tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Data directory, ``systems.csv``, ``boxes.csv`` and ``pure.csv``.
    """
    data_path = download_s3_data(filename=DATA_ZIP, key=DATA_KEY) / "binary_mixtures"
    return (
        data_path,
        pd.read_csv(data_path / "systems.csv"),
        pd.read_csv(data_path / "boxes.csv"),
        pd.read_csv(data_path / "pure.csv"),
    )


DATA_PATH, SYSTEMS, BOXES, PURE = load_data()
BOX_IDS = sorted(BOXES.box_id)
N_MOLECULES = {
    box.box_id: sum(int(c) for c in str(box.counts).split(";"))
    for box in BOXES.itertuples()
}
POINT_IDS = [mixture_box_id(s.system_id, s.x_box) for s in SYSTEMS.itertuples()]
MIX = mixing_matrix(SYSTEMS, BOXES, BOX_IDS)
# Points with an experimental mixture density (and V^E)
DENSITY_ROWS = np.flatnonzero(SYSTEMS.density_exp.notna().to_numpy())
VEX_ROWS = np.flatnonzero(SYSTEMS.vex_exp.notna().to_numpy())
VEX_SIGMA = SYSTEMS.vex_sigma.fillna(0.0).to_numpy()[VEX_ROWS]
DENSITY_BOXES = [POINT_IDS[i] for i in DENSITY_ROWS]
# Every pure liquid (at weight 0, as liquid_densities scores most of them)
_PURE_BOX = BOXES[BOXES.kind == "pure"].set_index(["smiles", "temperature"]).box_id
PURE_BOXES = [_PURE_BOX.loc[(p.smiles, p.temperature)] for p in PURE.itertuples()]

# Elements of the input boxes, for the app's element filter
write_struct_info(
    [DATA_PATH / "boxes" / f"{box_id}.xyz" for box_id in BOX_IDS], OUT_PATH
)


def model_box_result(model_name: str, box_id: str) -> BoxResult:
    """
    Calculate statistics of one model's run of one box.

    Parameters
    ----------
    model_name
        Model name.
    box_id
        Box identifier.

    Returns
    -------
    BoxResult
        Statistics of the run.

    Raises
    ------
    FileNotFoundError
        If the run's scalar log is missing.
    ValueError
        If the run did not reach ``TOTAL_PS``.
    """
    csv_path = CALC_PATH / model_name / f"{box_id}.csv"
    prod = read_production(csv_path, EQUILIBRATION_PS, TOTAL_PS)
    return box_result(prod, N_MOLECULES[box_id], label=f"{model_name}/{box_id}")


def model_frame_checks(model_name: str, box_id: str) -> tuple[bool, float]:
    """
    Reaction and demixing checks on frames every 100 ps of production.

    Parameters
    ----------
    model_name
        Model name.
    box_id
        Box identifier.

    Returns
    -------
    tuple[bool, float]
        Whether bonding changed and the mean like-neighbour fraction (NaN for
        pure boxes).
    """
    with Trajectory(CALC_PATH / model_name / f"{box_id}.traj") as traj:
        indices = [
            round(t / TRAJ_EVERY_PS)
            for t in FRAME_TIMES_PS
            if t / TRAJ_EVERY_PS < len(traj)
        ]
        frames = [traj[i] for i in indices]
    reference = read(DATA_PATH / "boxes" / f"{box_id}.xyz")
    mixture = bool(BOXES.set_index("box_id").loc[box_id, "kind"] == "mixture")
    return frame_checks(frames, reference, mixture)


@pytest.fixture(scope="module")
def box_results() -> dict[str, dict[str, BoxResult]]:
    """
    Get box statistics for all models.

    Returns
    -------
    dict[str, dict[str, BoxResult]]
        Statistics keyed by model and box.
    """
    return {
        model_name: {box_id: model_box_result(model_name, box_id) for box_id in BOX_IDS}
        for model_name in MODELS
    }


def box_estimates(
    results: dict[str, BoxResult], observable: str, box_ids: list[str]
) -> list[TimeSeriesEstimate]:
    """
    Collect estimates of one observable for a list of boxes.

    Parameters
    ----------
    results
        Box statistics of one model.
    observable
        ``h``, ``density`` or ``v_molar``.
    box_ids
        Boxes to return, in order.

    Returns
    -------
    list[TimeSeriesEstimate]
        Estimates, in the order of ``box_ids``.
    """
    return [results[box_id].estimates[observable] for box_id in box_ids]


@pytest.fixture(scope="module")
def flags(box_results) -> dict[str, dict[str, list[str]]]:
    """
    Flag convergence, reaction and demixing problems, and write them for the app.

    Parameters
    ----------
    box_results
        Box statistics for all models.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Flags keyed by model and box (boxes without flags are omitted).
    """
    sage = BOXES.set_index("box_id")
    all_flags: dict[str, dict[str, list[str]]] = {}
    diagnostics: dict[str, dict[str, dict]] = {}
    for model_name, results in box_results.items():
        model_flags: dict[str, list[str]] = {}
        diagnostics[model_name] = {}
        tests = []
        for box_id, result in results.items():
            box_flags = model_flags.setdefault(box_id, [])
            for obs in ("h", "density"):
                if result.estimates[obs].n_eff < MIN_N_EFF:
                    box_flags.append(f"low n_eff ({obs})")
                tests.append((box_id, f"drift ({obs})", result.p_drift[obs]))
            reacted, like = model_frame_checks(model_name, box_id)
            if reacted:
                box_flags.append("bonding changed")
            ref_like = sage.loc[box_id].get("like_fraction_sage", np.nan)
            ref_sd = sage.loc[box_id].get("like_fraction_sage_sd", np.nan)
            if np.isfinite(like) and np.isfinite(ref_like):
                if like - ref_like > max(DEMIX_SD * ref_sd, DEMIX_MIN):
                    box_flags.append("demixing")
            diagnostics[model_name][box_id] = {
                "n_eff_h": result.estimates["h"].n_eff,
                "n_eff_density": result.estimates["density"].n_eff,
                "like_fraction": like,
                "like_fraction_sage": ref_like,
            }
        if tests:
            adjusted = holm([p for *_, p in tests])
            for (box_id, name, _), p in zip(tests, adjusted, strict=True):
                if p < FLAG_P:
                    model_flags[box_id].append(f"{name}, p_adj = {p:.2g}")
        all_flags[model_name] = {box: f for box, f in model_flags.items() if f}

    # With --update, keep the saved flags of models not being analysed
    flags_path = OUT_PATH / "flags.json"
    saved = {
        "flags": merge_saved_models(all_flags, flags_path, key="flags"),
        "diagnostics": merge_saved_models(diagnostics, flags_path, key="diagnostics"),
    }
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with flags_path.open("w") as handle:
        json.dump(saved, handle, indent=1)
    return all_flags


def _per_model_metric(
    box_results: dict[str, dict[str, BoxResult]],
    observable: str,
    box_ids: list[str],
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    ref: np.ndarray,
    ref_sigma: np.ndarray,
) -> dict[str, MetricEstimate]:
    """
    Bootstrap a metric over box estimates for every model.

    Parameters
    ----------
    box_results
        Box statistics for all models.
    observable
        Observable passed to ``statistic_fn`` for each box in ``box_ids``.
    box_ids
        Boxes whose estimates are resampled.
    statistic_fn
        Function of (reference values, box means).
    ref
        Reference values.
    ref_sigma
        Standard uncertainties of the reference values.

    Returns
    -------
    dict[str, MetricEstimate]
        Metric for each model.
    """
    return {
        model_name: bootstrap_metric(
            statistic_fn,
            ref,
            box_estimates(results, observable, box_ids),
            ref_sigma_1sd=ref_sigma,
        )
        for model_name, results in box_results.items()
    }


def _mae_and_maze(
    box_results: dict[str, dict[str, BoxResult]],
    observable: str,
    box_ids: list[str],
    predict: Callable[[np.ndarray], np.ndarray],
    ref: np.ndarray,
    ref_sigma: np.ndarray,
    sigma_ref: np.ndarray,
) -> tuple[dict[str, MetricEstimate], dict[str, MetricEstimate]]:
    """
    Bootstrap the MAE and the MAZE (fixed σ_ref per point) for every model.

    Parameters
    ----------
    box_results
        Box statistics for all models.
    observable
        Observable whose box estimates are resampled.
    box_ids
        Boxes whose estimates are resampled.
    predict
        Maps the box means to predictions comparable with ``ref``.
    ref
        Reference values.
    ref_sigma
        Standard uncertainties of the reference values.
    sigma_ref
        Fixed scale for each point in the MAZE.

    Returns
    -------
    tuple[dict[str, MetricEstimate], dict[str, MetricEstimate]]
        MAE and MAZE for each model.
    """

    def statistic(scale):
        """
        Return a function of (reference values, box means) for the metric.

        Parameters
        ----------
        scale
            Scale for each point in the MAZE (1.0 for the MAE).

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Function of (reference values, box means).
        """
        return lambda ref, boxes: float(np.mean(np.abs(predict(boxes) - ref) / scale))

    mae, maze = (
        _per_model_metric(
            box_results, observable, box_ids, statistic(scale), ref, ref_sigma
        )
        for scale in (1.0, sigma_ref)
    )
    return mae, maze


@pytest.fixture(scope="module")
def hmix_errors(box_results) -> tuple[dict, dict]:
    """
    Get the ΔHmix MAE (kJ/mol) and MAZE.

    Parameters
    ----------
    box_results
        Box statistics for all models.

    Returns
    -------
    tuple[dict, dict]
        MAE and MAZE for each model.
    """
    return _mae_and_maze(
        box_results,
        "h",
        BOX_IDS,
        lambda h: MIX @ h,
        SYSTEMS.hmix_exp.to_numpy(),
        SYSTEMS.hmix_sigma.to_numpy(),
        SYSTEMS.sigma_ref_hmix.to_numpy(),
    )


@pytest.fixture(scope="module")
def density_errors(box_results) -> tuple[dict, dict]:
    """
    Get the mixture density MAE (g/cm³) and MAZE.

    Parameters
    ----------
    box_results
        Box statistics for all models.

    Returns
    -------
    tuple[dict, dict]
        MAE and MAZE for each model.
    """
    return _mae_and_maze(
        box_results,
        "density",
        DENSITY_BOXES,
        lambda rho: rho,
        SYSTEMS.density_exp.to_numpy()[DENSITY_ROWS],
        SYSTEMS.density_sigma.to_numpy()[DENSITY_ROWS],
        SYSTEMS.sigma_ref_density.to_numpy()[DENSITY_ROWS],
    )


@pytest.fixture(scope="module")
def vex_errors(box_results) -> tuple[dict, dict]:
    """
    Get the excess molar volume MAE (cm³/mol) and MAZE.

    Parameters
    ----------
    box_results
        Box statistics for all models.

    Returns
    -------
    tuple[dict, dict]
        MAE and MAZE for each model.
    """
    mix = MIX[VEX_ROWS]
    return _mae_and_maze(
        box_results,
        "v_molar",
        BOX_IDS,
        lambda v: mix @ v,
        SYSTEMS.vex_exp.to_numpy()[VEX_ROWS],
        VEX_SIGMA,
        SYSTEMS.sigma_ref_vex.to_numpy()[VEX_ROWS],
    )


@pytest.fixture(scope="module")
def pure_density_errors(box_results) -> tuple[dict, dict]:
    """
    Get the MAE (g/cm³) and MAZE of the pure densities.

    Parameters
    ----------
    box_results
        Box statistics for all models.

    Returns
    -------
    tuple[dict, dict]
        MAE and MAZE for each model.
    """
    return _mae_and_maze(
        box_results,
        "density",
        PURE_BOXES,
        lambda rho: rho,
        PURE.density.to_numpy(),
        PURE.sigma.to_numpy(),
        PURE.sigma_ref_density.to_numpy(),
    )


def point_estimates(
    results: dict[str, BoxResult], observable: str
) -> list[TimeSeriesEstimate]:
    """
    Per-point mixing estimates (ΔHmix or V^E) of one model.

    Parameters
    ----------
    results
        Box statistics of one model.
    observable
        ``h`` for ΔHmix or ``v_molar`` for V^E.

    Returns
    -------
    list[TimeSeriesEstimate]
        One estimate per row of ``systems.csv``.
    """
    estimates = box_estimates(results, observable, BOX_IDS)
    return [combine(row, estimates) for row in MIX]


# Per-model parity plots: metric they belong to, file stem, axis label, and the
# directory of per-point plots shown when a point is clicked
PARITY_PLOTS = {
    "hmix": ("ΔHmix / kJ mol⁻¹", "hmix_curves"),
    "density": ("Mixture density / g cm⁻³", "convergence_density"),
    "vex": ("V^E / cm³ mol⁻¹", "vex_curves"),
    "pure_density": ("Pure density / g cm⁻³", "convergence_density"),
}


def write_parity(
    filename: Path,
    model_name: str,
    labels: list[str],
    ref: list[float],
    pred: list[TimeSeriesEstimate],
    ref_sigma: list[float],
    label: str,
) -> None:
    """
    Write a one-model parity plot with 95 % error bars.

    Parameters
    ----------
    filename
        Output JSON path.
    model_name
        Model name.
    labels
        Box identifier of each point (stored in the point's ``customdata``).
    ref
        Experimental values.
    pred
        Predicted values with uncertainties.
    ref_sigma
        Experimental standard uncertainties, if known.
    label
        Quantity and unit for the axis labels.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ref": list(ref),
        model_name: [p.mean for p in pred],
        "pred_err": {model_name: [p.ci_95 for p in pred]},
    }
    data["ref_err"] = [1.96 * s if np.isfinite(s) else 0.0 for s in ref_sigma]
    plot_parity(
        filename=filename,
        title=f"{model_name}: click a point for details",
        x_label=f"Predicted {label}",
        y_label=f"Experimental {label}",
        hoverdata={"Labels": list(labels)},
    )(lambda: data)()


@pytest.fixture(scope="module")
def parity_plots(box_results) -> None:
    """
    Write per-model parity plots for every scored property.

    Parameters
    ----------
    box_results
        Box statistics for all models.
    """
    for model_name, results in box_results.items():
        hmix = point_estimates(results, "h")
        vex = point_estimates(results, "v_molar")
        density = box_estimates(results, "density", DENSITY_BOXES)
        pure = box_estimates(results, "density", PURE_BOXES)
        model_dir = OUT_PATH / model_name
        for prop, labels, ref, pred, sigma in (
            ("hmix", POINT_IDS, SYSTEMS.hmix_exp, hmix, SYSTEMS.hmix_sigma),
            (
                "density",
                DENSITY_BOXES,
                SYSTEMS.density_exp.iloc[DENSITY_ROWS],
                density,
                SYSTEMS.density_sigma.iloc[DENSITY_ROWS],
            ),
            (
                "vex",
                [POINT_IDS[i] for i in VEX_ROWS],
                SYSTEMS.vex_exp.iloc[VEX_ROWS],
                [vex[i] for i in VEX_ROWS],
                pd.Series(VEX_SIGMA),
            ),
            ("pure_density", PURE_BOXES, PURE.density, pure, PURE.sigma),
        ):
            write_parity(
                model_dir / f"parity_{prop}.json",
                model_name,
                labels,
                ref.tolist(),
                pred,
                sigma.tolist(),
                PARITY_PLOTS[prop][0],
            )


def write_scatter(filename: Path, data: dict, **kwargs) -> None:
    """
    Write a scatter figure with ``plot_scatter``.

    Parameters
    ----------
    filename
        Output JSON path.
    data
        Traces (and optional ``x_err``/``y_err``) in the ``plot_scatter`` format.
    **kwargs
        Title and axis labels passed to ``plot_scatter``.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("show_line", True)
    plot_scatter(filename=filename, **kwargs)(lambda: data)()


@pytest.fixture(scope="module")
def system_plots(box_results) -> None:
    """
    Write ΔHmix(x) and V^E(x) figures per model and system.

    Parameters
    ----------
    box_results
        Box statistics for all models.
    """
    for model_name, model_results in box_results.items():
        for observable, prop, label in (
            ("h", "hmix", "ΔHmix / kJ mol⁻¹"),
            ("v_molar", "vex", "V^E / cm³ mol⁻¹"),
        ):
            points = point_estimates(model_results, observable)
            for system_id, rows in SYSTEMS.groupby("system_id").groups.items():
                system = SYSTEMS.loc[list(rows)]
                x = system.x_box.tolist()
                data = {
                    "ref": [x, system[f"{prop}_exp"].tolist()],
                    model_name: [x, [points[i].mean for i in rows]],
                    "y_err": {model_name: [points[i].ci_95 for i in rows]},
                }
                sigma = system.hmix_sigma if prop == "hmix" else system.vex_sigma
                data["y_err"]["ref"] = (1.96 * sigma.fillna(0.0)).tolist()
                if f"{prop}_sage" in system:
                    data[SAGE_LABEL] = [x, system[f"{prop}_sage"].tolist()]
                # One copy per box, so clicking any point of the system opens it
                for i in rows:
                    write_scatter(
                        OUT_PATH
                        / model_name
                        / f"{prop}_curves"
                        / f"{POINT_IDS[i]}.json",
                        data,
                        title=f"{system_id} at {system.temperature.iloc[0]} K",
                        x_label=f"Mole fraction of {system_id.split('_')[0]}",
                        y_label=label,
                    )


def write_convergence(
    filename: Path,
    time: np.ndarray,
    values: np.ndarray,
    estimate: TimeSeriesEstimate,
    title: str,
    y_label: str,
) -> None:
    """
    Write block means, the production running mean and the final estimate.

    Parameters
    ----------
    filename
        Output JSON path.
    time
        Times of the whole run in ps, including equilibration.
    values
        Value at each time.
    estimate
        Production mean and its uncertainty.
    title
        Figure title.
    y_label
        Quantity and unit for the y axis.
    """
    data = pd.DataFrame({"t": time, "y": values})
    block = ((data.t - 1e-6) // CONVERGENCE_BLOCK_PS).clip(lower=0)
    means = data.groupby(block).mean()
    prod = data[data.t > EQUILIBRATION_PS]
    # Skip the first 50 ps, where the running mean is mostly noise
    running = prod.y.expanding().mean()[
        prod.t > EQUILIBRATION_PS + RUNNING_MEAN_SKIP_PS
    ]
    stride = max(1, len(prod) // 200)
    write_scatter(
        filename,
        {
            f"{CONVERGENCE_BLOCK_PS:.0f} ps block mean": [
                means.t.tolist(),
                means.y.tolist(),
            ],
            "running mean (production)": [
                prod.t[running.index].iloc[::stride].tolist(),
                running.iloc[::stride].tolist(),
            ],
            "production mean ± 95 % CI": [
                [float(prod.t.iloc[-1])],
                [estimate.mean],
            ],
            "y_err": {"production mean ± 95 % CI": [estimate.ci_95]},
        },
        title=title,
        x_label=f"Time / ps (production starts at {EQUILIBRATION_PS:.0f} ps)",
        y_label=y_label,
    )


@pytest.fixture(scope="module")
def convergence_plots(box_results, flags) -> None:
    """
    Write block-mean and running-mean time series per model and box.

    Parameters
    ----------
    box_results
        Box statistics for all models.
    flags
        Flags keyed by model and box.
    """
    for model_name, model_results in box_results.items():
        for box_id, result in model_results.items():
            data = pd.read_csv(CALC_PATH / model_name / f"{box_id}.csv")
            values = {
                "h": observables(data, N_MOLECULES[box_id])["h"],
                "density": data.density_g_cm3.to_numpy(),
            }
            box_flags = flags.get(model_name, {}).get(box_id)
            title_flags = "flags: " + "; ".join(box_flags) if box_flags else "no flags"
            for obs, label in (("h", "h / kJ mol⁻¹"), ("density", "density / g cm⁻³")):
                write_convergence(
                    OUT_PATH / model_name / f"convergence_{obs}" / f"{box_id}.json",
                    data.time_ps.to_numpy(),
                    values[obs],
                    result.estimates[obs],
                    title=f"{box_id}: {title_flags}",
                    y_label=label,
                )


@pytest.fixture(scope="module")
def vex_convergence_plots(box_results) -> None:
    """
    Write V^E block-mean and running-mean time series per model and point.

    V^E(t) combines the molar volumes of the mixture box and both pure boxes at
    each logged time, so its production mean is the scored V^E.

    Parameters
    ----------
    box_results
        Box statistics for all models.
    """
    for model_name, results in box_results.items():
        v_molar = {}
        for box_id in BOX_IDS:
            data = pd.read_csv(CALC_PATH / model_name / f"{box_id}.csv")
            v_molar[box_id] = (
                data.time_ps.to_numpy(),
                observables(data, N_MOLECULES[box_id])["v_molar"],
            )
        points = point_estimates(results, "v_molar")
        for i in VEX_ROWS:
            terms = [
                (w, *v_molar[b]) for w, b in zip(MIX[i], BOX_IDS, strict=True) if w
            ]
            time = terms[0][1]
            if not all(np.array_equal(t, time) for _, t, _ in terms):
                raise ValueError(
                    f"{model_name}/{POINT_IDS[i]}: boxes logged at different times"
                )
            write_convergence(
                OUT_PATH / model_name / "convergence_vex" / f"{POINT_IDS[i]}.json",
                time,
                sum(w * v for w, _, v in terms),
                points[i],
                title=f"{POINT_IDS[i]}: V^E from the mixture and pure boxes",
                y_label="V^E / cm³ mol⁻¹",
            )


@pytest.fixture(scope="module")
def flag_plots(box_results, flags) -> None:
    """
    Write a per-model overview of box flags against effective sample size.

    Clicking a point opens that box's enthalpy convergence plot.

    Parameters
    ----------
    box_results
        Box statistics for all models.
    flags
        Flags keyed by model and box.
    """
    for model_name, results in box_results.items():
        model_flags = flags.get(model_name, {})
        texts = ["; ".join(model_flags.get(b, [])) or "none" for b in BOX_IDS]
        write_scatter(
            OUT_PATH / model_name / "flag_summary.json",
            {
                "boxes": [
                    [results[b].estimates["h"].n_eff for b in BOX_IDS],
                    [len(model_flags.get(b, [])) for b in BOX_IDS],
                ]
            },
            title=f"{model_name}: flags per box (click a point for its time series)",
            x_label="Effective samples of h (production)",
            y_label="Number of flags",
            show_line=False,
            hoverdata={"Flags": texts, "Box": BOX_IDS},
        )


@pytest.fixture(scope="module")
def final_structures() -> None:
    """Write the last frame of each run for the app."""
    for model_name in MODELS:
        for box_id in BOX_IDS:
            with Trajectory(CALC_PATH / model_name / f"{box_id}.traj") as traj:
                atoms = traj[-1]
            (OUT_PATH / model_name).mkdir(parents=True, exist_ok=True)
            write(OUT_PATH / model_name / f"{box_id}.xyz", atoms)


@pytest.fixture(scope="module")
@build_table(
    filename=OUT_PATH / "binary_mixtures_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    hmix_errors,
    density_errors,
    vex_errors,
    pure_density_errors,
    flags,
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    hmix_errors
        ΔHmix MAE and MAZE for each model.
    density_errors
        Mixture density MAE and MAZE for each model.
    vex_errors
        V^E MAE and MAZE for each model.
    pure_density_errors
        Pure density MAE and MAZE for each model.
    flags
        Flags keyed by model and box.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "ΔHmix MAE": hmix_errors[0],
        "ΔHmix MAZE": hmix_errors[1],
        "Density MAE": density_errors[0],
        "Density MAZE": density_errors[1],
        "V^E MAE": vex_errors[0],
        "V^E MAZE": vex_errors[1],
        "Pure density MAE": pure_density_errors[0],
        "Pure density MAZE": pure_density_errors[1],
        "Flagged boxes": {model: len(flags.get(model, {})) for model in MODELS},
    }


def test_binary_mixtures(
    metrics: dict[str, dict],
    parity_plots: None,
    system_plots: None,
    convergence_plots: None,
    vex_convergence_plots: None,
    flag_plots: None,
    final_structures: None,
) -> None:
    """
    Run the binary mixtures analysis.

    Parameters
    ----------
    metrics
        All metric names and values for each model.
    parity_plots
        Per-model parity plots.
    system_plots
        Per-system ΔHmix(x) and V^E(x) figures.
    convergence_plots
        Per-box convergence figures.
    vex_convergence_plots
        Per-point V^E convergence figures.
    flag_plots
        Per-model flag overviews.
    final_structures
        Final frames for the app.
    """
    return
