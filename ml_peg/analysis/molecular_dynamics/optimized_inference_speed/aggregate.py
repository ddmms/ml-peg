"""Derive, aggregate and plot optimised inference speed results."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator
import pandas as pd
from plotly import colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml_peg.analysis.molecular_dynamics.optimized_inference_speed.validate import (
    get_reference_calculator,
    validate_run,
)
from ml_peg.analysis.utils.decorators import build_table, get_model_colour
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.contract import (
    RECORD_FILE,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.run import (
    load_container_models,
)

APP_DATA = APP_ROOT / "data" / "molecular_dynamics" / "optimized_inference_speed"
THRESHOLDS, TOOLTIPS, WEIGHTS = load_metrics_config(
    Path(__file__).with_name("metrics.yml")
)
SECONDS_PER_DAY = 86_400
STAT_COLUMNS = ("us_per_atom_step", "atom_steps_per_s", "ns_per_day", "gpu_peak_mib")
GROUP = ["model", "mlpeg_model", "precision", "material", "n_atoms"]


def derived_quantities(
    timestamps: dict[str, float], measured_steps: int, n_atoms: int, timestep_fs: float
) -> dict[str, float]:
    """
    Derive speed metrics from a run's timestamps.

    Parameters
    ----------
    timestamps
        Unix timestamps ``start``, ``setup_end``, ``warmup_end`` and ``measure_end``.
    measured_steps
        Number of measured MD steps.
    n_atoms
        Number of atoms.
    timestep_fs
        MD timestep in fs.

    Returns
    -------
    dict[str, float]
        Time per step, µs per atom-step, atom-steps per second, ns/day, and setup
        and warm-up times.
    """
    time_per_step = (timestamps["measure_end"] - timestamps["warmup_end"]) / (
        measured_steps
    )
    return {
        "time_per_step_s": time_per_step,
        "us_per_atom_step": time_per_step / n_atoms * 1e6,
        "atom_steps_per_s": n_atoms / time_per_step,
        "ns_per_day": timestep_fs * 1e-6 / time_per_step * SECONDS_PER_DAY,
        "setup_time_s": timestamps["setup_end"] - timestamps["start"],
        "warmup_time_s": timestamps["warmup_end"] - timestamps["setup_end"],
    }


def collect_runs(
    campaign: Path,
    reference_getter: Callable[[str], Calculator] = get_reference_calculator,
) -> pd.DataFrame:
    """
    Validate every run in a campaign and tabulate the results.

    Parameters
    ----------
    campaign
        Campaign directory.
    reference_getter
        Function returning the reference calculator for an ML-PEG model name.

    Returns
    -------
    pd.DataFrame
        One row per run.
    """
    spec = json.loads((campaign / "in" / "spec.json").read_text(encoding="utf8"))
    systems = {system["material"]: system for system in spec["systems"]}
    containers = load_container_models()
    rows = []
    for record_path in sorted((campaign / "out").glob(f"*/*/*/{RECORD_FILE}")):
        record = json.loads(record_path.read_text(encoding="utf8"))
        system = systems[record["material"]]
        size = next(s for s in system["sizes"] if s["n_atoms"] == record["n_atoms"])
        valid, checks, result = validate_run(
            record_path.parent,
            record,
            spec,
            containers.get(record["model"], {}),
            reference_getter,
        )
        reasons = [check.reason for check in checks if not check.ok]
        if valid is None and record.get("error"):
            reasons.append(record["error"])
        memory = record.get("memory", {})
        row = {
            "run_id": record["run_id"],
            "model": record["model"],
            "mlpeg_model": record["mlpeg_model"],
            "precision": record["precision"],
            "material": record["material"],
            "n_atoms": record["n_atoms"],
            "repeat": record["repeat"],
            "reference": size.get("reference", False),
            "status": record["status"],
            "valid": valid,
            "reasons": "; ".join(reasons),
            "exit_code": record.get("exit_code"),
            "wall_time_s": record.get("wall_time_s"),
            "gpu_method": memory.get("gpu_method"),
            "gpu_peak_mib": _mib(memory.get("gpu_peak_bytes")),
            "gpu_peak_measured_mib": _mib(memory.get("gpu_peak_measured_bytes")),
            "host_rss_peak_mib": _mib(memory.get("host_rss_peak_bytes")),
            "host_rss_peak_measured_mib": _mib(
                memory.get("host_rss_peak_measured_bytes")
            ),
            "engine": result.implementation.engine if result else None,
        }
        if valid and result.status == "ok":
            timestep = spec["md"]["timestep_fs"]
            if system["contains_hydrogen"]:
                timestep = spec["md"]["timestep_fs_if_hydrogen"]
            row["timestep_fs"] = timestep
            row.update(
                derived_quantities(
                    result.timestamps_unix.model_dump(),
                    result.steps.measured,
                    record["n_atoms"],
                    timestep,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _mib(value: float | None) -> float | None:
    """
    Convert bytes to MiB.

    Parameters
    ----------
    value
        Number of bytes.

    Returns
    -------
    float | None
        Number of MiB, if `value` is set.
    """
    return None if value is None else value / 2**20


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate valid, successful runs over repeats.

    Parameters
    ----------
    runs
        One row per run, from `collect_runs`.

    Returns
    -------
    pd.DataFrame
        Median, minimum and maximum of each quantity per model, precision, material
        and size.
    """
    if "us_per_atom_step" not in runs:
        return pd.DataFrame(columns=GROUP)
    ok = runs[(runs["status"] == "ok") & (runs["valid"] == True)]  # noqa: E712
    grouped = ok.groupby(GROUP)[list(STAT_COLUMNS)]
    summary = grouped.agg(["median", "min", "max"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary["n_valid"] = grouped.size()
    return summary.reset_index()


def _log_figure(
    summary: pd.DataFrame, column: str, y_label: str, title: str
) -> go.Figure:
    """
    Plot a quantity against atoms on log-log axes, one panel per material.

    Parameters
    ----------
    summary
        Aggregated results from `summarise`.
    column
        Quantity to plot; its median, min and max columns are used.
    y_label
        Y-axis label.
    title
        Figure title.

    Returns
    -------
    go.Figure
        Figure with one trace per model and precision.
    """
    materials = sorted(summary["material"].unique()) if len(summary) else [""]
    fig = make_subplots(
        rows=1, cols=len(materials), subplot_titles=materials, shared_yaxes=True
    )
    palette = colors.qualitative.Plotly
    for (model, mlpeg_model, precision), group in summary.groupby(
        ["model", "mlpeg_model", "precision"]
    ):
        for col, material in enumerate(materials, start=1):
            data = group[group["material"] == material].sort_values("n_atoms")
            if data.empty:
                continue
            median = data[f"{column}_median"]
            fig.add_trace(
                go.Scatter(
                    x=data["n_atoms"],
                    y=median,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": data[f"{column}_max"] - median,
                        "arrayminus": median - data[f"{column}_min"],
                    },
                    mode="lines+markers",
                    name=f"{model} ({precision})",
                    legendgroup=f"{model}-{precision}",
                    showlegend=col == 1,
                    line={"color": get_model_colour(mlpeg_model, palette)},
                    hovertemplate=(
                        "<b>%{x} atoms</b><br>median: %{y:.4g}<extra>"
                        f"{model} ({precision})</extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(type="log", title_text="Atoms")
    fig.update_yaxes(type="log")
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_layout(title=title)
    return fig


def plot_scaling(summary: pd.DataFrame) -> go.Figure:
    """
    Plot µs per atom-step against atoms.

    Parameters
    ----------
    summary
        Aggregated results from `summarise`.

    Returns
    -------
    go.Figure
        Scaling figure.
    """
    return _log_figure(
        summary, "us_per_atom_step", "µs / (atom · step)", "Inference cost scaling"
    )


def plot_memory(summary: pd.DataFrame, device_mib: float | None) -> go.Figure:
    """
    Plot peak GPU memory against atoms, with the device memory limit.

    Parameters
    ----------
    summary
        Aggregated results from `summarise`.
    device_mib
        Total device memory in MiB, if known.

    Returns
    -------
    go.Figure
        Memory figure.
    """
    fig = _log_figure(summary, "gpu_peak_mib", "Peak GPU memory / MiB", "GPU memory")
    if device_mib:
        fig.add_hline(
            y=device_mib,
            line_dash="dash",
            line_color="grey",
            annotation_text="device memory",
            row="all",
            col="all",
        )
    return fig


def plot_pareto() -> None:
    """
    Plot speed against the ML-PEG accuracy score.

    Raises
    ------
    NotImplementedError
        Not part of the proof of concept.
    """
    # TODO: plot µs/atom-step at the reference size against each model's overall
    # ML-PEG score, highlighting the Pareto front.
    raise NotImplementedError("Pareto plot is not implemented yet.")


def compute_metrics(runs: pd.DataFrame, summary: pd.DataFrame) -> dict[str, dict]:
    """
    Compute table metrics for each ML-PEG model.

    Parameters
    ----------
    runs
        One row per run, from `collect_runs`.
    summary
        Aggregated results from `summarise`.

    Returns
    -------
    dict[str, dict]
        Metric name to ML-PEG model to value.
    """
    metrics: dict[str, dict] = {
        "Time per atom-step": {},
        "Largest system": {},
        "Valid runs": {},
    }
    if runs.empty:
        return metrics
    reference = runs[runs["reference"]][["material", "n_atoms"]].drop_duplicates()
    at_reference = summary.merge(reference, on=["material", "n_atoms"])
    for model, group in at_reference.groupby("mlpeg_model"):
        # Best implementation and precision per material, averaged over materials
        best = group.groupby("material")["us_per_atom_step_median"].min()
        metrics["Time per atom-step"][model] = float(best.mean())
    for model, group in summary.groupby("mlpeg_model"):
        metrics["Largest system"][model] = int(group["n_atoms"].max())
    launched = runs[runs["valid"].notna()]
    for model, group in launched.groupby("mlpeg_model"):
        metrics["Valid runs"][model] = float(group["valid"].astype(bool).mean())
    return metrics


def write_metrics_table(metrics: dict[str, dict], app_dir: Path) -> dict[str, dict]:
    """
    Write the metrics table for the app.

    Parameters
    ----------
    metrics
        Metric name to ML-PEG model to value.
    app_dir
        Directory to write app data to.

    Returns
    -------
    dict[str, dict]
        The metrics.
    """

    @build_table(
        filename=app_dir / "optimized_inference_speed_metrics_table.json",
        metric_tooltips=TOOLTIPS,
        thresholds=THRESHOLDS,
        weights=WEIGHTS,
    )
    def table() -> dict[str, dict]:
        """
        Get the metrics.

        Returns
        -------
        dict[str, dict]
            Metric name to ML-PEG model to value.
        """
        return metrics

    return table()


def analyse_campaign(
    campaign: Path,
    app_dir: Path = APP_DATA,
    reference_getter: Callable[[str], Calculator] = get_reference_calculator,
) -> dict[str, Any]:
    """
    Validate, aggregate and plot a campaign.

    Writes ``results.csv``, ``summary.csv`` and HTML figures to
    ``<campaign>/analysis``, and the metrics table, figures and element info to
    `app_dir`.

    Parameters
    ----------
    campaign
        Campaign directory.
    app_dir
        Directory to write app data to.
    reference_getter
        Function returning the reference calculator for an ML-PEG model name.

    Returns
    -------
    dict[str, Any]
        Per-run results, summary, figures and metrics.
    """
    campaign = campaign.resolve()
    out_dir = campaign / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    app_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(campaign, reference_getter)
    summary = summarise(runs)
    runs.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    hardware_path = campaign / "out" / "hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else {}
    gpu = hardware.get("gpu") or {}
    device_mib = _mib(gpu.get("memory_total_bytes"))
    figures = {
        "scaling": plot_scaling(summary),
        "memory": plot_memory(summary, device_mib),
    }
    for name, fig in figures.items():
        fig.write_json(app_dir / f"figure_{name}.json")
        fig.write_html(out_dir / f"figure_{name}.html")

    spec = json.loads((campaign / "in" / "spec.json").read_text(encoding="utf8"))
    elements = sorted({e for system in spec["systems"] for e in system["elements"]})
    (app_dir / "info.json").write_text(json.dumps({"elements": elements}, indent=1))

    metrics = write_metrics_table(compute_metrics(runs, summary), app_dir)
    return {"runs": runs, "summary": summary, "figures": figures, "metrics": metrics}
