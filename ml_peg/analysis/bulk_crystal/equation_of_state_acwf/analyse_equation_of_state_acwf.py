"""Analyse the ACWF equation-of-state benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.data import atomic_numbers
from ase.io import read
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest

from ml_peg.analysis.bulk_crystal.equation_of_state_acwf.acwf_utils import (
    birch_murnaghan,
    delta,
    epsilon,
    fit_birch_murnaghan,
    nu,
    relative_difference,
)
from ml_peg.analysis.utils.decorators import build_table, plot_periodic_table
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "equation_of_state_acwf" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "equation_of_state_acwf"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATASETS = ("unaries", "oxides")
CONFIGURATIONS = (
    "BCC",
    "FCC",
    "SC",
    "Diamond",
    "XO",
    "XO2",
    "XO3",
    "X2O",
    "X2O3",
    "X2O5",
)

# The reference is non-spin-polarised PBE without +U. Most MLIP training sets are
# spin-polarised, and use PBE+U for the oxides of these elements (MP2020 settings).
PLUS_U_ELEMENTS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}
MAGNETIC_ELEMENTS = {"Cr", "Mn", "Fe", "Co", "Ni"} | {
    element
    for element, number in atomic_numbers.items()
    if 57 <= number <= 71 or 89 <= number <= 103
}

# DFT codes shown as baselines in the box plots, and the suffix of their result files
BASELINE_CODES = {"FLEUR": "fleur", "WIEN2k": "wien2k-dk_0.06", "VASP": "vasp"}

# Box plots of the signed relative differences: parameter and initial axis range (%)
BOX_PLOT_RANGES = {"V0": 20, "B0": 50, "B1": 100}

# Per-element periodic tables: metric name and file stem.
PERIODIC_TABLES = {
    "ε": "epsilon_periodic_table",
    "ν": "nu_periodic_table",
    "Unphysical E(V)": "unphysical_periodic_table",
    "Δ": "delta_periodic_table",
}


def is_flagged(element: str, configuration: str) -> bool:
    """
    Check whether the reference differs from typical MLIP training data.

    Parameters
    ----------
    element
        Element the structure is built for.
    configuration
        Configuration, such as ``BCC`` or ``X2O3``.

    Returns
    -------
    bool
        Whether the element is magnetic, or the structure is an oxide of an element
        for which +U is applied in Materials Project calculations.
    """
    is_oxide = configuration.startswith("X")
    return element in MAGNETIC_ELEMENTS or (is_oxide and element in PLUS_U_ELEMENTS)


def load_reference(source: str = "AE-average") -> dict[str, dict[str, float]]:
    """
    Load the all-electron PBE reference Birch-Murnaghan parameters.

    Parameters
    ----------
    source
        Suffix of the result files. Default is ``AE-average``, the reference. The
        values in ``BASELINE_CODES`` load the results of a DFT code instead.

    Returns
    -------
    dict[str, dict[str, float]]
        ``V0`` (Å³/atom), ``B0`` (eV/Å³) and ``B1`` for each structure, labelled as
        ``{element}-{configuration}``.
    """
    data_path = (
        download_s3_data(
            key="inputs/bulk_crystal/equation_of_state_acwf/equation_of_state_acwf.zip",
            filename="equation_of_state_acwf.zip",
        )
        / "equation_of_state_acwf"
        / "reference"
    )
    reference = {}
    for dataset in DATASETS:
        with open(
            data_path / f"results-{dataset}-verification-PBE-v1-{source}.json",
            encoding="utf8",
        ) as f:
            data = json.load(f)
        for key, fit in data["BM_fit_data"].items():
            label = key.replace("-X/", "-")
            reference[label] = {
                "V0": fit["min_volume"] / data["num_atoms_in_sim_cell"][key],
                "B0": fit["bulk_modulus_ev_ang3"],
                "B1": fit["bulk_deriv"],
            }
    return reference


def load_curves(model_name: str) -> dict[str, dict[str, np.ndarray]] | None:
    """
    Load the E(V) curves computed with a model.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    dict[str, dict[str, np.ndarray]] | None
        Volumes (Å³/atom) and energies (eV/atom) for each structure, or ``None`` if
        the model was not run.
    """
    curves = {}
    for dataset in DATASETS:
        filename = CALC_PATH / model_name / f"{dataset}.extxyz"
        if not filename.exists():
            return None
        for atoms in read(filename, index=":"):
            label = atoms.info["label"]
            curve = curves.setdefault(label, {"volumes": [], "energies": []})
            curve["volumes"].append(atoms.get_volume() / len(atoms))
            curve["energies"].append(atoms.info["mlip_energy"] / len(atoms))
    return {
        label: {key: np.array(values) for key, values in curve.items()}
        for label, curve in curves.items()
    }


def compare_curve(
    volumes: np.ndarray, energies: np.ndarray, reference: dict[str, float]
) -> dict[str, float | str | None]:
    """
    Fit a model E(V) curve and compare it with the reference.

    Parameters
    ----------
    volumes
        Volumes (Å³/atom).
    energies
        Energies (eV/atom).
    reference
        Reference ``V0``, ``B0`` and ``B1``.

    Returns
    -------
    dict[str, float | str | None]
        ``status`` and the fitted parameters and comparison measures. ``status`` is
        ``"ok"``, ``"failed"`` (a calculation failed), ``"no_minimum"`` (the fit has no
        minimum), or ``"v0_outside"`` (the minimum lies outside the sampled
        volumes). The measures are ``None`` unless a minimum was found.
    """
    result = dict.fromkeys(
        ("E0", "V0", "B0", "B1", "ε", "ν", "Δ", "V0 error", "B0 error", "B1 error")
    )
    if not np.all(np.isfinite(energies)):
        return result | {"status": "failed"}

    fit = fit_birch_murnaghan(volumes, energies)
    if fit is None:
        return result | {"status": "no_minimum"}

    tested = (fit["V0"], fit["B0"], fit["B1"])
    ref = (reference["V0"], reference["B0"], reference["B1"])
    inside = volumes.min() <= fit["V0"] <= volumes.max()
    return {
        "status": "ok" if inside else "v0_outside",
        "E0": fit["E0"],
        "V0": fit["V0"],
        "B0": fit["B0"],
        "B1": fit["B1"],
        "ε": epsilon(*tested, *ref),
        "ν": nu(*tested, *ref),
        "Δ": delta(*tested, *ref),
        "V0 error": 100 * abs(relative_difference(tested[0], ref[0])),
        "B0 error": 100 * abs(relative_difference(tested[1], ref[1])),
        "B1 error": 100 * abs(relative_difference(tested[2], ref[2])),
    }


def compare_model(
    model_curves: dict[str, dict[str, np.ndarray]],
    reference: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    Compare the curves of one model with the reference, for every structure.

    Parameters
    ----------
    model_curves
        Curves of the model for each structure.
    reference
        Reference parameters for each structure.

    Returns
    -------
    dict[str, dict]
        Comparison for each reference structure. A structure missing from the model
        outputs, for example after an interrupted run, is counted as failed.
    """
    missing = {"volumes": np.full(7, np.nan), "energies": np.full(7, np.nan)}
    return {
        label: compare_curve(**model_curves.get(label, missing), reference=ref)
        for label, ref in reference.items()
    }


@pytest.fixture(scope="module")
def reference() -> dict[str, dict[str, float]]:
    """
    Get the reference Birch-Murnaghan parameters.

    Returns
    -------
    dict[str, dict[str, float]]
        Reference parameters for each structure.
    """
    return load_reference()


@pytest.fixture(scope="module")
def curves() -> dict[str, dict[str, dict[str, np.ndarray]] | None]:
    """
    Get the E(V) curves of all models.

    Returns
    -------
    dict[str, dict[str, dict[str, np.ndarray]] | None]
        Curves for each model, or ``None`` for models that were not run.
    """
    return {model_name: load_curves(model_name) for model_name in MODELS}


@pytest.fixture(scope="module")
def comparisons(
    reference: dict[str, dict[str, float]],
    curves: dict[str, dict[str, dict[str, np.ndarray]] | None],
) -> dict[str, dict[str, dict] | None]:
    """
    Compare the curves of all models with the reference.

    Parameters
    ----------
    reference
        Reference parameters for each structure.
    curves
        Curves for each model.

    Returns
    -------
    dict[str, dict[str, dict] | None]
        Comparison for each structure and model, or ``None`` for models that were not
        run.
    """
    return {
        model_name: None
        if model_curves is None
        else compare_model(model_curves, reference)
        for model_name, model_curves in curves.items()
    }


def median_measure(results: list[dict]) -> float:
    """
    Get the median of a measure, counting curves without a minimum as infinite.

    Failed curves are counted as worse than any fitted curve rather than excluded, so
    that a model does not improve its score by failing difficult structures.

    Parameters
    ----------
    results
        Values of the measure, ``None`` where no minimum was found.

    Returns
    -------
    float
        Median value, ``inf`` if more than half the curves have no minimum, or
        ``nan`` if there are no structures.
    """
    if not results:
        return np.nan
    return float(np.median([np.inf if value is None else value for value in results]))


def model_measure(
    comparison: dict[str, dict] | None, measure: str, flagged: bool = True
) -> float | None:
    """
    Get the median of a measure over all structures for one model.

    Parameters
    ----------
    comparison
        Comparison for each structure of one model.
    measure
        Name of the measure.
    flagged
        Whether to include structures flagged by ``is_flagged``. Default is True.

    Returns
    -------
    float | None
        Median value, or ``None`` if the model was not run or more than half the
        curves have no minimum.
    """
    if comparison is None:
        return None
    value = median_measure(
        [
            result[measure]
            for label, result in comparison.items()
            if flagged or not is_flagged(*label.split("-"))
        ]
    )
    return value if np.isfinite(value) else None


def unphysical_share(comparison: dict[str, dict] | None) -> float | None:
    """
    Get the percentage of structures without a usable E(V) curve.

    A minimum outside the sampled volumes is not counted, as ν already penalises the
    error in V0.

    Parameters
    ----------
    comparison
        Comparison for each structure of one model.

    Returns
    -------
    float | None
        Percentage of structures for which a calculation failed or the fit has no
        minimum, or ``None`` if the model was not run.
    """
    if comparison is None:
        return None
    statuses = [result["status"] for result in comparison.values()]
    return (
        100
        * sum(status in {"failed", "no_minimum"} for status in statuses)
        / len(statuses)
    )


@pytest.fixture
@build_table(
    filename=OUT_PATH / "eos_acwf_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(comparisons: dict[str, dict[str, dict] | None]) -> dict[str, dict]:
    """
    Get all ACWF equation-of-state metrics.

    Parameters
    ----------
    comparisons
        Comparison for each structure and model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    table = {
        "ε": {},
        "ν": {},
        "Unphysical E(V)": {},
        "ε (no +U or magnetic)": {},
        "Δ": {},
        "V0 error": {},
        "B0 error": {},
        "B1 error": {},
    }
    for model_name, comparison in comparisons.items():
        for measure in ("ε", "ν", "Δ", "V0 error", "B0 error", "B1 error"):
            table[measure][model_name] = model_measure(comparison, measure)
        table["ε (no +U or magnetic)"][model_name] = model_measure(
            comparison, "ε", flagged=False
        )
        table["Unphysical E(V)"][model_name] = unphysical_share(comparison)
    return table


def element_values(comparison: dict[str, dict], measure: str) -> dict[str, float]:
    """
    Get the value of a metric for each element, over its ten structures.

    Parameters
    ----------
    comparison
        Comparison for each structure of one model.
    measure
        Name of the metric in ``PERIODIC_TABLES``.

    Returns
    -------
    dict[str, float]
        Median of the measure, or percentage of unphysical curves, for each element.
        A median without a minimum is shown as the "bad" threshold of the metric.
    """
    by_element = {}
    for label, result in comparison.items():
        by_element.setdefault(label.split("-")[0], []).append(result)

    values = {}
    for element, results in by_element.items():
        if measure == "Unphysical E(V)":
            values[element] = unphysical_share(dict(enumerate(results)))
        else:
            value = median_measure([result[measure] for result in results])
            values[element] = (
                value if np.isfinite(value) else DEFAULT_THRESHOLDS[measure]["bad"]
            )
    return values


def element_hoverdata(comparison: dict[str, dict]) -> dict[str, dict[str, str]]:
    """
    Get hover text listing ε and the status of each structure of an element.

    Parameters
    ----------
    comparison
        Comparison for each structure of one model.

    Returns
    -------
    dict[str, dict[str, str]]
        Hover text for each configuration and element.
    """
    hoverdata = {configuration: {} for configuration in CONFIGURATIONS}
    for label, result in comparison.items():
        element, configuration = label.split("-")
        if result["ε"] is None:
            text = result["status"]
        else:
            text = f"ε = {result['ε']:.3f}"
            if result["status"] != "ok":
                text += f" ({result['status']})"
        if is_flagged(element, configuration):
            text += " [+U/magnetic]"
        hoverdata[configuration][element] = text
    return hoverdata


@pytest.fixture
def periodic_tables(comparisons: dict[str, dict[str, dict] | None]) -> None:
    """
    Write per-model periodic-table heatmaps for each metric.

    Parameters
    ----------
    comparisons
        Comparison for each structure and model.
    """
    for model_name, comparison in comparisons.items():
        if comparison is None:
            continue
        hoverdata = element_hoverdata(comparison)
        for measure, file_stem in PERIODIC_TABLES.items():
            unit = DEFAULT_THRESHOLDS[measure]["unit"]
            title = f"{measure} ({unit})" if unit else measure
            plot_periodic_table(
                title=f"{title} - {model_name}",
                colorbar_title=title,
                hoverdata=hoverdata,
                filename=str(OUT_PATH / model_name / f"{file_stem}.json"),
                zmin=DEFAULT_THRESHOLDS[measure]["good"],
                zmax=DEFAULT_THRESHOLDS[measure]["bad"],
                colorscale="Viridis_r",
            )(element_values)(comparison, measure)


def signed_differences(
    parameters: dict[str, dict[str, float] | None],
    reference: dict[str, dict[str, float]],
) -> dict[str, list[float]]:
    """
    Get the signed relative differences in V0, B0 and B1 from the reference.

    Parameters
    ----------
    parameters
        Birch-Murnaghan parameters for each structure, ``None`` or missing where
        there are none.
    reference
        Reference parameters for each structure.

    Returns
    -------
    dict[str, list[float]]
        Relative differences (%), as defined by ACWF, for each parameter, over the
        structures with parameters.
    """
    differences = {name: [] for name in BOX_PLOT_RANGES}
    for label, ref in reference.items():
        tested = parameters.get(label)
        if tested is None or tested.get("V0") is None:
            continue
        for name in differences:
            differences[name].append(100 * relative_difference(tested[name], ref[name]))
    return differences


@pytest.fixture
def relative_errors_figure(
    reference: dict[str, dict[str, float]],
    comparisons: dict[str, dict[str, dict] | None],
) -> None:
    """
    Write box plots of the relative differences in V0, B0 and B1.

    The DFT codes in ``BASELINE_CODES`` are shown above the models, as in Figure 1 of
    Yu et al., Mater. Genome Eng. Adv. 2, e58 (2024).

    Parameters
    ----------
    reference
        Reference parameters for each structure.
    comparisons
        Comparison for each structure and model.
    """
    rows = {
        code: signed_differences(load_reference(suffix), reference)
        for code, suffix in BASELINE_CODES.items()
    }
    rows |= {
        model_name: signed_differences(comparison, reference)
        for model_name, comparison in comparisons.items()
        if comparison is not None
    }

    fig = make_subplots(
        rows=1,
        cols=len(BOX_PLOT_RANGES),
        shared_yaxes=True,
        horizontal_spacing=0.03,
    )
    for col, (name, limit) in enumerate(BOX_PLOT_RANGES.items(), start=1):
        for row, differences in rows.items():
            fig.add_trace(
                go.Box(
                    x=differences[name],
                    name=row,
                    orientation="h",
                    boxpoints="outliers",
                    marker={"size": 3},
                    line={"width": 1},
                    marker_color="#1f77b4" if row in BASELINE_CODES else "#2ca02c",
                    showlegend=False,
                    hovertemplate=f"{row}: %{{x:.2f}}%<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            title_text=f"{name} difference (%)", range=[-limit, limit], row=1, col=col
        )
        fig.add_vline(x=0, line={"color": "grey", "width": 1}, row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    # Dashed line between the DFT codes and the models
    fig.add_hline(
        y=len(BASELINE_CODES) - 0.5, line={"color": "grey", "dash": "dash", "width": 1}
    )
    fig.update_layout(
        title=(
            "Relative differences from the all-electron reference "
            "(structures without a fitted minimum are not shown)"
        ),
        height=250 + 40 * len(rows),
    )
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.write_json(OUT_PATH / "relative_errors_box.json")


@pytest.fixture
def eos_figures(
    reference: dict[str, dict[str, float]],
    curves: dict[str, dict[str, dict[str, np.ndarray]] | None],
    comparisons: dict[str, dict[str, dict] | None],
) -> None:
    """
    Write the E(V) curves of the ten structures of each element, for each model.

    Parameters
    ----------
    reference
        Reference parameters for each structure.
    curves
        Curves for each model.
    comparisons
        Comparison for each structure and model.
    """
    elements = sorted({label.split("-")[0] for label in reference})
    for model_name, model_curves in curves.items():
        if model_curves is None:
            continue
        out_dir = OUT_PATH / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for element in elements:
            fig = make_subplots(
                rows=2,
                cols=5,
                subplot_titles=CONFIGURATIONS,
                horizontal_spacing=0.05,
                vertical_spacing=0.15,
            )
            for index, configuration in enumerate(CONFIGURATIONS):
                label = f"{element}-{configuration}"
                ref = reference[label]
                result = comparisons[model_name][label]
                row, col = divmod(index, 5)
                position = {"row": row + 1, "col": col + 1}
                fine = np.linspace(0.92, 1.08, 81) * ref["V0"]
                fig.add_scatter(
                    x=fine / ref["V0"],
                    y=1000
                    * birch_murnaghan(fine, 0.0, ref["V0"], ref["B0"], ref["B1"]),
                    mode="lines",
                    line={"color": "black"},
                    name="PBE (all-electron)",
                    legendgroup="ref",
                    showlegend=index == 0,
                    **position,
                )
                if label not in model_curves:
                    fig.layout.annotations[index].text = f"{configuration}: failed"
                    continue
                volumes = model_curves[label]["volumes"]
                energies = model_curves[label]["energies"]
                if not np.any(np.isfinite(energies)):
                    fig.layout.annotations[index].text = f"{configuration}: failed"
                    continue
                e0 = result["E0"] if result["E0"] is not None else np.nanmin(energies)
                fig.add_scatter(
                    x=volumes / ref["V0"],
                    y=1000 * (energies - e0),
                    mode="markers",
                    marker={"color": "#d62728"},
                    name=model_name,
                    legendgroup="model",
                    showlegend=index == 0,
                    **position,
                )
                if result["V0"] is not None:
                    fig.add_scatter(
                        x=fine / ref["V0"],
                        y=1000
                        * birch_murnaghan(
                            fine, 0.0, result["V0"], result["B0"], result["B1"]
                        ),
                        mode="lines",
                        line={"color": "#d62728", "dash": "dash"},
                        name=f"{model_name} (fit)",
                        legendgroup="fit",
                        showlegend=index == 0,
                        **position,
                    )
                    fig.layout.annotations[
                        index
                    ].text = f"{configuration}: ε = {result['ε']:.2f}"
                else:
                    fig.layout.annotations[
                        index
                    ].text = f"{configuration}: {result['status']}"
            fig.update_xaxes(title_text="V / V0 (reference)", row=2)
            fig.update_yaxes(title_text="E - E0 (meV/atom)", col=1)
            fig.update_layout(
                title=f"{element}: E(V) - {model_name}",
                height=600,
            )
            fig.write_json(out_dir / f"{element}_eos_figure.json")


def test_equation_of_state_acwf(
    metrics: dict[str, dict],
    periodic_tables: None,
    eos_figures: None,
    relative_errors_figure: None,
) -> None:
    """
    Run the ACWF equation-of-state analysis.

    Parameters
    ----------
    metrics
        All metric names and values for all models.
    periodic_tables
        Per-model periodic-table heatmaps (side effect only).
    eos_figures
        Per-model and per-element E(V) figures (side effect only).
    relative_errors_figure
        Box plots of the relative differences in V0, B0 and B1 (side effect only).
    """
    get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*.extxyz",
        index="::7",
        info_keys=["label"],
        write_structs=False,
        out_path=OUT_PATH,
    )
