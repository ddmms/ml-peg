"""Analyse the Al-Cu-Mg-Zn metallurgy regression benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from warnings import warn

from ase.io import read, write
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "alloy_metallurgy" / "AlZnCuMg_regression" / "outputs"
OUT_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "AlZnCuMg_regression"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)
LATTICE_PROPERTIES = ("lattice_a", "lattice_b", "lattice_c")
ELASTIC_MODULI_PROPERTIES = ("k_voigt", "g_voigt")
ELASTIC_CONSTANT_PROPERTIES = tuple(
    f"C_{row + 1}{column + 1}" for row in range(6) for column in range(row + 1)
)
REQUIRED_ELEMENTS = ("Al", "Cu", "Mg", "Zn")
EV_PER_A2_TO_MJ_PER_M2 = 16021.76634
EV_TO_MEV = 1000.0
PURE_ELEMENT_PROPERTY_SPECS = {
    "8100": (
        "Al FCC",
        (
            ("a", "bulk", "lattice_a"),
            ("K", "elastic", "k_voigt"),
            ("G", "elastic", "g_voigt"),
            ("C₁₁", "elastic", "C_11"),
            ("C₁₂", "elastic", "C_21"),
            ("C₄₄", "elastic", "C_44"),
            ("γsurf (111)", "surfaces", "8100-SurfaceEnergy_111"),
            ("γsurf (100)", "surfaces", "8100-SurfaceEnergy_100"),
            ("γsurf (110)", "surfaces", "8100-SurfaceEnergy_110"),
            ("γSSF", "stacking_faults", "8100-StableSF"),
            ("γUSF", "stacking_faults", "8100-UnStableSF"),
        ),
    ),
    "635950": (
        "Cu FCC",
        (
            ("a", "bulk", "lattice_a"),
            ("K", "elastic", "k_voigt"),
            ("G", "elastic", "g_voigt"),
            ("C₁₁", "elastic", "C_11"),
            ("C₁₂", "elastic", "C_21"),
            ("C₄₄", "elastic", "C_44"),
            ("γsurf (111)", "surfaces", "635950-SurfaceEnergy_111"),
            ("γsurf (100)", "surfaces", "635950-SurfaceEnergy_100"),
            ("γsurf (110)", "surfaces", "635950-SurfaceEnergy_110"),
            ("γSSF", "stacking_faults", "635950-StableSF"),
            ("γUSF", "stacking_faults", "635950-UnStableSF"),
        ),
    ),
    "9226": (
        "Mg HCP",
        (
            ("a", "bulk", "lattice_a"),
            ("c", "bulk", "lattice_c"),
            ("K", "elastic", "k_voigt"),
            ("G", "elastic", "g_voigt"),
            ("C₁₁", "elastic", "C_11"),
            ("C₁₂", "elastic", "C_21"),
            ("C₁₃", "elastic", "C_31"),
            ("C₃₃", "elastic", "C_33"),
            ("C₄₄", "elastic", "C_44"),
            ("γsurf (0001)", "surfaces", "9226-SurfaceHCP_0001"),
            ("γsurf (101̅0)", "surfaces", "9226-SurfaceHCP_10m10"),
            ("γsurf (112̅0)", "surfaces", "9226-SurfaceHCP_11m20"),
            ("γsurf (11̅01)", "surfaces", "9226-SurfaceHCP_1m101"),
            ("γsurf (101̅2)", "surfaces", "9226-SurfaceHCP_10m12"),
            ("γsurf (112̅1)", "surfaces", "9226-SurfaceHCP_11m21"),
            ("γsurf (112̅2)", "surfaces", "9226-SurfaceHCP_11m22"),
        ),
    ),
    "122929": (
        "Zn HCP",
        (
            ("a", "bulk", "lattice_a"),
            ("c", "bulk", "lattice_c"),
            ("K", "elastic", "k_voigt"),
            ("G", "elastic", "g_voigt"),
            ("C₁₁", "elastic", "C_11"),
            ("C₁₂", "elastic", "C_21"),
            ("C₁₃", "elastic", "C_31"),
            ("C₃₃", "elastic", "C_33"),
            ("C₄₄", "elastic", "C_44"),
            ("γsurf (0001)", "surfaces", "122929-SurfaceHCP_0001"),
            ("γsurf (101̅0)", "surfaces", "122929-SurfaceHCP_10m10"),
            ("γsurf (112̅0)", "surfaces", "122929-SurfaceHCP_11m20"),
            ("γsurf (11̅01)", "surfaces", "122929-SurfaceHCP_1m101"),
            ("γsurf (101̅2)", "surfaces", "122929-SurfaceHCP_10m12"),
            ("γsurf (112̅1)", "surfaces", "122929-SurfaceHCP_11m21"),
            ("γsurf (112̅2)", "surfaces", "122929-SurfaceHCP_11m22"),
        ),
    ),
}


def write_benchmark_info() -> None:
    """Write benchmark-wide element requirements for application filtering."""
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(
        json.dumps({"elements": REQUIRED_ELEMENTS}, indent=1),
        encoding="utf8",
    )


def load_references() -> dict[str, Any]:
    """
    Load evalpot-format DFT references.

    Returns
    -------
    dict[str, Any]
        Reference data keyed by evalpot material parameter name.
    """
    reference_path = CALC_PATH / "references" / "DFT.json"
    try:
        with reference_path.open(encoding="utf8") as file:
            references = json.load(file)
        if not isinstance(references, dict):
            raise TypeError("expected a JSON object")
        return references
    except Exception as exc:
        warn(
            f"Unable to load reference data from {reference_path}: {exc}",
            stacklevel=2,
        )
        return {}


def reference_value(
    references: dict[str, Any], oqmd_id: str, property_name: str
) -> float | None:
    """
    Extract one scalar reference value.

    Parameters
    ----------
    references
        Evalpot reference dictionary.
    oqmd_id
        OQMD identifier without the ``OQMD_`` prefix.
    property_name
        Evalpot property suffix.

    Returns
    -------
    float | None
        Scalar value if present and numeric.
    """
    entry = references.get(f"{oqmd_id}-{property_name}")
    if not isinstance(entry, (list, tuple)) or not entry:
        return None
    try:
        value = float(entry[0])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def reference_series(references: dict[str, Any], key: str) -> list[float]:
    """
    Extract one numeric reference series from evalpot reference data.

    Parameters
    ----------
    references
        Evalpot reference dictionary.
    key
        Full evalpot material parameter name.

    Returns
    -------
    list[float]
        Numeric reference series, or an empty list when unavailable.
    """
    entry = references.get(key)
    if not isinstance(entry, (list, tuple)) or not entry:
        return []
    if not isinstance(entry[0], list):
        return []

    values = []
    for value in entry[0]:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(converted):
            return []
        values.append(converted)
    return values


def solute_solute_reference_series(
    references: dict[str, Any], reference_key: str
) -> list[float]:
    """
    Extract solute-solute references, allowing sorted DFT pair keys.

    Parameters
    ----------
    references
        Evalpot-format DFT reference dictionary.
    reference_key
        Solute-solute record key (e.g. ``8100-SolSol_Cu_Zn``).

    Returns
    -------
    list[float]
        Binding energies for the requested pair, or an empty list if not found.
    """
    values = reference_series(references, f"{reference_key}_BindingEnergy")
    if values:
        return values

    matrix_id, _, pair_label = reference_key.partition("-SolSol_")
    pair = pair_label.split("_")
    if not matrix_id or len(pair) != 2:
        return []
    sorted_key = f"{matrix_id}-SolSol_{'_'.join(sorted(pair))}"
    if sorted_key == reference_key:
        return []
    return reference_series(references, f"{sorted_key}_BindingEnergy")


def scalar_reference(references: dict[str, Any], key: str) -> float | None:
    """
    Extract one scalar reference value by full evalpot key.

    Parameters
    ----------
    references
        Evalpot reference dictionary.
    key
        Full evalpot material parameter name.

    Returns
    -------
    float | None
        Scalar value if present and numeric.
    """
    entry = references.get(key)
    if not isinstance(entry, (list, tuple)) or not entry:
        return None
    try:
        value = float(entry[0])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_model_records() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load calculated scalar records.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Records keyed by model name and OQMD ID.
    """
    records_by_model = {}
    for model_name in MODELS:
        record_path = CALC_PATH / model_name / "bulk_properties.json"
        if not record_path.exists():
            continue
        try:
            with record_path.open(encoding="utf8") as file:
                data = json.load(file)
            records_by_model[model_name] = {
                record["oqmd_id"]: record for record in data["structures"]
            }
        except Exception as exc:
            warn(f"Unable to load bulk records for {model_name}: {exc}", stacklevel=2)
    return records_by_model


def load_elastic_records() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load calculated elastic-property records.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Elastic records keyed by model name and OQMD ID.
    """
    records_by_model = {}
    for model_name in MODELS:
        record_path = CALC_PATH / model_name / "elastic_properties.json"
        if not record_path.exists():
            continue
        try:
            with record_path.open(encoding="utf8") as file:
                data = json.load(file)
            records_by_model[model_name] = {
                record["oqmd_id"]: record for record in data["structures"]
            }
        except Exception as exc:
            warn(
                f"Unable to load elastic records for {model_name}: {exc}",
                stacklevel=2,
            )
    return records_by_model


def load_solute_solute_records() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load calculated solute-solute binding records.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Solute-solute records keyed by model name and evalpot reference key.
    """
    records_by_model = {}
    for model_name in MODELS:
        record_path = CALC_PATH / model_name / "solute_solute_bindings.json"
        if not record_path.exists():
            continue
        try:
            with record_path.open(encoding="utf8") as file:
                data = json.load(file)
            records_by_model[model_name] = {
                record["reference_key"]: record for record in data["interactions"]
            }
        except Exception as exc:
            warn(
                f"Unable to load solute records for {model_name}: {exc}",
                stacklevel=2,
            )
    return records_by_model


def load_fault_surface_records() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """
    Load calculated surface, stacking-fault, and GSF records.

    Returns
    -------
    dict[str, dict[str, dict[str, dict[str, Any]]]]
        Records keyed by model name, record group, and evalpot reference key.
    """
    records_by_model = {}
    for model_name in MODELS:
        record_path = CALC_PATH / model_name / "fault_surface_properties.json"
        if not record_path.exists():
            continue
        try:
            with record_path.open(encoding="utf8") as file:
                data = json.load(file)
            records_by_model[model_name] = {
                group_name: {
                    record["reference_key"]: record
                    for record in data.get(group_name, [])
                }
                for group_name in (
                    "surfaces",
                    "stacking_faults",
                    "gsf",
                    "solute_stacking_faults",
                )
            }
        except Exception as exc:
            warn(
                f"Unable to load fault records for {model_name}: {exc}",
                stacklevel=2,
            )
    return records_by_model


def common_structure_ids(
    records_by_model: dict[str, dict[str, dict[str, Any]]],
    property_name: str,
) -> list[str]:
    """
    Get structure IDs present in all model outputs and reference data.

    Parameters
    ----------
    records_by_model
        Calculated records keyed by model.
    property_name
        Evalpot reference property suffix.

    Returns
    -------
    list[str]
        Common OQMD IDs.
    """
    references = load_references()
    if not records_by_model:
        return []

    common_ids = set.intersection(
        *(set(records) for records in records_by_model.values())
    )
    valid_ids = []
    for oqmd_id in sorted(common_ids):
        try:
            ref_value = reference_value(references, oqmd_id, property_name)
            model_values = [
                float(model_records[oqmd_id][property_name])
                for model_records in records_by_model.values()
            ]
            if ref_value is not None and all(map(math.isfinite, model_values)):
                valid_ids.append(oqmd_id)
        except Exception as exc:
            warn(f"Unable to process {oqmd_id} {property_name}: {exc}", stacklevel=2)
    return valid_ids


def get_property_structure_ids(property_name: str) -> list[str]:
    """
    Get structure IDs used by one property parity plot.

    Parameters
    ----------
    property_name
        Evalpot reference property suffix.

    Returns
    -------
    list[str]
        OQMD IDs available across the current outputs and references.
    """
    return common_structure_ids(load_model_records(), property_name)


def get_lattice_component_labels() -> list[str]:
    """
    Get labels used by the lattice-constant parity plot.

    Returns
    -------
    list[str]
        Structure and lattice-axis labels available across outputs and references.
    """
    records_by_model = load_model_records()
    labels = []
    for property_name in LATTICE_PROPERTIES:
        axis = property_name.removeprefix("lattice_")
        labels.extend(
            f"{oqmd_id} {axis}"
            for oqmd_id in common_structure_ids(records_by_model, property_name)
        )
    return labels


def scalar_property_values(property_name: str) -> dict[str, list[float]]:
    """
    Get reference and predicted scalar values for one property.

    Parameters
    ----------
    property_name
        Evalpot reference property suffix and calc-record key.

    Returns
    -------
    dict[str, list[float]]
        Reference and model values.
    """
    references = load_references()
    records_by_model = load_model_records()
    structure_ids = common_structure_ids(records_by_model, property_name)

    results = {"ref": []} | {model_name: [] for model_name in records_by_model}
    for oqmd_id in structure_ids:
        ref_value = reference_value(references, oqmd_id, property_name)
        if ref_value is None:
            continue
        results["ref"].append(ref_value)
        for model_name, model_records in records_by_model.items():
            results[model_name].append(float(model_records[oqmd_id][property_name]))

    return results


def multi_property_values(property_names: tuple[str, ...]) -> dict[str, list[float]]:
    """
    Get flattened reference and predicted values for related scalar properties.

    Parameters
    ----------
    property_names
        Evalpot reference property suffixes and calc-record keys.

    Returns
    -------
    dict[str, list[float]]
        Reference and model values flattened in property then structure order.
    """
    references = load_references()
    records_by_model = load_model_records()
    results = {"ref": []} | {model_name: [] for model_name in records_by_model}

    for property_name in property_names:
        for oqmd_id in common_structure_ids(records_by_model, property_name):
            ref_value = reference_value(references, oqmd_id, property_name)
            if ref_value is None:
                continue
            results["ref"].append(ref_value)
            for model_name, model_records in records_by_model.items():
                results[model_name].append(float(model_records[oqmd_id][property_name]))

    return results


def property_values_from_records(
    records_by_model: dict[str, dict[str, dict[str, Any]]],
    property_name: str,
) -> dict[str, list[float]]:
    """
    Get reference and predicted values from a supplied record collection.

    Parameters
    ----------
    records_by_model
        Calculated records keyed by model.
    property_name
        Evalpot reference property suffix and calc-record key.

    Returns
    -------
    dict[str, list[float]]
        Reference and model values.
    """
    references = load_references()
    structure_ids = common_structure_ids(records_by_model, property_name)
    results = {"ref": []} | {model_name: [] for model_name in records_by_model}

    for oqmd_id in structure_ids:
        ref_value = reference_value(references, oqmd_id, property_name)
        if ref_value is None:
            continue
        results["ref"].append(ref_value)
        for model_name, model_records in records_by_model.items():
            results[model_name].append(float(model_records[oqmd_id][property_name]))

    return results


def multi_property_values_from_records(
    records_by_model: dict[str, dict[str, dict[str, Any]]],
    property_names: tuple[str, ...],
) -> dict[str, list[float]]:
    """
    Get flattened reference and predicted values from supplied records.

    Parameters
    ----------
    records_by_model
        Calculated records keyed by model.
    property_names
        Evalpot reference property suffixes and calc-record keys.

    Returns
    -------
    dict[str, list[float]]
        Reference and model values flattened in property then structure order.
    """
    references = load_references()
    results = {"ref": []} | {model_name: [] for model_name in records_by_model}

    for property_name in property_names:
        for oqmd_id in common_structure_ids(records_by_model, property_name):
            ref_value = reference_value(references, oqmd_id, property_name)
            if ref_value is None:
                continue
            results["ref"].append(ref_value)
            for model_name, model_records in records_by_model.items():
                results[model_name].append(float(model_records[oqmd_id][property_name]))

    return results


def labels_from_records(
    records_by_model: dict[str, dict[str, dict[str, Any]]],
    property_name: str,
) -> list[str]:
    """
    Get hover labels for one property from supplied records.

    Parameters
    ----------
    records_by_model
        Calculated records keyed by model.
    property_name
        Evalpot reference property suffix and calc-record key.

    Returns
    -------
    list[str]
        Common OQMD IDs.
    """
    return common_structure_ids(records_by_model, property_name)


def multi_property_labels_from_records(
    records_by_model: dict[str, dict[str, dict[str, Any]]],
    property_names: tuple[str, ...],
) -> list[str]:
    """
    Get hover labels for flattened multi-property data.

    Parameters
    ----------
    records_by_model
        Calculated records keyed by model.
    property_names
        Evalpot reference property suffixes and calc-record keys.

    Returns
    -------
    list[str]
        Structure and property labels.
    """
    labels = []
    for property_name in property_names:
        labels.extend(
            f"{oqmd_id} {property_name}"
            for oqmd_id in common_structure_ids(records_by_model, property_name)
        )
    return labels


def has_series_data(values: dict[str, list[float]]) -> bool:
    """
    Check whether a parity series contains reference and model data.

    Parameters
    ----------
    values
        Reference and model values.

    Returns
    -------
    bool
        True if the series can be plotted and scored.
    """
    reference_values = values.get("ref", [])
    model_values = [
        series for model_name, series in values.items() if model_name != "ref"
    ]
    if not reference_values or not model_values:
        return False
    expected_length = len(reference_values)
    return all(
        len(series) == expected_length and all(math.isfinite(value) for value in series)
        for series in [reference_values, *model_values]
    )


def write_parity_plot(
    values: dict[str, list[float]],
    *,
    filename: Path,
    title: str,
    x_label: str,
    y_label: str,
    hoverdata: dict[str, list[str]],
) -> None:
    """
    Write a parity plot for optional analysis data.

    Parameters
    ----------
    values
        Reference and model values.
    filename
        Output Plotly JSON path.
    title
        Plot title.
    x_label
        Predicted-value axis label.
    y_label
        Reference-value axis label.
    hoverdata
        Hover labels for the plot points.
    """

    @plot_parity(
        filename=filename,
        title=title,
        x_label=x_label,
        y_label=y_label,
        hoverdata=hoverdata,
    )
    def _plot_values() -> dict[str, list[float]]:
        """
        Return pre-collected reference and model value arrays.

        Returns
        -------
        dict[str, list[float]]
            Reference and per-model value lists passed to the parity decorator.
        """
        return values

    _plot_values()


def errors_from_values(values: dict[str, list[float]]) -> dict[str, float]:
    """
    Get mean absolute errors from reference and model values.

    Parameters
    ----------
    values
        Reference and model values.

    Returns
    -------
    dict[str, float]
        MAE by model.
    """
    return {
        model_name: mae(values["ref"], model_values)
        for model_name, model_values in values.items()
        if model_name != "ref"
    }


def write_pure_element_plot(filename: Path) -> None:
    """
    Write pure-element bulk, elastic, surface, and stacking-fault deviations.

    Parameters
    ----------
    filename
        Output path for the combined Plotly JSON figure.
    """
    references = load_references()
    bulk_records = load_model_records()
    elastic_records = load_elastic_records()
    fault_records = load_fault_surface_records()
    model_names = list(dict.fromkeys((*bulk_records, *elastic_records, *fault_records)))
    if not references or not model_names:
        return

    fig = make_subplots(
        rows=len(PURE_ELEMENT_PROPERTY_SPECS),
        cols=1,
        subplot_titles=[title for title, _ in PURE_ELEMENT_PROPERTY_SPECS.values()],
        vertical_spacing=0.08,
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    plotted = False

    for row, (structure_id, (_, property_specs)) in enumerate(
        PURE_ELEMENT_PROPERTY_SPECS.items(), start=1
    ):
        fig.add_hline(
            y=0,
            line={"color": "black", "dash": "dot", "width": 1},
            row=row,
            col=1,
        )
        for color_index, model_name in enumerate(model_names):
            labels = []
            deviations = []
            hover_values = []
            for label, source, property_name in property_specs:
                reference_key = (
                    property_name
                    if source in ("surfaces", "stacking_faults")
                    else f"{structure_id}-{property_name}"
                )
                reference = scalar_reference(references, reference_key)
                if reference in (None, 0):
                    continue

                if source == "bulk":
                    record = bulk_records.get(model_name, {}).get(structure_id, {})
                    value_key = property_name
                elif source == "elastic":
                    record = elastic_records.get(model_name, {}).get(structure_id, {})
                    value_key = property_name
                else:
                    record = (
                        fault_records.get(model_name, {})
                        .get(source, {})
                        .get(property_name, {})
                    )
                    value_key = (
                        "surface_energy"
                        if source == "surfaces"
                        else "stacking_fault_energy"
                    )
                if value_key not in record:
                    continue

                try:
                    prediction = float(record[value_key])
                    if not math.isfinite(prediction):
                        raise ValueError("non-finite prediction")
                    labels.append(label)
                    deviations.append(100.0 * (prediction - reference) / reference)
                    hover_values.append([prediction, reference])
                except Exception as exc:
                    warn(
                        f"Unable to process {model_name} {reference_key}: {exc}",
                        stacklevel=2,
                    )

            if not labels:
                continue
            plotted = True
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=deviations,
                    customdata=hover_values,
                    mode="lines+markers",
                    name=model_name,
                    line={"color": colors[color_index % len(colors)], "width": 1.5},
                    marker={"size": 7},
                    showlegend=(row == 1),
                    hovertemplate=(
                        "%{x}<br>Deviation: %{y:.2f}%"
                        "<br>Predicted: %{customdata[0]:.4g}"
                        "<br>DFT: %{customdata[1]:.4g}<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=row,
                col=1,
            )

    if not plotted:
        return
    fig.update_layout(
        title_text="Pure-element properties",
        height=1100,
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.04,
            "yanchor": "bottom",
        },
        margin={"b": 90, "l": 80, "r": 30, "t": 150},
    )
    fig.update_xaxes(tickangle=-35, showgrid=True)
    fig.update_yaxes(title_text="Deviation from DFT / %", zeroline=False)
    fig.write_json(filename)


def write_custom_solute_solute_plot(records_by_model: dict, filename: Path):
    """
    Write a multi-panel Plotly JSON of solute-solute binding energies.

    Parameters
    ----------
    records_by_model
        Mapping of model name to solute-solute binding record dicts.
    filename
        Output path for the Plotly JSON figure.
    """
    references = load_references()
    common_keys = sorted(
        set.intersection(*(set(records) for records in records_by_model.values()))
    )
    if not common_keys:
        return
    num_keys = len(common_keys)
    cols = min(3, num_keys)
    rows = math.ceil(num_keys / max(1, cols))

    subplot_titles = []
    for reference_key in common_keys:
        try:
            matrix_id, _, pair_label = reference_key.partition("-SolSol_")
            matrix = {"8100": "Al", "635950": "Cu"}.get(matrix_id, matrix_id)
            solute_1, solute_2 = pair_label.split("_")
            subplot_titles.append(f"{solute_1}-{solute_2} in {matrix}")
        except Exception as exc:
            warn(f"Unable to format plot title {reference_key}: {exc}", stacklevel=2)
            subplot_titles.append(reference_key)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    for i, reference_key in enumerate(common_keys):
        row = i // cols + 1
        col = i % cols + 1

        reference_values = solute_solute_reference_series(references, reference_key)
        if reference_values:
            x_vals = list(range(1, len(reference_values) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=reference_values,
                    mode="lines+markers",
                    name="DFT",
                    marker={"color": "black", "symbol": "square"},
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        for c_idx, (model_name, model_records) in enumerate(records_by_model.items()):
            if reference_key in model_records:
                model_vals = model_records[reference_key].get("binding_energies")
                if not isinstance(model_vals, list):
                    continue
                x_vals = list(range(1, len(model_vals) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=model_vals,
                        mode="lines+markers",
                        name=model_name,
                        marker={
                            "color": colors[c_idx % len(colors)],
                            "symbol": "circle",
                        },
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title_text="Solute-solute binding energies in Al and Cu",
        height=max(400, 300 * rows),
    )
    for i in range(1, num_keys + 1):
        fig.layout[f"yaxis{i}"].title = "Binding energy / meV"
        fig.layout[f"xaxis{i}"].title = "Nearest-neighbor index"

    fig.write_json(filename)


def write_custom_gsf_plot(records_by_model: dict, filename: Path):
    """
    Write a multi-panel Plotly JSON of generalized stacking-fault energies.

    Parameters
    ----------
    records_by_model
        Mapping of model name to fault-surface property record dicts.
    filename
        Output path for the Plotly JSON figure.
    """
    references = load_references()
    common_keys = sorted(
        set.intersection(
            *(set(records["gsf"]) for records in records_by_model.values())
        )
    )
    if not common_keys:
        return
    num_keys = len(common_keys)
    cols = min(3, num_keys)
    rows = math.ceil(num_keys / max(1, cols))

    subplot_titles = {
        "NOTINOQMD_00001-GSF_0m11": "Al₂Cu θ (01̅1)",
        "NOTINOQMD_00002-GSF_111": "Al₃Cu θ″ (111)",
    }
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[subplot_titles.get(key, key) for key in common_keys],
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2"]

    for i, reference_key in enumerate(common_keys):
        row = i // cols + 1
        col = i % cols + 1
        reference_values = reference_series(references, f"{reference_key}_normE")
        if reference_values:
            x_vals = list(range(1, len(reference_values) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=[value * EV_PER_A2_TO_MJ_PER_M2 for value in reference_values],
                    mode="lines+markers",
                    name="DFT",
                    marker={"color": "black", "symbol": "square"},
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        for c_idx, (model_name, model_records) in enumerate(records_by_model.items()):
            if reference_key in model_records["gsf"]:
                model_vals = model_records["gsf"][reference_key].get("norm_energies")
                if not isinstance(model_vals, list):
                    continue
                x_vals = list(range(1, len(model_vals) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=[value * EV_PER_A2_TO_MJ_PER_M2 for value in model_vals],
                        mode="lines+markers",
                        name=model_name,
                        marker={
                            "color": colors[c_idx % len(colors)],
                            "symbol": "circle",
                        },
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title_text="Generalized stacking-fault energies",
        height=max(400, 300 * rows),
    )
    for i in range(1, num_keys + 1):
        fig.layout[f"yaxis{i}"].title = "Normalized GSF energy / mJ m⁻²"
        fig.layout[f"xaxis{i}"].title = "Site index"

    fig.write_json(filename)


def write_custom_solute_sf_plot(records_by_model: dict, filename: Path):
    """
    Write a multi-panel Plotly JSON of solute-stacking-fault interaction energies.

    Parameters
    ----------
    records_by_model
        Mapping of model name to fault-surface property record dicts.
    filename
        Output path for the Plotly JSON figure.
    """
    references = load_references()
    common_keys = sorted(
        set.intersection(
            *(
                set(records["solute_stacking_faults"])
                for records in records_by_model.values()
            )
        )
    )
    if not common_keys:
        return
    num_keys = len(common_keys)
    cols = min(3, num_keys)
    rows = math.ceil(num_keys / max(1, cols))

    subplot_titles = []
    for reference_key in common_keys:
        matrix_id, _, solute = reference_key.partition("-SolSF_")
        matrix = {"8100": "Al", "635950": "Cu"}.get(matrix_id, matrix_id)
        subplot_titles.append(f"{solute} in {matrix}")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2"]

    for i, reference_key in enumerate(common_keys):
        row = i // cols + 1
        col = i % cols + 1
        reference_values = reference_series(references, reference_key)
        if reference_values:
            x_vals = list(range(1, len(reference_values) + 1))  # layer indices
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=[value * EV_TO_MEV for value in reference_values],
                    mode="lines+markers",
                    name="DFT",
                    marker={"color": "black", "symbol": "square"},
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        for c_idx, (model_name, model_records) in enumerate(records_by_model.items()):
            if reference_key in model_records["solute_stacking_faults"]:
                model_vals = model_records["solute_stacking_faults"][reference_key].get(
                    "interaction_energies"
                )
                if not isinstance(model_vals, list):
                    continue
                x_vals = list(range(1, len(model_vals) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=[value * EV_TO_MEV for value in model_vals],
                        mode="lines+markers",
                        name=model_name,
                        marker={
                            "color": colors[c_idx % len(colors)],
                            "symbol": "circle",
                        },
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title_text="Solute-stacking-fault interactions", height=max(400, 300 * rows)
    )
    for i in range(1, num_keys + 1):
        fig.layout[f"yaxis{i}"].title = "Interaction energy / meV"
        fig.layout[f"xaxis{i}"].title = "SF index"

    fig.write_json(filename)


def solute_solute_binding_values(
    records_by_model: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Get flattened solute-solute binding values and labels.

    Parameters
    ----------
    records_by_model
        Calculated solute-solute records keyed by model.

    Returns
    -------
    tuple[dict[str, list[float]], list[str]]
        Reference/model values in meV and hover labels for the flattened series.
    """
    references = load_references()
    results = {"ref": []} | {model_name: [] for model_name in records_by_model}
    labels = []
    if not records_by_model:
        return results, labels

    common_keys = sorted(
        set.intersection(*(set(records) for records in records_by_model.values()))
    )
    for reference_key in common_keys:
        try:
            reference_values = solute_solute_reference_series(references, reference_key)
            if not reference_values:
                continue
            model_values_by_name = {
                model_name: model_records[reference_key]["binding_energies"]
                for model_name, model_records in records_by_model.items()
            }
            common_length = min(
                len(reference_values),
                *(len(values) for values in model_values_by_name.values()),
            )
            for index in range(common_length):
                model_values = {
                    model_name: float(values[index])
                    for model_name, values in model_values_by_name.items()
                }
                if not all(map(math.isfinite, model_values.values())):
                    continue
                results["ref"].append(reference_values[index])
                for model_name, model_value in model_values.items():
                    results[model_name].append(model_value)
                labels.append(f"{reference_key} shell {index + 1}")
        except Exception as exc:
            warn(
                f"Unable to process solute record {reference_key}: {exc}",
                stacklevel=2,
            )
    return results, labels


def solute_solute_metrics() -> dict[str, dict[str, float]]:
    """
    Build optional solute-solute binding metrics.

    Returns
    -------
    dict[str, dict[str, float]]
        Solute-solute binding MAE by model, or an empty dictionary when outputs
        are absent.
    """
    records_by_model = load_solute_solute_records()
    if not records_by_model:
        return {}

    values, labels = solute_solute_binding_values(records_by_model)
    if not has_series_data(values):
        return {}

    write_custom_solute_solute_plot(
        records_by_model,
        filename=OUT_PATH / "figure_solute_solute_bindings.json",
    )
    return {"Solute-Solute Binding MAE": errors_from_values(values)}


def fault_surface_scalar_values(
    records_by_model: dict[str, dict[str, dict[str, dict[str, Any]]]],
    group_name: str,
    value_key: str,
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Get flattened scalar fault/surface values and labels.

    Parameters
    ----------
    records_by_model
        Fault/surface records keyed by model and group.
    group_name
        Record group inside ``fault_surface_properties.json``.
    value_key
        Scalar value key inside each record.

    Returns
    -------
    tuple[dict[str, list[float]], list[str]]
        Reference/model values and hover labels.
    """
    references = load_references()
    results = {"ref": []} | {model_name: [] for model_name in records_by_model}
    labels = []
    if not records_by_model:
        return results, labels

    model_groups = records_by_model.values()
    common_keys = sorted(
        set.intersection(
            *(set(model_records[group_name]) for model_records in model_groups)
        )
    )
    for reference_key in common_keys:
        try:
            ref_value = scalar_reference(references, reference_key)
            if ref_value is None:
                continue
            model_values = {
                model_name: float(model_records[group_name][reference_key][value_key])
                for model_name, model_records in records_by_model.items()
            }
            if not all(map(math.isfinite, model_values.values())):
                continue
            results["ref"].append(ref_value)
            for model_name, model_value in model_values.items():
                results[model_name].append(model_value)
            labels.append(reference_key)
        except Exception as exc:
            warn(
                f"Unable to process {group_name} record {reference_key}: {exc}",
                stacklevel=2,
            )
    return results, labels


def gsf_values(
    records_by_model: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Get flattened GSF normalized energies and labels.

    Parameters
    ----------
    records_by_model
        Fault/surface records keyed by model and group.

    Returns
    -------
    tuple[dict[str, list[float]], list[str]]
        Reference/model values in mJ/m^2 and hover labels.
    """
    references = load_references()
    results = {"ref": []} | {model_name: [] for model_name in records_by_model}
    labels = []
    if not records_by_model:
        return results, labels

    common_keys = sorted(
        set.intersection(
            *(set(model_records["gsf"]) for model_records in records_by_model.values())
        )
    )
    for reference_key in common_keys:
        try:
            reference_values = reference_series(references, f"{reference_key}_normE")
            if not reference_values:
                continue
            model_values_by_name = {
                model_name: model_records["gsf"][reference_key]["norm_energies"]
                for model_name, model_records in records_by_model.items()
            }
            common_length = min(
                len(reference_values),
                *(len(values) for values in model_values_by_name.values()),
            )
            for index in range(common_length):
                model_values = {
                    model_name: float(values[index])
                    for model_name, values in model_values_by_name.items()
                }
                if not all(map(math.isfinite, model_values.values())):
                    continue
                results["ref"].append(reference_values[index] * EV_PER_A2_TO_MJ_PER_M2)
                for model_name, model_value in model_values.items():
                    results[model_name].append(model_value * EV_PER_A2_TO_MJ_PER_M2)
                labels.append(f"{reference_key} point {index + 1}")
        except Exception as exc:
            warn(
                f"Unable to process GSF record {reference_key}: {exc}",
                stacklevel=2,
            )
    return results, labels


def solute_stacking_fault_values(
    records_by_model: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Get flattened solute-stacking-fault interaction values and labels.

    Parameters
    ----------
    records_by_model
        Fault/surface records keyed by model and group.

    Returns
    -------
    tuple[dict[str, list[float]], list[str]]
        Reference/model values in meV and hover labels.
    """
    references = load_references()
    results = {"ref": []} | {model_name: [] for model_name in records_by_model}
    labels = []
    if not records_by_model:
        return results, labels

    common_keys = sorted(
        set.intersection(
            *(
                set(model_records["solute_stacking_faults"])
                for model_records in records_by_model.values()
            )
        )
    )
    for reference_key in common_keys:
        try:
            reference_values = reference_series(references, reference_key)
            if not reference_values:
                continue
            model_values_by_name = {
                model_name: model_records["solute_stacking_faults"][reference_key][
                    "interaction_energies"
                ]
                for model_name, model_records in records_by_model.items()
            }
            common_length = min(
                len(reference_values),
                *(len(values) for values in model_values_by_name.values()),
            )
            for index in range(common_length):
                model_values = {
                    model_name: float(values[index])
                    for model_name, values in model_values_by_name.items()
                }
                if not all(map(math.isfinite, model_values.values())):
                    continue
                results["ref"].append(reference_values[index] * EV_TO_MEV)
                for model_name, model_value in model_values.items():
                    results[model_name].append(model_value * EV_TO_MEV)
                labels.append(f"{reference_key} layer {index}")
        except Exception as exc:
            warn(
                f"Unable to process solute-SF record {reference_key}: {exc}",
                stacklevel=2,
            )
    return results, labels


def fault_surface_metrics() -> dict[str, dict[str, float]]:
    """
    Build optional surface, stacking-fault, and GSF metrics.

    Returns
    -------
    dict[str, dict[str, float]]
        Fault/surface metrics by model, or an empty dictionary when outputs are
        absent.
    """
    records_by_model = load_fault_surface_records()
    if not records_by_model:
        return {}

    metrics = {}
    surface_values, surface_labels = fault_surface_scalar_values(
        records_by_model,
        "surfaces",
        "surface_energy",
    )
    if has_series_data(surface_values):
        write_parity_plot(
            surface_values,
            filename=OUT_PATH / "figure_surface_energies.json",
            title="Surface energies",
            x_label="Predicted surface energy / mJ m^-2",
            y_label="DFT surface energy / mJ m^-2",
            hoverdata={"Surface": surface_labels},
        )
        metrics["Surface Energy MAE"] = errors_from_values(surface_values)

    stacking_values, stacking_labels = fault_surface_scalar_values(
        records_by_model,
        "stacking_faults",
        "stacking_fault_energy",
    )
    if has_series_data(stacking_values):
        write_parity_plot(
            stacking_values,
            filename=OUT_PATH / "figure_stacking_fault_energies.json",
            title="Stacking-fault energies",
            x_label="Predicted stacking-fault energy / mJ m^-2",
            y_label="DFT stacking-fault energy / mJ m^-2",
            hoverdata={"Fault": stacking_labels},
        )
        metrics["Stacking Fault Energy MAE"] = errors_from_values(stacking_values)

    gsf_energy_values, gsf_labels = gsf_values(records_by_model)
    if has_series_data(gsf_energy_values):
        write_custom_gsf_plot(
            records_by_model,
            filename=OUT_PATH / "figure_gsf_energies.json",
        )
        metrics["GSF Energy MAE"] = errors_from_values(gsf_energy_values)

    solute_sf_values, solute_sf_labels = solute_stacking_fault_values(records_by_model)
    if has_series_data(solute_sf_values):
        write_custom_solute_sf_plot(
            records_by_model,
            filename=OUT_PATH / "figure_solute_stacking_faults.json",
        )
        metrics["Solute-Stacking Fault MAE"] = errors_from_values(solute_sf_values)

    return metrics


def elastic_metrics() -> dict[str, dict[str, float]]:
    """
    Build optional elastic-moduli and elastic-constant metrics.

    Returns
    -------
    dict[str, dict[str, float]]
        Elastic metrics by model, or an empty dictionary when elastic outputs are
        absent.
    """
    records_by_model = load_elastic_records()
    if not records_by_model:
        return {}

    metrics = {}
    plot_specs = {
        "k_voigt": (
            "Bulk Modulus MAE",
            OUT_PATH / "figure_bulk_modulus.json",
            "Bulk moduli",
            "Predicted bulk modulus / GPa",
            "DFT bulk modulus / GPa",
        ),
        "g_voigt": (
            "Shear Modulus MAE",
            OUT_PATH / "figure_shear_modulus.json",
            "Shear moduli",
            "Predicted shear modulus / GPa",
            "DFT shear modulus / GPa",
        ),
    }
    for property_name, plot_spec in plot_specs.items():
        metric_name, filename, title, x_label, y_label = plot_spec
        values = property_values_from_records(records_by_model, property_name)
        if not has_series_data(values):
            continue
        write_parity_plot(
            values,
            filename=filename,
            title=title,
            x_label=x_label,
            y_label=y_label,
            hoverdata={"OQMD ID": labels_from_records(records_by_model, property_name)},
        )
        metrics[metric_name] = errors_from_values(values)

    elastic_constant_values = multi_property_values_from_records(
        records_by_model,
        ELASTIC_CONSTANT_PROPERTIES,
    )
    if has_series_data(elastic_constant_values):
        write_parity_plot(
            elastic_constant_values,
            filename=OUT_PATH / "figure_elastic_constants.json",
            title="Elastic constants",
            x_label="Predicted elastic constant / GPa",
            y_label="DFT elastic constant / GPa",
            hoverdata={
                "Elastic constant": multi_property_labels_from_records(
                    records_by_model,
                    ELASTIC_CONSTANT_PROPERTIES,
                )
            },
        )
        metrics["Elastic Constant MAE"] = errors_from_values(elastic_constant_values)

    return metrics


STRUCTURE_IDS = get_property_structure_ids("solformation_energy")
LATTICE_COMPONENT_LABELS = get_lattice_component_labels()
BETA_STRUCTURE_IDS = get_property_structure_ids("angle_beta")


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_solute_formation_energy.json",
    title="Solute-referenced formation energies",
    x_label="Predicted solute-referenced formation energy / eV atom^-1",
    y_label="DFT solute-referenced formation energy / eV atom^-1",
    hoverdata={"OQMD ID": STRUCTURE_IDS},
)
def solute_formation_energies() -> dict[str, list[float]]:
    """
    Get reference and predicted solute-referenced formation energies.

    Returns
    -------
    dict[str, list[float]]
        Reference and model solute-referenced formation energies in eV/atom.
    """
    return scalar_property_values("solformation_energy")


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_volume_peratom.json",
    title="Volumes per atom",
    x_label="Predicted volume / Angstrom^3 atom^-1",
    y_label="DFT volume / Angstrom^3 atom^-1",
    hoverdata={"OQMD ID": STRUCTURE_IDS},
)
def volumes_per_atom() -> dict[str, list[float]]:
    """
    Get reference and predicted volumes per atom.

    Returns
    -------
    dict[str, list[float]]
        Reference and model volumes in Angstrom^3/atom.
    """
    return scalar_property_values("volume_peratom")


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_constants.json",
    title="Lattice constants",
    x_label="Predicted lattice constant / Angstrom",
    y_label="DFT lattice constant / Angstrom",
    hoverdata={"Structure axis": LATTICE_COMPONENT_LABELS},
)
def lattice_constants() -> dict[str, list[float]]:
    """
    Get reference and predicted lattice constants.

    Returns
    -------
    dict[str, list[float]]
        Reference and model lattice constants in Angstrom.
    """
    return multi_property_values(LATTICE_PROPERTIES)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_beta_angle.json",
    title="Beta angles",
    x_label="Predicted beta angle / degrees",
    y_label="DFT beta angle / degrees",
    hoverdata={"OQMD ID": BETA_STRUCTURE_IDS},
)
def beta_angles() -> dict[str, list[float]]:
    """
    Get reference and predicted beta angles.

    Returns
    -------
    dict[str, list[float]]
        Reference and model beta angles in degrees.
    """
    return scalar_property_values("angle_beta")


@pytest.fixture
def solute_formation_energy_errors(
    solute_formation_energies: dict[str, list[float]],
) -> dict[str, float]:
    """
    Get solute-referenced formation-energy mean absolute errors.

    Parameters
    ----------
    solute_formation_energies
        Reference and predicted solute-referenced formation energies.

    Returns
    -------
    dict[str, float]
        MAE by model.
    """
    return {
        model_name: mae(solute_formation_energies["ref"], values)
        for model_name, values in solute_formation_energies.items()
        if model_name != "ref"
    }


@pytest.fixture
def volume_errors(volumes_per_atom: dict[str, list[float]]) -> dict[str, float]:
    """
    Get volume-per-atom mean absolute errors.

    Parameters
    ----------
    volumes_per_atom
        Reference and predicted volumes.

    Returns
    -------
    dict[str, float]
        MAE by model.
    """
    return {
        model_name: mae(volumes_per_atom["ref"], values)
        for model_name, values in volumes_per_atom.items()
        if model_name != "ref"
    }


@pytest.fixture
def lattice_constant_errors(
    lattice_constants: dict[str, list[float]],
) -> dict[str, float]:
    """
    Get lattice-constant mean absolute errors.

    Parameters
    ----------
    lattice_constants
        Reference and predicted lattice constants.

    Returns
    -------
    dict[str, float]
        MAE by model.
    """
    return {
        model_name: mae(lattice_constants["ref"], values)
        for model_name, values in lattice_constants.items()
        if model_name != "ref"
    }


@pytest.fixture
def beta_angle_errors(beta_angles: dict[str, list[float]]) -> dict[str, float]:
    """
    Get beta-angle mean absolute errors.

    Parameters
    ----------
    beta_angles
        Reference and predicted beta angles.

    Returns
    -------
    dict[str, float]
        MAE by model.
    """
    return {
        model_name: mae(beta_angles["ref"], values)
        for model_name, values in beta_angles.items()
        if model_name != "ref"
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "AlZnCuMg_regression_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    solute_formation_energy_errors: dict[str, float],
    volume_errors: dict[str, float],
    lattice_constant_errors: dict[str, float],
    beta_angle_errors: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Get all first-slice benchmark metrics.

    Parameters
    ----------
    solute_formation_energy_errors
        Solute-referenced formation-energy MAEs.
    volume_errors
        Volume-per-atom MAEs.
    lattice_constant_errors
        Lattice-constant MAEs.
    beta_angle_errors
        Beta-angle MAEs.

    Returns
    -------
    dict[str, dict[str, float]]
        Metrics by model.
    """
    copy_structures_to_app_data()
    write_pure_element_plot(OUT_PATH / "figure_pure_element_properties.json")
    results = {
        "Solute-Referenced Formation Energy MAE": solute_formation_energy_errors,
        "Volume MAE": volume_errors,
        "Lattice Constant MAE": lattice_constant_errors,
        "Beta Angle MAE": beta_angle_errors,
    }
    results.update(solute_solute_metrics())
    results.update(elastic_metrics())
    results.update(fault_surface_metrics())
    return results


def copy_structures_to_app_data() -> None:
    """Copy calculated structures into the app data directory."""
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue
        output_dir = OUT_PATH / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for structure_path in sorted(model_dir.glob("OQMD_*.xyz")):
            try:
                atoms = read(structure_path)
                write(output_dir / structure_path.name, atoms)
            except Exception as exc:
                warn(
                    f"Unable to copy structure {structure_path}: {exc}",
                    stacklevel=2,
                )


def test_alzncumg_regression(metrics: dict[str, dict[str, float]]) -> None:
    """
    Run Al-Cu-Mg-Zn metallurgy regression analysis.

    Parameters
    ----------
    metrics
        First-slice analysis metrics.
    """
    write_benchmark_info()
    return
