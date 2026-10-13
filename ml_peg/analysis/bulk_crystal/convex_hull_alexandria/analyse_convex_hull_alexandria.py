"""Analyse the Alexandria convex-hull benchmark."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from ase import Atoms
from ase.io import read
import numpy as np
import plotly.graph_objects as go
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
import pytest

from ml_peg.analysis.utils.decorators import (
    build_table,
    plot_density_scatter,
    plot_periodic_table,
)
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "convex_hull_alexandria" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "convex_hull_alexandria"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Entries within this energy of the hull (eV/atom) are classified as stable
STABLE_TOL = 1e-6

# Signed errors shown as violins: quantity, file stem and initial range (meV/atom),
# as in Figure 5 of Yu et al., Mater. Genome Eng. Adv. 2, e58 (2024)
VIOLINS = {
    "e_form": ("Formation energy", "figure_e_form_violin", 200),
    "e_hull": ("Energy above hull", "figure_e_hull_violin", 200),
}

PERIODIC_TABLES = {
    "Energy above hull MAE": "e_hull_periodic_table",
    "Formation energy MAE": "e_form_periodic_table",
}
# Colour range of the periodic tables (meV/atom), wider than the metric thresholds so
# that the worst elements stay distinguishable (Figure 4 of Yu et al. uses 0-400)
PERIODIC_TABLE_RANGE = (0, 100)


def load_reference() -> list[Atoms]:
    """
    Get the reference structures and PBE energies.

    Returns
    -------
    list[Atoms]
        Structures, with ``mat_id``, ``dft_energy`` and ``e_above_hull`` in ``info``.
    """
    data_path = (
        download_s3_data(
            key="inputs/bulk_crystal/convex_hull_alexandria/convex_hull_alexandria.zip",
            filename="convex_hull_alexandria.zip",
        )
        / "convex_hull_alexandria"
    )
    return read(data_path / "structures.extxyz", index=":")


def load_energies(model_name: str) -> dict[str, float] | None:
    """
    Get the energies computed by a model.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    dict[str, float] | None
        Total energy (eV) for each ``mat_id``, ``nan`` where the calculation
        failed, or ``None`` if the model was not run.
    """
    path = CALC_PATH / model_name / "structures.extxyz"
    if not path.exists():
        return None
    return {
        atoms.info["mat_id"]: float(atoms.info["mlip_energy"])
        for atoms in read(path, index=":")
    }


def hull_energies(
    structures: list[Atoms], energies: dict[str, float]
) -> dict[str, dict[str, float | str | None]]:
    """
    Get the formation energy and energy above the hull of each binary.

    The hull of each binary system is built from the energies of its binaries and of
    the elemental ground states in ``structures``. A system in which any calculation
    failed is not analysed.

    Parameters
    ----------
    structures
        Unary ground states and binaries.
    energies
        Total energy (eV) for each ``mat_id``, ``nan`` where the calculation failed.

    Returns
    -------
    dict[str, dict[str, float | str | None]]
        For each binary ``mat_id``: its ``system``, ``e_form`` and ``e_hull``
        (eV/atom), ``None`` if a calculation in its system failed.
    """
    elements = {}
    systems = defaultdict(list)
    for atoms in structures:
        mat_id = atoms.info["mat_id"]
        entry = PDEntry(
            Composition(atoms.get_chemical_formula()),
            energies.get(mat_id, np.nan),
            name=mat_id,
        )
        symbols = sorted(set(atoms.get_chemical_symbols()))
        if len(symbols) == 1:
            elements[symbols[0]] = entry
        else:
            systems["-".join(symbols)].append(entry)

    results = {}
    for system, entries in systems.items():
        unaries = [elements[symbol] for symbol in system.split("-")]
        if not np.all(np.isfinite([e.energy for e in entries + unaries])):
            for entry in entries:
                results[entry.name] = {"system": system, "e_form": None, "e_hull": None}
            continue
        hull = PhaseDiagram(entries + unaries)
        for entry in entries:
            results[entry.name] = {
                "system": system,
                "e_form": hull.get_form_energy_per_atom(entry),
                "e_hull": hull.get_e_above_hull(entry),
            }
    return results


def compare_model(
    structures: list[Atoms], energies: dict[str, float]
) -> dict[str, dict[str, float | str | None]]:
    """
    Compare the hull of a model with the reference hull, for each binary.

    Parameters
    ----------
    structures
        Unary ground states and binaries, with the reference ``dft_energy``.
    energies
        Total energy (eV) computed by the model for each ``mat_id``.

    Returns
    -------
    dict[str, dict[str, float | str | None]]
        For each binary ``mat_id``: its ``system``, and the reference and model
        ``e_form`` and ``e_hull`` (eV/atom), the model values ``None`` if a
        calculation in its system failed.
    """
    reference = hull_energies(
        structures, {a.info["mat_id"]: a.info["dft_energy"] for a in structures}
    )
    predicted = hull_energies(structures, energies)
    return {
        mat_id: {
            "system": ref["system"],
            "e_form_ref": ref["e_form"],
            "e_form_pred": predicted[mat_id]["e_form"],
            "e_hull_ref": ref["e_hull"],
            "e_hull_pred": predicted[mat_id]["e_hull"],
        }
        for mat_id, ref in reference.items()
    }


def classification_f1(e_hull_ref: np.ndarray, e_hull_pred: np.ndarray) -> float:
    """
    Get the F1 score of the classification of binaries as stable.

    Parameters
    ----------
    e_hull_ref
        Reference energies above the hull (eV/atom).
    e_hull_pred
        Predicted energies above the hull (eV/atom).

    Returns
    -------
    float
        F1 score, with stable (on the hull) as the positive class.
    """
    stable_ref = e_hull_ref < STABLE_TOL
    stable_pred = e_hull_pred < STABLE_TOL
    true_positives = np.sum(stable_ref & stable_pred)
    if true_positives == 0:
        return 0.0
    precision = true_positives / np.sum(stable_pred)
    recall = true_positives / np.sum(stable_ref)
    return float(2 * precision * recall / (precision + recall))


def model_metrics(comparison: dict[str, dict] | None) -> dict[str, float | None]:
    """
    Get the metrics of one model.

    Parameters
    ----------
    comparison
        Comparison for each binary of one model.

    Returns
    -------
    dict[str, float | None]
        Metric values, ``None`` if the model was not run.
    """
    names = (
        "Energy above hull MAE",
        "Stability F1",
        "Failed systems",
        "Formation energy MAE",
        "Formation energy RMSE",
        "Formation energy R²",
    )
    if comparison is None:
        return dict.fromkeys(names)

    systems = {result["system"] for result in comparison.values()}
    failed_systems = {
        result["system"]
        for result in comparison.values()
        if result["e_hull_pred"] is None
    }
    failed = 100 * len(failed_systems) / len(systems)

    results = [r for r in comparison.values() if r["e_hull_pred"] is not None]
    if not results:
        return dict.fromkeys(names) | {"Failed systems": failed}
    values = {
        key: np.array([result[key] for result in results])
        for key in ("e_form_ref", "e_form_pred", "e_hull_ref", "e_hull_pred")
    }
    e_form_error = values["e_form_pred"] - values["e_form_ref"]
    e_form_spread = np.sum((values["e_form_ref"] - values["e_form_ref"].mean()) ** 2)
    return {
        "Energy above hull MAE": 1000
        * float(np.mean(np.abs(values["e_hull_pred"] - values["e_hull_ref"]))),
        "Stability F1": classification_f1(values["e_hull_ref"], values["e_hull_pred"]),
        "Failed systems": failed,
        "Formation energy MAE": 1000 * float(np.mean(np.abs(e_form_error))),
        "Formation energy RMSE": 1000 * float(np.sqrt(np.mean(e_form_error**2))),
        "Formation energy R²": float(1 - np.sum(e_form_error**2) / e_form_spread),
    }


@pytest.fixture(scope="module")
def structures() -> list[Atoms]:
    """
    Get the reference structures and PBE energies.

    Returns
    -------
    list[Atoms]
        Unary ground states and binaries.
    """
    return load_reference()


@pytest.fixture(scope="module")
def comparisons(structures: list[Atoms]) -> dict[str, dict[str, dict] | None]:
    """
    Compare the hulls of all models with the reference.

    Parameters
    ----------
    structures
        Unary ground states and binaries.

    Returns
    -------
    dict[str, dict[str, dict] | None]
        Comparison for each binary and model, or ``None`` for models that were not
        run.
    """
    comparisons = {}
    for model_name in MODELS:
        energies = load_energies(model_name)
        comparisons[model_name] = (
            None if energies is None else compare_model(structures, energies)
        )
    return comparisons


@pytest.fixture
@build_table(
    filename=OUT_PATH / "convex_hull_alexandria_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(comparisons: dict[str, dict[str, dict] | None]) -> dict[str, dict]:
    """
    Get all convex-hull metrics.

    Parameters
    ----------
    comparisons
        Comparison for each binary and model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    table = defaultdict(dict)
    for model_name, comparison in comparisons.items():
        for name, value in model_metrics(comparison).items():
            table[name][model_name] = value
    return dict(table)


def element_values(comparison: dict[str, dict], key: str) -> dict[str, float]:
    """
    Get the MAE of a quantity over the binaries containing each element.

    Parameters
    ----------
    comparison
        Comparison for each binary of one model.
    key
        Quantity, ``e_hull`` or ``e_form``.

    Returns
    -------
    dict[str, float]
        MAE (meV/atom) for each element.
    """
    errors = defaultdict(list)
    for result in comparison.values():
        if result["e_hull_pred"] is None:
            continue
        for element in result["system"].split("-"):
            errors[element].append(abs(result[f"{key}_pred"] - result[f"{key}_ref"]))
    return {element: 1000 * float(np.mean(e)) for element, e in errors.items()}


@pytest.fixture
def periodic_tables(comparisons: dict[str, dict[str, dict] | None]) -> None:
    """
    Write per-model periodic-table heatmaps of the MAEs, for each element.

    Parameters
    ----------
    comparisons
        Comparison for each binary and model.
    """
    for model_name, comparison in comparisons.items():
        if comparison is None:
            continue
        counts = defaultdict(int)
        for result in comparison.values():
            if result["e_hull_pred"] is not None:
                for element in result["system"].split("-"):
                    counts[element] += 1
        hoverdata = {"Binaries": dict(counts)}
        for metric, file_stem in PERIODIC_TABLES.items():
            key = "e_hull" if metric.startswith("Energy above hull") else "e_form"
            plot_periodic_table(
                title=f"{metric} (meV/atom) - {model_name}",
                colorbar_title=f"{metric} (meV/atom)",
                hoverdata=hoverdata,
                filename=str(OUT_PATH / model_name / f"{file_stem}.json"),
                zmin=PERIODIC_TABLE_RANGE[0],
                zmax=PERIODIC_TABLE_RANGE[1],
                colorscale="Viridis_r",
                value_format=".0f",
            )(element_values)(comparison, key)


def density_inputs(
    comparisons: dict[str, dict[str, dict] | None], key: str
) -> dict[str, dict]:
    """
    Get the reference and predicted values of a quantity for the density plots.

    Parameters
    ----------
    comparisons
        Comparison for each binary and model.
    key
        Quantity, ``e_hull`` or ``e_form``.

    Returns
    -------
    dict[str, dict]
        Reference and predicted values (meV/atom) and MAE for each model.
    """
    inputs = {}
    for model_name, comparison in comparisons.items():
        if comparison is None:
            inputs[model_name] = {}
            continue
        results = [r for r in comparison.values() if r["e_hull_pred"] is not None]
        ref = np.array([1000 * result[f"{key}_ref"] for result in results])
        pred = np.array([1000 * result[f"{key}_pred"] for result in results])
        inputs[model_name] = {
            "ref": ref.tolist(),
            "pred": pred.tolist(),
            "metric": float(np.mean(np.abs(pred - ref))) if len(ref) else None,
            "meta": {},
        }
    return inputs


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_e_hull_density.json",
    title="Energy above the hull",
    x_label="PBE energy above hull (meV/atom)",
    y_label="Predicted energy above hull (meV/atom)",
)
def e_hull_density(comparisons: dict[str, dict[str, dict] | None]) -> dict[str, dict]:
    """
    Get the density-scatter inputs for the energy above the hull.

    Parameters
    ----------
    comparisons
        Comparison for each binary and model.

    Returns
    -------
    dict[str, dict]
        Density-scatter data for each model.
    """
    return density_inputs(comparisons, "e_hull")


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_e_form_density.json",
    title="Formation energy",
    x_label="PBE formation energy (meV/atom)",
    y_label="Predicted formation energy (meV/atom)",
)
def e_form_density(comparisons: dict[str, dict[str, dict] | None]) -> dict[str, dict]:
    """
    Get the density-scatter inputs for the formation energy.

    Parameters
    ----------
    comparisons
        Comparison for each binary and model.

    Returns
    -------
    dict[str, dict]
        Density-scatter data for each model.
    """
    return density_inputs(comparisons, "e_form")


def signed_errors(comparison: dict[str, dict], key: str) -> list[float]:
    """
    Get the signed errors of a quantity, over the binaries of systems that did not fail.

    Parameters
    ----------
    comparison
        Comparison for each binary of one model.
    key
        Quantity, ``e_hull`` or ``e_form``.

    Returns
    -------
    list[float]
        Predicted minus reference values (meV/atom).
    """
    return [
        1000 * (result[f"{key}_pred"] - result[f"{key}_ref"])
        for result in comparison.values()
        if result[f"{key}_pred"] is not None
    ]


@pytest.fixture
def violins(comparisons: dict[str, dict[str, dict] | None]) -> None:
    """
    Write violin plots of the signed errors of all models.

    Parameters
    ----------
    comparisons
        Comparison for each binary and model.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    for key, (name, file_stem, limit) in VIOLINS.items():
        fig = go.Figure()
        for model_name, comparison in comparisons.items():
            if comparison is None:
                continue
            fig.add_trace(
                go.Violin(
                    y=signed_errors(comparison, key),
                    name=model_name,
                    points="outliers",
                    marker={"size": 2},
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False,
                )
            )
        fig.add_hline(y=0, line={"color": "grey", "width": 1})
        fig.update_layout(
            title=f"{name} error distribution",
            yaxis={
                "title": {"text": f"{name} difference (meV/atom)"},
                "range": [-limit, limit],
            },
        )
        fig.write_json(OUT_PATH / f"{file_stem}.json")


def hull_rows(
    structures: list[Atoms], comparison: dict[str, dict]
) -> dict[str, list[dict]]:
    """
    Get the data needed to draw the reference and model hulls of each system.

    Parameters
    ----------
    structures
        Unary ground states and binaries.
    comparison
        Comparison for each binary of one model.

    Returns
    -------
    dict[str, list[dict]]
        For each system that did not fail, its elements and binaries, with
        ``mat_id``, ``formula``, ``x`` (fraction of the second element of the
        system), the reference and model ``e_form`` and ``e_hull`` (meV/atom,
        rounded to 0.1), and whether it is on the reference and model hulls.
    """
    atoms_by_id = {atoms.info["mat_id"]: atoms for atoms in structures}
    rows = defaultdict(list)
    for mat_id, result in comparison.items():
        if result["e_hull_pred"] is None:
            continue
        symbols = atoms_by_id[mat_id].get_chemical_symbols()
        rows[result["system"]].append(
            {
                "mat_id": mat_id,
                "formula": atoms_by_id[mat_id].get_chemical_formula(mode="reduce"),
                "x": round(
                    symbols.count(result["system"].split("-")[1]) / len(symbols), 4
                ),
            }
            | {
                key: round(1000 * result[key], 1)
                for key in ("e_form_ref", "e_form_pred", "e_hull_ref", "e_hull_pred")
            }
            | {
                "stable_ref": bool(result["e_hull_ref"] < STABLE_TOL),
                "stable_pred": bool(result["e_hull_pred"] < STABLE_TOL),
            }
        )
    zero = dict.fromkeys(("e_form_ref", "e_form_pred", "e_hull_ref", "e_hull_pred"), 0)
    zero |= {"stable_ref": True, "stable_pred": True}
    for system, system_rows in rows.items():
        for x, element in enumerate(system.split("-")):
            system_rows.append({"mat_id": element, "formula": element, "x": x} | zero)
    return dict(rows)


@pytest.fixture
def hull_data(
    structures: list[Atoms], comparisons: dict[str, dict[str, dict] | None]
) -> None:
    """
    Write the data needed to draw the reference and model hulls, for each model.

    Parameters
    ----------
    structures
        Unary ground states and binaries.
    comparisons
        Comparison for each binary and model.
    """
    for model_name, comparison in comparisons.items():
        if comparison is None:
            continue
        out_dir = OUT_PATH / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # One list per field for each system, to keep the file small
        columns = {
            system: {key: [row[key] for row in rows] for key in rows[0]}
            for system, rows in hull_rows(structures, comparison).items()
        }
        with open(out_dir / "hull_data.json", "w", encoding="utf8") as file:
            json.dump(columns, file, separators=(",", ":"))


def test_convex_hull_alexandria(
    metrics: dict[str, dict],
    periodic_tables: None,
    e_hull_density: dict[str, dict],
    e_form_density: dict[str, dict],
    violins: None,
    hull_data: None,
) -> None:
    """
    Run the Alexandria convex-hull analysis.

    Parameters
    ----------
    metrics
        All metric names and values for all models.
    periodic_tables
        Per-model periodic-table heatmaps (side effect only).
    e_hull_density
        Density scatter of the energy above the hull (side effect only).
    e_form_density
        Density scatter of the formation energy (side effect only).
    violins
        Violin plots of the signed errors (side effect only).
    hull_data
        Data to draw the reference and model hulls (side effect only).
    """
    get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*.extxyz",
        info_keys=["mat_id"],
        write_structs=False,
        out_path=OUT_PATH,
    )
