"""Analyse the Alexandria convex-hull benchmark."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from ase import Atoms
from ase.io import read
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from ml_peg.calcs.bulk_crystal.convex_hull_alexandria.prepare_inputs import STABLE_TOL
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "convex_hull_alexandria" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "convex_hull_alexandria"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Signed errors shown as violins: quantity, file stem and initial range (meV/atom),
# as in Figure 5 of Yu et al., Mater. Genome Eng. Adv. 2, e58 (2024). The range
# shows the tails: it holds 99% of the errors of each of the first models run
VIOLINS = {
    "e_form": ("Formation energy", "figure_e_form_violin", 500),
    "e_hull": ("Energy above hull", "figure_e_hull_violin", 500),
}

PERIODIC_TABLES = {
    "Energy above hull MAE": "e_hull_periodic_table",
    "Formation energy MAE": "e_form_periodic_table",
}
# Colour range of the periodic tables (meV/atom), wider than the metric thresholds so
# that the worst elements stay distinguishable (Figure 4 of Yu et al. uses 0-400)
PERIODIC_TABLE_RANGE = (0, 100)

# Colour scale limit for the energy above the hull of unstable phases (meV/atom)
E_HULL_COLOUR_MAX = 200


def load_outputs(model_name: str) -> list[Atoms] | None:
    """
    Get the structures and energies computed by a model.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    list[Atoms] | None
        Structures, with ``mat_id``, the reference ``dft_energy`` and
        ``e_above_hull``, and ``mlip_energy`` (``nan`` where the calculation
        failed) in ``info``, or ``None`` if the model was not run.
    """
    path = CALC_PATH / model_name / "structures.extxyz"
    if not path.exists():
        return None
    return read(path, index=":")


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


def reference_hull(structures: list[Atoms]) -> dict[str, dict[str, float | str]]:
    """
    Get the formation energy and energy above the PBE hull of each binary.

    Parameters
    ----------
    structures
        Unary ground states and binaries, with the reference ``dft_energy``.

    Returns
    -------
    dict[str, dict[str, float | str]]
        For each binary ``mat_id``: its ``system``, ``e_form`` and ``e_hull``
        (eV/atom).
    """
    return hull_energies(
        structures, {a.info["mat_id"]: a.info["dft_energy"] for a in structures}
    )


def compare_model(
    structures: list[Atoms],
    reference: dict[str, dict[str, float | str]],
    energies: dict[str, float],
) -> dict[str, dict[str, float | str | None]]:
    """
    Compare the hull of a model with the reference hull, for each binary.

    Parameters
    ----------
    structures
        Unary ground states and binaries.
    reference
        Reference hull energies of each binary, from ``reference_hull``.
    energies
        Total energy (eV) computed by the model for each ``mat_id``, missing or
        ``nan`` where the calculation failed.

    Returns
    -------
    dict[str, dict[str, float | str | None]]
        For each binary ``mat_id``: its ``system``, and the reference and model
        ``e_form`` and ``e_hull`` (eV/atom), the model values ``None`` if a
        calculation in its system failed.
    """
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
def model_outputs() -> tuple[list[Atoms], dict[str, dict[str, float] | None]]:
    """
    Get the reference structures and the energies of all models from their outputs.

    Returns
    -------
    tuple[list[Atoms], dict[str, dict[str, float] | None]]
        Unary ground states and binaries, with the reference data in ``info``, and
        the total energy (eV) for each ``mat_id`` of each model, or ``None`` for
        models that were not run.
    """
    structures = {}
    energies = {}
    for model_name in MODELS:
        outputs = load_outputs(model_name)
        if outputs is None:
            energies[model_name] = None
            continue
        energies[model_name] = {}
        for atoms in outputs:
            mat_id = atoms.info["mat_id"]
            energies[model_name][mat_id] = float(atoms.info["mlip_energy"])
            structures.setdefault(mat_id, atoms)
    return list(structures.values()), energies


@pytest.fixture(scope="module")
def structures(
    model_outputs: tuple[list[Atoms], dict[str, dict[str, float] | None]],
) -> list[Atoms]:
    """
    Get the reference structures and PBE energies.

    Parameters
    ----------
    model_outputs
        Reference structures, and energies of all models.

    Returns
    -------
    list[Atoms]
        Unary ground states and binaries.
    """
    return model_outputs[0]


@pytest.fixture(scope="module")
def comparisons(
    model_outputs: tuple[list[Atoms], dict[str, dict[str, float] | None]],
) -> dict[str, dict[str, dict] | None]:
    """
    Compare the hulls of all models with the reference.

    Parameters
    ----------
    model_outputs
        Reference structures, and energies of all models.

    Returns
    -------
    dict[str, dict[str, dict] | None]
        Comparison for each binary and model, or ``None`` for models that were not
        run.
    """
    structures, energies = model_outputs
    reference = reference_hull(structures)
    return {
        model_name: (
            None
            if model_energies is None
            else compare_model(structures, reference, model_energies)
        )
        for model_name, model_energies in energies.items()
    }


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


def system_maes(comparison: dict[str, dict]) -> dict[str, float]:
    """
    Get the energy above hull MAE of each system, over its binaries.

    Parameters
    ----------
    comparison
        Comparison for each binary of one model.

    Returns
    -------
    dict[str, float]
        MAE (meV/atom) of each system that did not fail, from the largest to the
        smallest.
    """
    errors = defaultdict(list)
    for result in comparison.values():
        if result["e_hull_pred"] is not None:
            errors[result["system"]].append(
                abs(result["e_hull_pred"] - result["e_hull_ref"])
            )
    maes = {system: 1000 * float(np.mean(e)) for system, e in errors.items()}
    return dict(sorted(maes.items(), key=lambda item: -item[1]))


def hull_figure(data: dict[str, list], system: str, model: str) -> go.Figure:
    """
    Draw the reference and model hulls of a system side by side.

    Parameters
    ----------
    data
        Data of the system, one list per field.
    system
        System, such as ``Cu-Zn``.
    model
        Name of the model.

    Returns
    -------
    go.Figure
        Formation energy against composition, with the hull, for PBE and the model.
    """
    first, second = system.split("-")
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=("PBE (Alexandria)", model),
    )
    x = np.array(data["x"])
    hover = [
        f"{formula} ({mat_id})"
        for formula, mat_id in zip(data["formula"], data["mat_id"], strict=True)
    ]
    for col, source in ((1, "ref"), (2, "pred")):
        e_form = np.array(data[f"e_form_{source}"])
        e_hull = np.array(data[f"e_hull_{source}"])
        stable = np.array(data[f"stable_{source}"])
        order = np.argsort(x[stable])
        fig.add_scatter(
            x=x[stable][order],
            y=e_form[stable][order],
            mode="lines",
            line={"color": "black"},
            hoverinfo="skip",
            showlegend=False,
            row=1,
            col=col,
        )
        texts = [
            f"{text}<br>E_form = {ef:.0f} meV/atom<br>E_hull = {eh:.0f} meV/atom"
            for text, ef, eh in zip(hover, e_form, e_hull, strict=True)
        ]
        fig.add_scatter(
            x=x[~stable],
            y=e_form[~stable],
            mode="markers",
            marker={
                "color": e_hull[~stable],
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": E_HULL_COLOUR_MAX,
                "size": 8,
                "colorbar": {"title": "E_hull (meV/atom)"} if col == 1 else None,
                "showscale": col == 1,
            },
            text=np.array(texts)[~stable],
            hoverinfo="text",
            name="Unstable",
            showlegend=col == 1,
            row=1,
            col=col,
        )
        fig.add_scatter(
            x=x[stable],
            y=e_form[stable],
            mode="markers",
            marker={"color": "black", "size": 10},
            text=np.array(texts)[stable],
            hoverinfo="text",
            name="Stable",
            showlegend=col == 1,
            row=1,
            col=col,
        )
    # Show the phases that are stable in PBE on the model hull too
    stable_ref = np.array(data["stable_ref"])
    fig.add_scatter(
        x=x[stable_ref],
        y=np.array(data["e_form_pred"])[stable_ref],
        mode="markers",
        marker={
            "color": "rgba(0,0,0,0)",
            "size": 16,
            "line": {"color": "#d62728", "width": 2},
        },
        hoverinfo="skip",
        name="Stable in PBE",
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"x in {first}(1-x){second}x", range=[-0.03, 1.03])
    fig.update_yaxes(title_text="Formation energy (meV/atom)", col=1)
    fig.update_layout(
        title=f"{system}: PBE and {model} convex hulls",
        height=500,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


@pytest.fixture
def hull_figures(
    structures: list[Atoms], comparisons: dict[str, dict[str, dict] | None]
) -> None:
    """
    Write the PBE and model hulls of each system, and the MAE of each system.

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
        out_dir = OUT_PATH / model_name / "hulls"
        out_dir.mkdir(parents=True, exist_ok=True)
        for system, rows in hull_rows(structures, comparison).items():
            data = {key: [row[key] for row in rows] for key in rows[0]}
            fig = hull_figure(data, system, model_name)
            # Leave out the default template, restored when the app loads the figure,
            # to keep the files small
            fig.layout.template = None
            fig.write_json(out_dir / f"{system}.json")
        with open(
            OUT_PATH / model_name / "hull_systems.json", "w", encoding="utf8"
        ) as f:
            json.dump(system_maes(comparison), f, separators=(",", ":"))


def test_convex_hull_alexandria(
    metrics: dict[str, dict],
    periodic_tables: None,
    e_hull_density: dict[str, dict],
    e_form_density: dict[str, dict],
    violins: None,
    hull_figures: None,
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
    hull_figures
        PBE and model hulls of each system (side effect only).
    """
    get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*.extxyz",
        info_keys=["mat_id"],
        write_structs=False,
        out_path=OUT_PATH,
    )
