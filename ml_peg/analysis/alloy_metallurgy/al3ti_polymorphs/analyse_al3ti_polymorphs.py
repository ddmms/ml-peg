"""Analyse the Al3Ti polymorph benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
from plotly.colors import qualitative
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "alloy_metallurgy" / "al3ti_polymorphs" / "outputs"
OUT_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "al3ti_polymorphs"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

REQUIRED_ELEMENTS = ("Al", "Ti")
AL_REFERENCE = "Al_fcc"
TI_REFERENCE = "Ti_hcp"
POLYMORPHS = ("D022", "D023", "L12")
PLOT_EQUAL_LENGTH_TOLERANCE_ANGSTROM = 1e-3


def write_benchmark_info() -> None:
    """Write benchmark-wide element requirements for application filtering."""
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(
        json.dumps({"elements": REQUIRED_ELEMENTS}, indent=1),
        encoding="utf8",
    )


def load_references() -> dict[str, Any]:
    """
    Load references for all polymorph computations.

    Returns
    -------
    dict[str, Any]
        Reference data keyed by polymorph name.
    """
    ref_path = CALC_PATH / "references" / "DFT.json"
    try:
        with ref_path.open(encoding="utf8") as file:
            references = json.load(file)
        if not isinstance(references, dict):
            raise TypeError("expected a JSON object")
        return references
    except Exception as exc:
        warn(
            f"Unable to load reference data from {ref_path}: {exc}",
            stacklevel=2,
        )
        return {}


@pytest.fixture
def reference_data() -> dict[str, Any]:
    """
    Load DFT references once for the analysis fixtures.

    Returns
    -------
    dict[str, Any]
        Reference data keyed by polymorph name.
    """
    return load_references()


def load_structure(structure: str, data_path: Path) -> Atoms | None:
    """
    Load a structure from a xyz file.

    Parameters
    ----------
    structure
        Name of the structure to be loaded.
    data_path
        Path to the root of the data directory.

    Returns
    -------
    Atoms
        Loaded structure.
    None
        If loading failed.
    """
    structure_path = data_path / f"{structure}.xyz"
    try:
        return read(structure_path)
    except Exception as exc:
        warn(f"Error loading structure {structure}: {exc}", stacklevel=2)
        return None


def energy_per_atom(atoms: Atoms, model_name: str, structure: str) -> float:
    """
    Get energy per atom, returning NaN if evaluation fails.

    Parameters
    ----------
    atoms
        Structure with a stored calculator result.
    model_name
        Model that produced the structure.
    structure
        Name of the structure.

    Returns
    -------
    float
        Potential energy per atom, or NaN on failure.
    """
    try:
        energy = atoms.get_potential_energy() / len(atoms)
        if not np.isfinite(energy):
            raise ValueError("non-finite energy")
        return float(energy)
    except Exception as exc:
        warn(
            f"Error getting energy for model {model_name}, "
            f"structure {structure}: {exc}",
            stacklevel=2,
        )
        return np.nan


@pytest.fixture
def relaxed_polymorphs() -> dict[str, list[Atoms | None]]:
    """
    Load each model's relaxed polymorphs in ``POLYMORPHS`` order.

    Returns
    -------
    dict[str, list[Atoms | None]]
        Relaxed structures by model, with missing structures represented by None.
    """
    structures = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        structures[model_name] = (
            [load_structure(name, model_dir) for name in POLYMORPHS]
            if model_dir.exists()
            else [None] * len(POLYMORPHS)
        )
    return structures


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_formation_energies.json",
    title="Al3Ti Formation Energies",
    x_label="Predicted formation energy / eV/atom",
    y_label="Reference formation energy / eV/atom",
    hoverdata={"Polymorph": POLYMORPHS},
)
def formation_energies(
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, list[float]]:
    """
    Get formation energies for all Al3Ti polymorphs.

    Parameters
    ----------
    reference_data
        DFT reference values for the three polymorphs.
    relaxed_polymorphs
        Relaxed structures by model, in polymorph order.

    Returns
    -------
    dict[str, list[float]]
        Dictionary of reference and predicted formation energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        # Initialize nan results
        results[model_name] = [np.nan] * len(POLYMORPHS)

        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        al_struct = load_structure(AL_REFERENCE, model_dir)
        ti_struct = load_structure(TI_REFERENCE, model_dir)
        if al_struct is None or ti_struct is None:
            # Formation energies need both elemental references.
            continue
        al_energy = energy_per_atom(al_struct, model_name, AL_REFERENCE)
        ti_energy = energy_per_atom(ti_struct, model_name, TI_REFERENCE)
        if not np.isfinite(al_energy) or not np.isfinite(ti_energy):
            continue

        predictions = []
        for name, poly_struct in zip(
            POLYMORPHS, relaxed_polymorphs[model_name], strict=True
        ):
            if poly_struct is None:
                predictions.append(np.nan)
                continue
            poly_energy = energy_per_atom(poly_struct, model_name, name)
            formation_energy = poly_energy - 0.75 * al_energy - 0.25 * ti_energy
            predictions.append(formation_energy)
        results[model_name] = predictions

    # Store reference energies
    missing = [name for name in POLYMORPHS if name not in reference_data]
    if missing:
        raise ValueError(f"Missing DFT references for: {', '.join(missing)}")
    ref = [
        float(reference_data[name]["formation_energy_0k_ev_atom"]["value"])
        for name in POLYMORPHS
    ]
    results["ref"] = ref

    return results


def pairwise_order_accuracy(reference: list[float], prediction: list[float]) -> float:
    """
    Return the percentage of polymorph pairs ordered as in the reference.

    Parameters
    ----------
    reference
        Reference formation energies, in the order given by ``POLYMORPHS``.
    prediction
        Predicted formation energies, in the same order.

    Returns
    -------
    float
        Percentage of pairwise comparisons with matching energy order.
    """
    reference_values = np.asarray(reference, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if (
        reference_values.ndim != 1
        or prediction_values.ndim != 1
        or reference_values.size < 2
        or reference_values.size != prediction_values.size
        or not np.isfinite(reference_values).all()
        or not np.isfinite(prediction_values).all()
    ):
        return np.nan

    correct_pairs = 0
    total_pairs = 0
    for i in range(reference_values.size):
        for j in range(i + 1, reference_values.size):
            reference_order = np.sign(reference_values[i] - reference_values[j])
            prediction_order = np.sign(prediction_values[i] - prediction_values[j])
            correct_pairs += int(reference_order == prediction_order)
            total_pairs += 1

    return float(100.0 * correct_pairs / total_pairs)


def write_formation_energy_ordering_plots(
    formation_energies: dict[str, list[float]],
) -> None:
    """
    Save DFT and predicted energies in DFT stability order for each model.

    Parameters
    ----------
    formation_energies
        Reference and predicted formation energies in ``POLYMORPHS`` order.
    """
    reference = np.asarray(formation_energies["ref"], dtype=float)
    order = np.argsort(reference) if np.isfinite(reference).all() else np.arange(3)
    names = [POLYMORPHS[i] for i in order]

    for model_name in MODELS:
        prediction = np.asarray(
            formation_energies.get(model_name, [np.nan] * len(POLYMORPHS)),
            dtype=float,
        )
        accuracy = pairwise_order_accuracy(reference.tolist(), prediction.tolist())
        if np.isfinite(accuracy):
            agreements = [
                sum(
                    np.sign(reference[i] - reference[j])
                    == np.sign(prediction[i] - prediction[j])
                    for j in range(len(POLYMORPHS))
                    if j != i
                )
                for i in order
            ]
            colors = ["#D55E00", "#E69F00", "#009E73"]
            symbols = ["x", "diamond", "circle"]
            marker = {
                "color": [colors[count] for count in agreements],
                "symbol": [symbols[count] for count in agreements],
                "size": 13,
            }
            hover = [f"{count}/2 pairs agree" for count in agreements]
            score = f"{accuracy:.1f}% ({round(accuracy * 3 / 100)}/3 pairs)"
        else:
            marker = {"color": "#888888", "symbol": "circle", "size": 13}
            hover = ["Ordering unavailable"] * len(POLYMORPHS)
            score = "unavailable"

        figure = go.Figure()
        figure.add_scatter(
            x=names,
            y=reference[order],
            mode="lines+markers",
            name="DFT reference",
            line={"color": "#555555", "dash": "dash"},
            marker={"color": "#555555", "symbol": "diamond", "size": 11},
            hovertemplate="%{x}<br>DFT: %{y:.4f} eV/atom<extra></extra>",
        )
        figure.add_scatter(
            x=names,
            y=prediction[order],
            mode="lines+markers",
            name=model_name,
            line={"color": "#377EB8"},
            marker=marker,
            customdata=hover,
            hovertemplate=(
                "%{x}<br>Predicted: %{y:.4f} eV/atom<br>%{customdata}<extra></extra>"
            ),
        )
        for count, color, symbol in (
            (2, "#009E73", "circle"),
            (1, "#E69F00", "diamond"),
            (0, "#D55E00", "x"),
        ):
            figure.add_scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{count}/2 pairs agree",
                marker={"color": color, "symbol": symbol, "size": 11},
                hoverinfo="skip",
            )
        figure.update_layout(
            title=(
                f"{model_name}: Al3Ti formation energies<br>"
                f"<sup>Ordering accuracy: {score}</sup>"
            ),
            xaxis={"title": "Polymorphs in DFT stability order", "type": "category"},
            yaxis={"title": "Formation energy / eV/atom"},
        )
        output_path = OUT_PATH / model_name / "figure_formation_energy_ordering.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_json(output_path)


@pytest.fixture
def formation_energy_errors(formation_energies) -> dict[str, float]:
    """
    Get mean absolute error for formation energies.

    Parameters
    ----------
    formation_energies
        Dictionary of reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted formation energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if formation_energies.get(model_name):
            results[model_name] = mae(
                formation_energies["ref"], formation_energies[model_name]
            )
        else:
            results[model_name] = np.nan
    return results


@pytest.fixture
def polymorph_order_accuracies(formation_energies) -> dict[str, float]:
    """
    Calculate pairwise stability-order accuracy for each model.

    Parameters
    ----------
    formation_energies
        Dictionary of reference and predicted formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted order accuracies for all models.
    """
    reference = formation_energies["ref"]
    return {
        model_name: pairwise_order_accuracy(
            reference, formation_energies.get(model_name, [])
        )
        for model_name in MODELS
    }


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_volumes_per_atom.json",
    title="Al3Ti Volumes per Atom",
    x_label="Predicted volume / Å³/atom",
    y_label="Reference volume / Å³/atom",
    hoverdata={"Polymorph": POLYMORPHS},
)
def volumes_per_atom(
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, list[float]]:
    """
    Get DFT and relaxed volumes per atom in ``POLYMORPHS`` order.

    Parameters
    ----------
    reference_data
        DFT reference values for the three polymorphs.
    relaxed_polymorphs
        Relaxed structures by model, in polymorph order.

    Returns
    -------
    dict[str, list[float]]
        Dictionary of predicted and reference volumes per atom.
    """
    volumes = {
        "ref": [
            float(reference_data[name]["volume_per_atom_angstrom3"]["value"])
            for name in POLYMORPHS
        ]
    }
    for model_name, structures in relaxed_polymorphs.items():
        values = []
        for atoms in structures:
            if atoms is None or not len(atoms) or atoms.cell.rank != 3:
                values.append(np.nan)
                continue
            volume = atoms.get_volume() / len(atoms)
            values.append(volume if np.isfinite(volume) and volume > 0 else np.nan)
        volumes[model_name] = values
    return volumes


def lattice_parameters(
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, list[float]]:
    """
    Collect all three cell lengths for each polymorph for plotting.

    Parameters
    ----------
    reference_data
        DFT lattice parameters for the three polymorphs.
    relaxed_polymorphs
        Relaxed structures by model, in polymorph order.

    Returns
    -------
    dict[str, list[float]]
        Reference and predicted lengths in polymorph and axis order.
    """
    lengths = {"ref": []}
    for name in POLYMORPHS:
        a_ref = float(reference_data[name]["lattice_a_angstrom"]["value"])
        c_ref = (
            a_ref
            if name == "L12"
            else float(reference_data[name]["lattice_c_angstrom"]["value"])
        )
        lengths["ref"].extend([a_ref, a_ref, c_ref])

    for model_name, structures in relaxed_polymorphs.items():
        values = []
        for atoms in structures:
            values.extend(
                [np.nan] * 3
                if atoms is None or atoms.cell.rank != 3 or atoms.cell.volume <= 0
                else atoms.cell.lengths().tolist()
            )
        lengths[model_name] = values
    return lengths


def write_lattice_parameter_plot(lengths: dict[str, list[float]]) -> None:
    """
    Plot signed cell-length errors separately for each polymorph.

    Show symmetry-equivalent axes once when they agree within 0.001 Å.

    Parameters
    ----------
    lengths
        Reference and predicted lengths in polymorph and axis order.
    """
    reference = np.asarray(lengths["ref"]).reshape(len(POLYMORPHS), 3)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("D0₂₂", "D0₂₃", "L1₂"),
        shared_yaxes=True,
    )
    visible_axes = [set() for _ in POLYMORPHS]
    plotted_models = [
        name
        for name, values in lengths.items()
        if name != "ref" and np.isfinite(values).any()
    ]
    for model_index, model_name in enumerate(plotted_models):
        values = lengths[model_name]
        prediction = np.asarray(values).reshape(len(POLYMORPHS), 3)
        first_trace = True
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = 100 * (prediction / reference - 1)
        for polymorph_index, name in enumerate(POLYMORPHS):
            if not np.isfinite(prediction[polymorph_index]).any():
                continue
            axes = [0]
            if not np.isclose(
                prediction[polymorph_index, 1],
                prediction[polymorph_index, 0],
                rtol=0,
                atol=PLOT_EQUAL_LENGTH_TOLERANCE_ANGSTROM,
            ):
                axes.append(1)
            if name != "L12" or not np.isclose(
                prediction[polymorph_index, 2],
                prediction[polymorph_index, 0],
                rtol=0,
                atol=PLOT_EQUAL_LENGTH_TOLERANCE_ANGSTROM,
            ):
                axes.append(2)
            visible_axes[polymorph_index].update(axes)
            figure.add_scatter(
                x=["abc"[axis] for axis in axes],
                y=deviation[polymorph_index, axes].tolist(),
                customdata=np.column_stack(
                    (
                        prediction[polymorph_index, axes],
                        reference[polymorph_index, axes],
                        prediction[polymorph_index, axes]
                        - reference[polymorph_index, axes],
                    )
                ).tolist(),
                mode="markers",
                name=model_name,
                legendgroup=model_name,
                showlegend=first_trace,
                marker={
                    "color": qualitative.Plotly[model_index % len(qualitative.Plotly)],
                    "size": 9,
                },
                hovertemplate=(
                    f"{model_name} · {name} %{{x}}<br>"
                    "Deviation: %{y:+.2f}%<br>"
                    "Difference: %{customdata[2]:+.3f} Å<br>"
                    "Predicted: %{customdata[0]:.3f} Å<br>"
                    "DFT: %{customdata[1]:.3f} Å<extra></extra>"
                ),
                row=1,
                col=polymorph_index + 1,
            )
            first_trace = False
    figure.add_hline(y=0, line_color="#555555", line_dash="dash")
    for index, axes in enumerate(visible_axes, start=1):
        figure.update_xaxes(
            title_text="Cell axis",
            categoryorder="array",
            categoryarray=["abc"[axis] for axis in sorted(axes)],
            row=1,
            col=index,
        )
    figure.update_yaxes(title_text="Deviation from DFT / %", row=1, col=1)
    figure.update_layout(title="Al3Ti lattice parameter deviations", height=460)
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    figure.write_json(OUT_PATH / "figure_lattice_parameters.json")


@pytest.fixture
def volume_errors(volumes_per_atom: dict[str, list[float]]) -> dict[str, float]:
    """
    Get the volume-per-atom mean absolute error for each model.

    Parameters
    ----------
    volumes_per_atom
        Reference and model volumes per atom in polymorph order.

    Returns
    -------
    dict[str, float]
        Volume-per-atom MAE in Å³/atom for each model.
    """
    return {
        model_name: mae(volumes_per_atom["ref"], values)
        for model_name, values in volumes_per_atom.items()
        if model_name != "ref"
    }


@pytest.fixture
def lattice_parameter_errors(
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, float]:
    """
    Get equally polymorph-weighted lattice parameter relative errors.

    The two basal lengths share one parameter weight in each tetragonal phase.

    Parameters
    ----------
    reference_data
        DFT lattice parameters for the three polymorphs.
    relaxed_polymorphs
        Relaxed structures by model, in polymorph order.

    Returns
    -------
    dict[str, float]
        Mean absolute percentage error for each model.
    """
    errors = {}
    for model_name, structures in relaxed_polymorphs.items():
        polymorph_errors = []
        for name, atoms in zip(POLYMORPHS, structures, strict=True):
            if atoms is None or atoms.cell.rank != 3 or atoms.cell.volume <= 0:
                polymorph_errors.append(np.nan)
                continue
            lengths = atoms.cell.lengths()
            a_ref = float(reference_data[name]["lattice_a_angstrom"]["value"])
            if name == "L12":
                polymorph_errors.append(np.mean(np.abs(lengths / a_ref - 1)))
            else:
                c_ref = float(reference_data[name]["lattice_c_angstrom"]["value"])
                basal_error = np.mean(np.abs(lengths[:2] / a_ref - 1))
                c_error = abs(lengths[2] / c_ref - 1)
                polymorph_errors.append((basal_error + c_error) / 2)
        errors[model_name] = (
            float(100.0 * np.mean(polymorph_errors))
            if np.isfinite(polymorph_errors).all()
            else np.nan
        )
    return errors


@pytest.fixture
@build_table(
    filename=OUT_PATH / "al3ti_metrics.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    formation_energy_errors: dict[str, float],
    polymorph_order_accuracies: dict[str, float],
    volume_errors: dict[str, float],
    lattice_parameter_errors: dict[str, float],
    formation_energies: dict[str, list[float]],
    reference_data: dict[str, Any],
    relaxed_polymorphs: dict[str, list[Atoms | None]],
) -> dict[str, dict]:
    """
    Get all Al3Ti metrics.

    Parameters
    ----------
    formation_energy_errors
        Mean absolute formation energy error averaged across all polymorphs.
    polymorph_order_accuracies
        Pairwise accuracy of the predicted polymorph stability ordering.
    volume_errors
        Volume-per-atom mean absolute errors across all polymorphs.
    lattice_parameter_errors
        Equally polymorph-weighted lattice parameter relative errors.
    formation_energies
        Reference and predicted formation energies used for ordering plots.
    reference_data
        DFT lattice parameters used for the deviation plot.
    relaxed_polymorphs
        Relaxed structures used for the deviation plot.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    write_lattice_parameter_plot(lattice_parameters(reference_data, relaxed_polymorphs))
    write_formation_energy_ordering_plots(formation_energies)
    copy_structures_to_app_data()
    return {
        "Formation Energy MAE": formation_energy_errors,
        "Relative Stability Ordering Accuracy": polymorph_order_accuracies,
        "Volume MAE": volume_errors,
        "Lattice Parameter MAPE": lattice_parameter_errors,
    }


def copy_structures_to_app_data() -> None:
    """Replace app structures with valid outputs, removing stale files on failure."""
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        output_dir = OUT_PATH / model_name
        for name in POLYMORPHS:
            source = model_dir / f"{name}.xyz"
            target = output_dir / source.name
            temporary = output_dir / f"{name}.xyz.tmp"
            target.unlink(missing_ok=True)
            if not source.exists():
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                write(temporary, read(source), format="extxyz")
                temporary.replace(target)
            except Exception as exc:
                warn(f"Unable to copy structure {source}: {exc}", stacklevel=2)
            finally:
                temporary.unlink(missing_ok=True)


def test_al3ti_polymorphs(metrics: dict[str, dict[str, float]]) -> None:
    """
    Run al3ti polymorph regression analysis.

    Parameters
    ----------
    metrics
        Polymorph analysis metrics.
    """
    write_benchmark_info()
    return
