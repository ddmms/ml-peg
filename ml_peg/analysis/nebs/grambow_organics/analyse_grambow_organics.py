"""Analyse the Grambow organics NEB convergence benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import write
import numpy as np
import plotly.graph_objects as go
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    FINAL_CONVERGENCE_THRESHOLD,
    NEB_DATASET_FILENAME,
)
from mlipaudit.io import load_model_output_from_disk

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegGrambowOrganicsBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegGrambowOrganicsBenchmark.name

CALC_PATH = CALCS_ROOT / "nebs" / "grambow_organics" / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / "grambow_organics"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def check_dataset() -> None:
    """
    Check the dataset saved by the calculation is available.

    The calculation copies and unzips the downloaded dataset into its outputs,
    so the analysis does not need to download the input data again.

    Raises
    ------
    ValueError
        If the dataset is missing from the calculation outputs.
    """
    dataset_path = CALC_PATH / BENCHMARK / NEB_DATASET_FILENAME
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist. Please run the calculation.")


def write_neb_profiles(model_name: str, states: list, references: list) -> None:
    """
    Export final NEB bands, individual profiles and a reaction overview.

    Parameters
    ----------
    model_name
        Registry model name used for the output directory.
    states
        Final simulation states in dataset order, including failed reactions.
    references
        Reference reactions in the same dataset order, with energies in kcal/mol.
    """
    model_path = OUT_PATH / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    profile_path = model_path / "profiles"
    profile_path.mkdir(parents=True, exist_ok=True)
    reactions, barriers, ref_barriers, converged = [], [], [], []
    for index, state in enumerate(states):
        if state is None:
            continue
        energies = np.asarray(state.potential_energy, dtype=float)
        positions = np.asarray(state.positions)
        if (
            energies.ndim != 1
            or len(energies) == 0
            or len(energies) != len(positions)
            or not np.isfinite(energies).all()
            or not np.isfinite(positions).all()
        ):
            continue
        # Keep dataset indices so failed reactions never shift structure mappings.
        reaction = str(index + 1)
        converged.append(
            bool(
                np.sqrt((state.forces**2).sum(axis=1).max())
                < FINAL_CONVERGENCE_THRESHOLD
            )
        )
        reactions.append(reaction)
        # Convert ASE energies to the reference dataset units, kcal/mol.
        relative_energies = (energies - energies[0]) * units.mol / units.kcal
        barriers.append(float(relative_energies.max()))
        figure = go.Figure()
        frames = [
            Atoms(numbers=state.atomic_numbers, positions=position)
            for position in positions
        ]
        write(model_path / f"band_{reaction}.extxyz", frames)
        figure.add_scatter(
            x=list(range(len(energies))),
            y=relative_energies.tolist(),
            name=f"Reaction {reaction}",
            mode="lines+markers",
            customdata=[reaction] * len(energies),
            hovertemplate=(
                "Reaction %{customdata}<br>Image %{x}<br>"
                "Relative energy: %{y:.3f} kcal/mol<extra></extra>"
            ),
        )
        reference = references[index]
        molecules = [
            reference.reactants,
            reference.transition_state,
            reference.products,
        ]
        ref_energies = np.array([molecule.energy for molecule in molecules])
        ref_energies = ref_energies - ref_energies[0]
        ref_barriers.append(float(ref_energies[1]))
        write(
            model_path / f"band_ref_{reaction}.extxyz",
            [
                Atoms(symbols=molecule.atom_symbols, positions=molecule.coordinates)
                for molecule in molecules
            ],
        )
        figure.add_hline(
            y=float(ref_energies[1]),
            line_dash="dash",
            line_color="red",
            line_width=1,
            annotation_text=f"DFT barrier: {ref_energies[1]:.3f} kcal/mol",
            annotation_position="top left",
        )
        figure.update_layout(
            title=f"Reaction {reaction}",
            xaxis_title="NEB image",
            yaxis_title="Relative energy / kcal/mol",
            showlegend=True,
        )
        figure.write_json(profile_path / f"{reaction}.json")

    statuses = [
        "Converged" if value else "Unconverged (barrier estimate)"
        for value in converged
    ]

    @plot_parity(
        x_label="MLIP forward barrier / kcal/mol",
        y_label="DFT forward barrier / kcal/mol",
        hoverdata={"Convergence": statuses, "Reaction": reactions},
        symbol_by=statuses,
        group_colors={
            "Converged": "#2ca02c",
            "Unconverged (barrier estimate)": "#ff7f0e",
        },
        filename=model_path / "figure_neb_overview.json",
    )
    def barrier_parity() -> dict:
        """
        Collect forward barriers for the shared parity plot.

        Returns
        -------
        dict
            Model and DFT barriers in matching reaction order.
        """
        return {model_name: barriers, "ref": ref_barriers}

    barrier_parity()


def barrier_height_mae(states: list, references: list) -> float:
    """
    Compare all final forward barriers with DFT, including unconverged estimates.

    Parameters
    ----------
    states
        Final simulation states in dataset order, including failed reactions.
    references
        Reference reactions in the same order, with energies in kcal/mol.

    Returns
    -------
    float
        Barrier MAE in kcal/mol, or NaN for missing or invalid results.
    """
    if not references or len(states) != len(references):
        return np.nan
    predictions, targets = [], []
    for state, reference in zip(states, references, strict=True):
        if state is None:
            return np.nan
        energies = np.asarray(state.potential_energy, dtype=float)
        if energies.ndim != 1 or len(energies) < 2 or not np.isfinite(energies).all():
            return np.nan
        predictions.append(float(energies.max() - energies[0]) * units.mol / units.kcal)
        targets.append(reference.transition_state.energy - reference.reactants.energy)
    if not np.isfinite(targets).all():
        return np.nan
    return mae(targets, predictions)


@pytest.fixture
def neb_metrics() -> dict[str, dict[str, float]]:
    """
    Get convergence rates and forward barrier errors for each model.

    Missing reactions invalidate both metrics. Unconverged reactions remain in
    the barrier error, with their final paths treated as barrier estimates.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric names mapped to per-model values.
    """
    check_dataset()
    benchmark = MlPegGrambowOrganicsBenchmark(
        force_field=Calculator(), data_input_dir=CALC_PATH, run_mode="standard"
    )
    references = [benchmark._grambow_data[key] for key in benchmark._reaction_ids]
    results = {"Convergence Rate": {}, "Barrier Height MAE": {}}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegGrambowOrganicsBenchmark
        )
        states = model_output.simulation_states
        write_neb_profiles(model_name, states, references)
        results["Barrier Height MAE"][model_name] = barrier_height_mae(
            states, references
        )
        if (
            not states
            or len(states) != len(references)
            or any(state is None for state in states)
        ):
            results["Convergence Rate"][model_name] = np.nan
            continue
        n_converged = sum(
            np.sqrt((state.forces**2).sum(axis=1).max()) < FINAL_CONVERGENCE_THRESHOLD
            for state in states
        )
        results["Convergence Rate"][model_name] = n_converged / len(references)
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write the combined element set to ``info.json`` for filtering.

    Returns
    -------
    dict
        Mapping with the sorted list of elements present in the dataset.
    """
    check_dataset()

    benchmark = MlPegGrambowOrganicsBenchmark(
        force_field=Calculator(),
        data_input_dir=CALC_PATH,
        run_mode="standard",
    )
    elements = sorted(
        {
            symbol
            for reaction in benchmark._grambow_data.values()
            for molecule in (
                reaction.reactants,
                reaction.products,
                reaction.transition_state,
            )
            for symbol in molecule.atom_symbols
        }
    )
    info = {"elements": elements}
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(json.dumps(info, indent=1))
    return info


@pytest.fixture
@build_table(
    filename=OUT_PATH / "grambow_organics_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(neb_metrics: dict[str, dict[str, float]]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    neb_metrics
        Convergence rates and barrier errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return neb_metrics


def test_grambow_organics(metrics: dict[str, dict], struct_info: dict) -> None:
    """
    Run Grambow organics NEB analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Grambow organics metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
