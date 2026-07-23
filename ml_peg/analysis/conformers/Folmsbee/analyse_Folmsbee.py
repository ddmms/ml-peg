"""Analyse Folmsbee benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write

# Optional extra (ml-peg[mlipaudit]); skip if not installed.
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.conformer_selection.conformer_selection import (
    ConformerSelectionModelOutput,
)

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegConformerSelectionBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "conformers" / "Folmsbee" / "outputs"
OUT_PATH = APP_ROOT / "data" / "conformers" / "Folmsbee"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def labels() -> list[str]:
    """
    Get list of system names.

    Returns
    -------
    list[str]
        List of all system names.
    """
    mock_path = CALC_PATH / "mock" / "model_output.json"
    if not mock_path.exists():
        raise ValueError(f"{mock_path} does not exist. Please run mock calculation.")
    raw = mock_path.read_text()
    output = ConformerSelectionModelOutput.model_validate_json(raw)
    benchmark = MlPegConformerSelectionBenchmark(
        force_field=Calculator(),
        data_input_dir=CALC_PATH,
        run_mode="standard",
    )

    # Get labels and elements, sorted consistently
    label_element_pairs = sorted(
        (
            f"{output_molecule.molecule_name}_conf{i}",
            benchmark_molecule.atom_symbols,
        )
        for output_molecule, benchmark_molecule in zip(
            output.molecules, benchmark._folmsbee_data, strict=True
        )
        for i in range(len(output_molecule.predicted_energy_profile))
    )
    labels_list, elements = map(list, zip(*label_element_pairs, strict=True))

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump({"elements": elements}, f, indent=1)

    return labels_list


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to ``(benchmark, ConformerSelectionResult)``.
    """
    results = {}
    for model_name in MODELS:
        model_path = CALC_PATH / model_name / "model_output.json"
        if not model_path.exists():
            results[model_name] = (None, None)
            continue
        benchmark = MlPegConformerSelectionBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        raw = model_path.read_text()
        benchmark.model_output = ConformerSelectionModelOutput.model_validate_json(raw)
        results[model_name] = (benchmark, benchmark.analyze())
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_folmsbee.json",
    title="Energies",
    x_label="Predicted energy / kcal/mol",
    y_label="Reference energy / kcal/mol",
    hoverdata={
        "Labels": labels(),
    },
)
def conformer_energies(analyze_results) -> dict[str, list]:
    """
    Get conformer energies for all systems.

    Parameters
    ----------
    analyze_results
        Mapping of model name to ``(benchmark, ConformerSelectionResult)``.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted barrier heights.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        benchmark, result = analyze_results[model_name]
        if benchmark is None or result is None:
            continue

        result_by_name = {m.molecule_name: m for m in result.molecules}
        data_by_name = {m.molecule_name: m for m in benchmark._folmsbee_data}

        for label in labels():
            mol_name, conf_str = label.rsplit("_conf", 1)
            i = int(conf_str)
            molecule = result_by_name[mol_name]

            results[model_name].append(float(molecule.predicted_energy_profile[i]))
            if not ref_stored:
                results["ref"].append(float(molecule.reference_energy_profile[i]))

            # Write structures for app
            data_mol = data_by_name[mol_name]
            atoms = Atoms(
                symbols=data_mol.atom_symbols,
                positions=data_mol.conformer_coordinates[i],
            )
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)
        ref_stored = True
    return results


@pytest.fixture
def get_mae(conformer_energies) -> dict[str, float]:
    """
    Get mean absolute error for conformer energies.

    Parameters
    ----------
    conformer_energies
        Dictionary of reference and predicted conformer energies.

    Returns
    -------
    dict[str, float]
        Per-molecule MAE averaged over all molecules, for each model.
    """
    label_mols = [lbl.rsplit("_conf", 1)[0] for lbl in labels()]
    results = {}
    for model_name in MODELS:
        groups: dict[str, tuple[list, list]] = {}

        if not conformer_energies[model_name]:
            results[model_name] = None
            continue

        for mol, ref, pred in zip(
            label_mols,
            conformer_energies["ref"],
            conformer_energies[model_name],
            strict=True,
        ):
            groups.setdefault(mol, ([], []))
            groups[mol][0].append(ref)
            groups[mol][1].append(pred)
        mol_maes = [mae(ref, pred) for ref, pred in groups.values()]
        results[model_name] = sum(mol_maes) / len(mol_maes)
    return results


@pytest.fixture
def get_score(analyze_results) -> dict[str, float | None]:
    """
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to ``(benchmark, ConformerSelectionResult)``.

    Returns
    -------
    dict[str, float | None]
        The mlipaudit per-molecule soft-threshold score (0 to 1) for each model.
    """
    return {
        model_name: result.score if result is not None else None
        for model_name, (_, result) in analyze_results.items()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "folmsbee_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(get_mae: dict[str, float], get_score: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.
    get_score
        The mlipaudit benchmark scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
        "Conformer Score": get_score,
    }


def test_folmsbee(metrics: dict[str, dict]) -> None:
    """
    Run Folmsbee analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Folmsbee metric results provided by fixtures.
    """
