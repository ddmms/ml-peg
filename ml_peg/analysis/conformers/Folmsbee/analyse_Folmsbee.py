"""Analyse Folmsbee benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from mlipaudit.benchmarks.conformer_selection.conformer_selection import (
    ConformerSelectionModelOutput,
)
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegConformerSelectionBenchmark
from ml_peg.calcs.utils.utils import download_s3_data  # noqa: F401
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


def labels() -> list:
    """
    Get list of system names.

    Returns
    -------
    list
        List of all system names.
    """
    for model_name in MODELS:
        raw = (CALC_PATH / model_name / "model_output.json").read_text()
        output = ConformerSelectionModelOutput.model_validate_json(raw)
        labels_list = sorted(
            f"{m.molecule_name}_conf{i}"
            for m in output.molecules
            for i in range(len(m.predicted_energy_profile))
        )
        break
    return labels_list


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
def conformer_energies() -> dict[str, list]:
    """
    Get conformer energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted barrier heights.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    data_input_dir = download_s3_data(
        key="inputs/conformers/Folmsbee/conformer_selection.zip",
        filename="conformer_selection.zip",
    )

    for model_name in MODELS:
        benchmark = MlPegConformerSelectionBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        raw = (CALC_PATH / model_name / "model_output.json").read_text()
        benchmark.model_output = ConformerSelectionModelOutput.model_validate_json(raw)
        result = benchmark.analyze()

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
@build_table(
    filename=OUT_PATH / "folmsbee_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(get_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
    }


def test_folmsbee(metrics: dict[str, dict]) -> None:
    """
    Run Folmsbee analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Folmsbee metric results provided by fixtures.
    """
