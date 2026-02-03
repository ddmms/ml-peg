"""Analyse Relastab benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest
from scipy.stats import kendalltau, spearmanr

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
# D3_MODEL_NAMES = build_d3_name_map(MODELS)
D3_MODEL_NAMES = {m: m for m in MODELS}
CALC_PATH = CALCS_ROOT / "interstitial" / "Relastab" / "outputs"
OUT_PATH = APP_ROOT / "data" / "interstitial" / "Relastab"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of system names.

    Returns
    -------
    list[str]
        List of system names.
    """
    system_names = []

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            system_names = [xyz.stem for xyz in xyz_files]
            if system_names:
                break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_energy.json",
    title="Relastab Energies (Shifted)",
    x_label="Predicted Energy (Shifted) / eV",
    y_label="Reference Energy (Shifted) / eV",
    hoverdata={
        "System": get_system_names(),
    },
)
def stability_energies() -> dict[str, list]:
    """
    Get energies for Relastab systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))

        # Temporary lists to sort together
        temp_data = []

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            e_config = atoms.get_potential_energy()
            ref_energy = atoms.info.get("ref", None)

            if ref_energy is None:
                continue

            temp_data.append(
                {
                    "name": xyz_file.name,
                    "atoms": atoms,
                    "pred": e_config,
                    "ref": ref_energy,
                }
            )

            # Copy structure to output for App
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / xyz_file.name, atoms)

        # To make the plot nicer, mean shifting is usually good for total energies.

        preds = [d["pred"] for d in temp_data]
        refs = [d["ref"] for d in temp_data]

        if not preds or not refs:
            continue

        # Shift energies by mean to make them comparable on plot
        mean_pred = sum(preds) / len(preds)
        mean_ref = sum(refs) / len(refs)

        results[model_name] = [p - mean_pred for p in preds]

        if not ref_stored:
            results["ref"] = [r - mean_ref for r in refs]
            ref_stored = True

    return results


@pytest.fixture
def ranking_metrics(stability_energies) -> dict[str, dict[str, float]]:
    """
    Compute ranking metrics (KendallTau and Spearman).

    Parameters
    ----------
    stability_energies
        Dictionary of reference and predicted energies.

    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary of ranking metrics per model.
    """
    results = {}

    ref_values = stability_energies.get("ref", [])
    if not ref_values:
        return {}

    for model_name in MODELS:
        pred_values = stability_energies.get(model_name, [])
        if not pred_values or len(pred_values) != len(ref_values):
            results[model_name] = {"KendallTau": None, "Spearman": None}
            continue

        tau, _ = kendalltau(ref_values, pred_values)
        spearman, _ = spearmanr(ref_values, pred_values)

        results[model_name] = {"KendallTau": tau, "Spearman": spearman}

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "relastab_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(ranking_metrics: dict[str, dict[str, float]]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    ranking_metrics
        Dictionary of ranking metrics per model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    # Reshape for build_table: {MetricName: {ModelName: Value}}
    reshaped = {"KendallTau": {}, "Spearman": {}}

    for model, scores in ranking_metrics.items():
        reshaped["KendallTau"][model] = scores.get("KendallTau")
        reshaped["Spearman"][model] = scores.get("Spearman")

    return reshaped


def test_relastab_analysis(metrics: dict[str, dict]) -> None:
    """
    Run Relastab analysis test.

    Parameters
    ----------
    metrics
        All Relastab metrics.
    """
    return
