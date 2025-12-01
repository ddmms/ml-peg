"""Analyse Wiggle150 benchmark."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase.io import read
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = {model: f"{model}-D3" for model in MODELS}
CALC_PATH = CALCS_ROOT / "molecular" / "Wiggle150" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "Wiggle150"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)


def _sorted_xyz_files(model_dir: Path) -> list[Path]:
    """
    Return `.xyz` files for a model sorted by filename index.

    Parameters
    ----------
    model_dir : Path
        Directory containing per-model Wiggle150 outputs.

    Returns
    -------
    list[Path]
        Sorted list of `.xyz` files.
    """
    return sorted(model_dir.glob("*.xyz"), key=lambda path: int(path.stem))


def _extract_metadata() -> dict[str, list[str]]:
    """
    Return structure identifiers and molecule labels from the first model output.

    Returns
    -------
    dict[str, list[str]]
        Ordered structure identifiers and molecule labels keyed by ``"structures"``
        and ``"molecules"``.
    """
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue
        xyz_files = _sorted_xyz_files(model_dir)
        if not xyz_files:
            continue

        structures: list[str] = []
        molecules: list[str] = []
        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            structures.append(atoms.info.get("structure", xyz_file.stem))
            molecules.append(atoms.info.get("molecule", ""))
        return {"structures": structures, "molecules": molecules}
    return {"structures": [], "molecules": []}


METADATA = _extract_metadata()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_relative_energies.json",
    title="Wiggle150 Relative Energies",
    x_label="Predicted relative energy / kcal/mol",
    y_label="Reference relative energy / kcal/mol",
    hoverdata={
        "Structure": METADATA["structures"],
        "Molecule": METADATA["molecules"],
    },
)
def relative_energies() -> dict[str, list[float]]:
    """
    Collect relative energies for all Wiggle150 conformers.

    Returns
    -------
    dict[str, list[float]]
        Mapping of model -> predicted relative energies, plus reference values.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[float]] = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            results[model_name] = []
            continue

        xyz_files = _sorted_xyz_files(model_dir)
        if not xyz_files:
            results[model_name] = []
            continue

        model_predictions: list[float] = []

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            model_predictions.append(atoms.info["relative_energy_pred_kcal"])

            if not ref_stored:
                results["ref"].append(atoms.info["relative_energy_ref_kcal"])

            dest_dir = OUT_PATH / model_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(xyz_file, dest_dir / xyz_file.name)

        results[model_name] = model_predictions

        if not ref_stored and results["ref"]:
            ref_stored = True

    return results


@pytest.fixture
def wiggle150_mae(relative_energies) -> dict[str, float]:
    """
    Compute MAE for Wiggle150 benchmark.

    Parameters
    ----------
    relative_energies : dict[str, list[float]]
        Mapping of model names to predicted relative energies, including the
        reference values under the ``"ref"`` key.

    Returns
    -------
    dict[str, float]
        Mean absolute error for each model (kcal/mol).
    """
    ref = relative_energies["ref"]
    mae_values: dict[str, float] = {}
    for model_name in MODELS:
        predictions = relative_energies[model_name]
        if ref and predictions:
            mae_values[model_name] = mae(ref, predictions)
        else:
            mae_values[model_name] = None
    return mae_values


@pytest.fixture
@build_table(
    filename=OUT_PATH / "wiggle150_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(wiggle150_mae: dict[str, float]) -> dict[str, dict]:
    """
    Aggregate Wiggle150 metrics.

    Parameters
    ----------
    wiggle150_mae : dict[str, float]
        Mean absolute error values per model.

    Returns
    -------
    dict[str, dict]
        Dictionary keyed by metric name containing per-model results.
    """
    return {
        "MAE": wiggle150_mae,
    }


def test_wiggle150(metrics: dict[str, dict]) -> None:
    """
    Run Wiggle150 analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Wiggle150 metric results provided by fixtures.
    """
    return
