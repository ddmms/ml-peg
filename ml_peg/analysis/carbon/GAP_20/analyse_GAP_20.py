"""Analyse GAP-20 carbon benchmark (DFT/optB88vdW, relative to isolated atom)."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

CALC_PATH = CALCS_ROOT / "carbon" / "GAP_20" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "GAP_20"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

METRIC_KEY = "Energy MAE"


@pytest.fixture
def all_energies() -> dict[str, dict]:
    """
    Load calc results for all models.

    Returns
    -------
    dict[str, dict]
        Model -> {ref, pred, config_type} lists.
    """
    data: dict[str, dict] = {
        model: {"ref": [], "pred": [], "config_type": []} for model in MODELS
    }

    for model in MODELS:
        struct_dir = OUT_PATH / model
        struct_dir.mkdir(parents=True, exist_ok=True)
        atoms_list = read(CALC_PATH / model / "results.xyz", ":")
        for idx, atoms in enumerate(tqdm(atoms_list, desc=model)):
            data[model]["ref"].append(atoms.info["ref_energy_rel"])
            data[model]["pred"].append(atoms.info["pred_energy_rel"])
            data[model]["config_type"].append(atoms.info.get("config_type", "unknown"))
            write(struct_dir / f"{idx}.xyz", atoms)

    return data


@pytest.fixture
@cell_to_scatter(
    filename=OUT_PATH / "GAP_20_scatter.json",
    x_label="Reference energy / eV atom⁻¹",
    y_label="Predicted energy / eV atom⁻¹",
)
def interactive_dataset(all_energies: dict) -> dict:
    """
    Build cell_to_scatter dataset with config_type as hover label.

    Parameters
    ----------
    all_energies
        Model -> {ref, pred, config_type} from all_energies fixture.

    Returns
    -------
    dict
        Data bundle consumed by cell_to_scatter decorator.
    """
    dataset: dict = {
        "metrics": {METRIC_KEY: METRIC_KEY},
        "models": {},
    }
    for model in MODELS:
        d = all_energies[model]
        points = [
            {"id": ct, "ref": r, "pred": p}
            for ct, r, p in zip(d["config_type"], d["ref"], d["pred"], strict=True)
        ]
        dataset["models"][model] = {"metrics": {METRIC_KEY: {"points": points}}}
    return dataset


@pytest.fixture
@build_table(
    filename=OUT_PATH / "GAP_20_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(all_energies: dict) -> dict:
    """
    Compute overall energy MAE per model.

    Parameters
    ----------
    all_energies
        Model -> {ref, pred, config_type} from all_energies fixture.

    Returns
    -------
    dict
        Metric name -> model -> MAE value.
    """
    return {
        METRIC_KEY: {
            model: mae(all_energies[model]["ref"], all_energies[model]["pred"])
            for model in MODELS
        }
    }


def test_gap_20(interactive_dataset: dict, metrics: dict) -> None:
    """
    Run GAP-20 analysis (drives all fixtures).

    Parameters
    ----------
    interactive_dataset
        Pre-generated scatter figures from cell_to_scatter fixture.
    metrics
        Energy MAE per model from metrics fixture.
    """
    return
