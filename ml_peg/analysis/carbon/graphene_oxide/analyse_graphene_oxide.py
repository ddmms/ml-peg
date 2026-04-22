"""Analyse graphene oxide benchmark (DFT/PBE, relative to isolated atoms)."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "carbon" / "graphene_oxide" / "outputs"
OUT_PATH = APP_ROOT / "data" / "carbon" / "graphene_oxide"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

METRIC_KEY = "Energy MAE"


def _decode_config(config_type: str) -> str:
    """
    Append decoded ratio labels to a config_type string.

    Parameters
    ----------
    config_type
        Raw config type string, e.g. '0.10-0.50' or '0.10-0.50-0.30'.

    Returns
    -------
    str
        Config type with ratio labels, e.g. '0.10-0.50 (O/C=0.10, OH/O=0.50)'.
    """
    parts = config_type.split("-")
    if len(parts) == 2:
        return f"{config_type} (O/C={parts[0]}, OH/O={parts[1]})"
    if len(parts) == 3:
        return f"{config_type} (O/C={parts[0]}, OH/O={parts[1]}, edge={parts[2]})"
    return config_type


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
    filename=OUT_PATH / "graphene_oxide_scatter.json",
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
            {"id": _decode_config(ct), "ref": r, "pred": p}
            for ct, r, p in zip(d["config_type"], d["ref"], d["pred"], strict=True)
        ]
        dataset["models"][model] = {"metrics": {METRIC_KEY: {"points": points}}}
    return dataset


@pytest.fixture
@build_table(
    filename=OUT_PATH / "graphene_oxide_metrics_table.json",
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


def test_graphene_oxide(interactive_dataset: dict, metrics: dict) -> None:
    """
    Run graphene oxide analysis (drives all fixtures).

    Parameters
    ----------
    interactive_dataset
        Pre-generated scatter figures from cell_to_scatter fixture.
    metrics
        Energy MAE per model from metrics fixture.
    """
    return
