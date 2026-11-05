"""Analyse elemental cohesive energy benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest
import yaml

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "elemental_cohesive_energy" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "elemental_cohesive_energy"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

with open(CALC_PATH.parent / "reference.yml", encoding="utf8") as file:
    REFERENCE = yaml.safe_load(file)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    index="0",
    info_keys=["name"],
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


@pytest.fixture
def cohesive_energies() -> dict[str, list]:
    """
    Get predicted cohesive energies for all elements.

    Returns
    -------
    dict[str, list]
        Dictionary of predicted cohesive energies for all models.
    """
    results = {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        for name in INFO["name"]:
            struct_file = CALC_PATH / model_name / f"{name}.xyz"
            if not struct_file.is_file():
                results[model_name].append(np.nan)
                continue

            crystal, atom = read(struct_file, index=":")
            results[model_name].append(
                atom.get_potential_energy()
                - crystal.get_potential_energy() / len(crystal)
            )

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{name}.xyz", [crystal, atom])

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_cohesive_energies_pbe.json",
    title="Cohesive energies",
    x_label="Predicted cohesive energy / eV/atom",
    y_label="PBE cohesive energy / eV/atom",
    hoverdata={"Element": INFO["name"]},
)
def cohesive_energies_pbe(cohesive_energies: dict[str, list]) -> dict[str, list]:
    """
    Get PBE and predicted cohesive energies for all elements.

    Parameters
    ----------
    cohesive_energies
        Dictionary of predicted cohesive energies for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of PBE and predicted cohesive energies.
    """
    return {"ref": [REFERENCE[name]["pbe"] for name in INFO["name"]]} | (
        cohesive_energies
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_cohesive_energies_exp.json",
    title="Cohesive energies",
    x_label="Predicted cohesive energy / eV/atom",
    y_label="Experimental cohesive energy / eV/atom",
    hoverdata={"Element": INFO["name"]},
)
def cohesive_energies_exp(cohesive_energies: dict[str, list]) -> dict[str, list]:
    """
    Get experimental and predicted cohesive energies for all elements.

    Parameters
    ----------
    cohesive_energies
        Dictionary of predicted cohesive energies for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of experimental and predicted cohesive energies.
    """
    return {"ref": [REFERENCE[name]["experiment"] for name in INFO["name"]]} | (
        cohesive_energies
    )


@pytest.fixture
@build_table(
    filename=OUT_PATH / "elemental_cohesive_energy_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    cohesive_energies_pbe: dict[str, list], cohesive_energies_exp: dict[str, list]
) -> dict[str, dict]:
    """
    Get all elemental cohesive energy metrics.

    Parameters
    ----------
    cohesive_energies_pbe
        Dictionary of PBE and predicted cohesive energies.
    cohesive_energies_exp
        Dictionary of experimental and predicted cohesive energies.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE (PBE)": {
            mlip: mae(cohesive_energies_pbe["ref"], cohesive_energies_pbe[mlip])
            for mlip in MODELS
        },
        "MAE (Experimental)": {
            mlip: mae(cohesive_energies_exp["ref"], cohesive_energies_exp[mlip])
            for mlip in MODELS
        },
    }


def test_elemental_cohesive_energy(metrics: dict[str, dict]) -> None:
    """
    Run elemental cohesive energy test.

    Parameters
    ----------
    metrics
        All elemental cohesive energy metrics.
    """
    return
