"""Analyse the isomer energy benchmarks within the GSCDB138 collection."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "isomers" / "GSCDB138" / "outputs"
OUT_PATH = APP_ROOT / "data" / "isomers" / "GSCDB138"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)
DATASETS = [
    "A19Rel6",
    "ACONF",
    "AlkIsomer11",
    "Amino20x4",
    "BUT14DIOL",
    "C20C246",
    "C60ISO7",
    "DIE60",
    "EIE22",
    "H2O16Rel4",
    "H2O20Rel9",
    "ICONF",
    "IDISP",
    "ISO34",
    "ISOL23",
    "ISOMERIZATION20",
    "MCONF",
    "PArel",
    "PCONF21",
    "Pentane13",
    "S66Rel7",
    "SCONF",
    "Styrene42",
    "SW49Rel28",
    "TAUT15",
    "UPU23",
]


def get_system_names() -> list[str]:
    """
    Get list of reaction system names from the first available model.

    Returns
    -------
    list[str]
        List of base system names (without suffixes).
    """
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            system_names = []
            for system_path in sorted(model_dir.glob("*_ts.xyz")):
                system_names.append(system_path.stem.replace("_ts", ""))
            if system_names:
                return system_names
    return []


def get_barrier_labels() -> list[str]:
    """
    Get list of barrier labels for plotting (two per reaction system).

    Returns
    -------
    list[str]
        List of barrier labels with reaction context.
    """
    return [
        label
        for system in get_system_names()
        for label in [f"{system} (TS-Reactants)", f"{system} (TS-Products)"]
    ]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_gscdb138_isomers.json",
    title="Reaction barriers",
    x_label="Predicted relative energy / eV",
    y_label="Reference relative energy / eV",
    hoverdata={
        "System": [
            system_name
            for system_name in get_system_names()
            for _ in range(2)  # Duplicate each system name for both barriers
        ],
        "Barrier Type": [
            barrier_type
            for _ in get_system_names()
            for barrier_type in ["TS-Reactants", "TS-Products"]
        ],
    },
)
def relative_energies() -> dict[str, list]:
    """
    Get relative energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS} | {"dataset": []}
    ref_stored = False

    # system_names = get_system_names()
    for model_name in MODELS:
        for xyz_path in (CALC_PATH / model_name).glob("*.xyz"):
            atoms_list = read(xyz_path, ":")
            results["dataset"].append(atoms_list[0].info["dataset"])
            results[model_name].append(atoms_list[0].info["model_rel_energy"])
            if not ref_stored:
                results["ref"].append(atoms_list[0].info["ref_rel_energy"])

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)

            # Write individual structures
            write(structs_dir / f"{xyz_path.stem}.xyz", atoms_list)
            ref_stored = True  # Ensure reference energies are only stored once.
    return results


# @pytest.fixture
def get_mae(relative_energies, dataset) -> dict[str, float]:
    """
    Get mean absolute error for relative energies.

    Parameters
    ----------
    relative_energies
        Dictionary of reference and predicted relative energies.
    dataset
        Datasets to use.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted relative energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        ref_energies = [
            relative_energies["ref"][i]
            for i, d in enumerate(relative_energies["dataset"])
            if d == dataset
        ]
        pred_energies = [
            relative_energies[model_name][i]
            for i, d in enumerate(relative_energies["dataset"])
            if d == dataset
        ]
        results[model_name] = mae(ref_energies, pred_energies)
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "gscdb138_isomers_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics() -> dict[str, dict]:
    """
    Get all metrics.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        f"{dataset} MAE": get_mae(relative_energies, dataset) for dataset in DATASETS
    }


def test_gscdb138(metrics: dict[str, dict]) -> None:
    """
    Run GSCDB138 test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
