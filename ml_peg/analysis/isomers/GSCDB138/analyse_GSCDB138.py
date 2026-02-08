"""Analyse the isomer energy benchmarks within the GSCDB138 collection."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write

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
    # "C20C246",
    # "C60ISO7",
    "DIE60",
    "EIE22",
    # "H2O16Rel4",
    # "H2O20Rel9",
    "ICONF",
    "IDISP",
    "ISO34",
    # "ISOL23",
    "ISOMERIZATION20",
    "MCONF",
    "PArel",
    "PCONF21",
    # "Pentane13",
    "S66Rel7",
    "SCONF",
    # "Styrene42",
    # "SW49Rel28",
    "TAUT15",
    "UPU23",
]

EV_TO_KCAL = units.mol / units.kcal


def get_system_names(calc_path: Path, dataset: str) -> list[str]:
    """
    Get list of system names from the first available model for a dataset.

    Parameters
    ----------
    calc_path
        Path to calculation outputs.
    dataset
        Dataset to get relative energies for.

    Returns
    -------
    list[str]
        List of systems in the dataset.
    """
    for model_name in MODELS:
        model_dir = calc_path / model_name
        if model_dir.exists():
            system_names = []
            for system_path in sorted(model_dir.glob(f"{dataset}*.xyz")):
                system_names.append(system_path.stem.split("_")[1])
            if system_names:
                return system_names
    return []


def get_relative_energy(
    dataset: str, calc_path: Path, out_path: Path
) -> dict[str, list]:
    """
    Get all relative energies for a specific dataset.

    Parameters
    ----------
    dataset
        Dataset to get relative energies for.
    calc_path
        Path to calculation outputs.
    out_path
        Path to write outputs to.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energies for a dataset.
    """
    system_names = get_system_names(calc_path=calc_path, dataset=dataset)

    @plot_parity(
        filename=out_path / f"figure_gscdb138_isomers_{dataset}.json",
        title=f"GSCDB138 isomer {dataset} relative energies",
        x_label="Predicted relative energy / kcal/mol",
        y_label="Reference relative energy / kcal/mol",
        hoverdata={"System": system_names},
    )
    def relative_energies() -> dict[str, list]:
        """
        Get relative energies for all systems.

        Returns
        -------
        dict[str, list]
            Dictionary of all reference and predicted relative energies.
        """
        results = {"ref": []} | {mlip: [] for mlip in MODELS}
        ref_stored = False

        for model_name in MODELS:
            model_dir = calc_path / model_name
            for system_name in system_names:
                xyz_path = model_dir / f"{dataset}_{system_name}.xyz"
                atoms_list = read(xyz_path, ":")

                results[model_name].append(
                    atoms_list[0].info["model_rel_energy"] * EV_TO_KCAL
                )

                # Only store ref and dataset info once (from first model)
                if not ref_stored:
                    results["ref"].append(
                        atoms_list[0].info["ref_rel_energy"] * EV_TO_KCAL
                    )

                # Write structures for app
                structs_dir = out_path / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / f"{system_name}.xyz", atoms_list)
            ref_stored = True
        return results

    return relative_energies()


def get_mae(dataset: str, calc_path: Path, out_path: Path) -> dict[str, float]:
    """
    Get mean absolute error for relative energies for a specific dataset.

    Parameters
    ----------
    dataset
        Dataset name to filter by.
    calc_path
        Path to calculation outputs.
    out_path
        Path to write outputs to.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted relative energy errors for all models.
    """
    results = {}
    relative_energies = get_relative_energy(
        dataset=dataset, calc_path=calc_path, out_path=out_path
    )
    for model_name in MODELS:
        results[model_name] = mae(
            relative_energies["ref"], relative_energies[model_name]
        )
    return results


def get_metrics(datasets: list[str], calc_path: Path, out_path: Path) -> None:
    """
    Get metrics.

    Parameters
    ----------
    datasets
        List of dataset name to filter by.
    calc_path
        Path to calculation outputs.
    out_path
        Path to write outputs to.
    """

    @build_table(
        filename=out_path / "gscdb138_metrics_table.json",
        metric_tooltips=DEFAULT_TOOLTIPS,
        thresholds=DEFAULT_THRESHOLDS,
        mlip_name_map=D3_MODEL_NAMES,
    )
    def metrics() -> dict[str, dict]:
        """
        Get a metric for each dataset.

        Returns
        -------
        dict[str, dict]
            Metric names and values for all models.
        """
        return {
            f"{dataset} MAE": get_mae(
                dataset=dataset, calc_path=calc_path, out_path=out_path
            )
            for dataset in datasets
        }

    metrics()


def test_gscdb138() -> None:
    """Run GSCDB138 test."""
    get_metrics(datasets=DATASETS, calc_path=CALC_PATH, out_path=OUT_PATH)
