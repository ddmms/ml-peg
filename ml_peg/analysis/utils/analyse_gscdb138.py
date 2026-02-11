"""Analyse GSCDB138 for arbitrary subsets."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, mae
from ml_peg.app.utils.utils import Thresholds
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

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
            for system_path in sorted(model_dir.glob(f"{dataset}_*.xyz")):
                system_name = system_path.stem[len(dataset) + 1 :]
                system_names.append(system_name)
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
        filename=out_path / f"figure_gscdb138_{dataset}.json",
        title=f"GSCDB138 {dataset} relative energies",
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
                xyz_name = f"{dataset}_{system_name}.xyz"
                xyz_path = model_dir / xyz_name
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
                write(structs_dir / xyz_name, atoms_list)
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


def get_gscdb138_metrics(
    datasets: list[str],
    calc_path: Path,
    out_path: Path,
    metric_tooltips: dict[str, str],
    thresholds: Thresholds,
    weights: dict[str, float],
) -> None:
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
    metric_tooltips
        Tooltips for table metric headers.
    thresholds
        Mapping of metric names to dicts containing "good", "bad", and a "unit" entry.
    weights
        Default weights for metrics.
    """

    @build_table(
        filename=out_path / "gscdb138_metrics_table.json",
        metric_tooltips=metric_tooltips,
        thresholds=thresholds,
        weights=weights,
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
