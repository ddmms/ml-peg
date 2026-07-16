"""Analyse energy response benchmark."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "electric_field" / "energy_response" / "outputs"
OUT_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_molecule_labels(dataset: str | None = None) -> list[str]:
    """
    Get molecule labels for field structures, optionally filtered to one dataset.

    Parameters
    ----------
    dataset
        Dataset filename to filter by (e.g. ``"ALKANES.xyz"``). If ``None``,
        returns labels for all datasets in sorted order.

    Returns
    -------
    list[str]
        Chemical formula for each field-structure, across the requested datasets.
    """
    labels = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue
        dataset_paths = sorted(model_dir.glob("*.xyz"))
        if dataset is not None:
            dataset_paths = [p for p in dataset_paths if p.name == dataset]
        for dataset_path in dataset_paths:
            structs = read(dataset_path, index=":")
            labels.extend(
                s.get_chemical_formula()
                for s in structs
                if any(s.info.get("external_field", [0, 0, 0]))
            )
        break  # labels only needed from one model
    return labels


@pytest.fixture
def energy_response() -> dict[str, dict]:
    """
    Get energy responses for all structures.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping model names (and "ref") to per-dataset energy
        response arrays.
    """
    results = {"ref": {}} | {mlip: {} for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        datasets = sorted(f.name for f in model_dir.glob("*.xyz"))
        if not datasets:
            continue

        for dataset in datasets:
            structs = read(CALC_PATH / model_name / dataset, index=":")

            no_field_structs = [s for s in structs if not any(s.info["external_field"])]
            field_structs = [s for s in structs if any(s.info["external_field"])]

            if not field_structs:
                warn(f"No field structures found in {dataset}, skipping.", stacklevel=2)
                continue

            # Model predictions
            no_field = np.array([s.get_potential_energy() for s in no_field_structs])
            field = np.array([s.get_potential_energy() for s in field_structs])
            valid = ~(np.isnan(no_field) | np.isnan(field))

            if not np.any(valid):
                warn(
                    f"All energies NaN for {model_name}/{dataset}, skipping.",
                    stacklevel=2,
                )
                continue

            results[model_name][dataset] = np.abs(field[valid] - no_field[valid])

            # Write structures to app data (once, alongside computing results)
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / dataset, structs)

            # Write one xyz file per field structure for the structure viewer
            dataset_stem = Path(dataset).stem
            valid_field_structs = [
                s for s, v in zip(field_structs, valid, strict=True) if v
            ]
            for i, s in enumerate(valid_field_structs):
                out_file = structs_dir / f"{dataset_stem}_{i:04d}.xyz"
                if not out_file.exists():
                    write(out_file, [s])

            # Reference values from ORCA — computed once, aligned to same valid mask
            if not ref_stored:
                ref_no_field = np.array(
                    [s.info["REF_energy"] for s in no_field_structs]
                )
                ref_field = np.array([s.info["REF_energy"] for s in field_structs])
                results["ref"][dataset] = np.abs(ref_field[valid] - ref_no_field[valid])

        if results[model_name]:
            ref_stored = True

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_energy_responses.json",
    title="Energy response of linear organic molecules",
    x_label="Predicted energy response / eV",
    y_label="Reference energy response / eV",
    hoverdata={
        "Molecule": get_molecule_labels(),
    },
)
def energy_responses(energy_response: dict[str, dict]) -> dict[str, list]:
    """
    Get energy responses for all datasets.

    Parameters
    ----------
    energy_response
        Per-dataset reference and predicted energy responses.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted energy responses.
    """
    flattened = {}
    for key, res in energy_response.items():
        arrays = list(res.values())
        flattened[key] = np.concatenate(arrays).tolist() if arrays else None
    return flattened


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_alkane_energy_responses.json",
    title="Energy response of alkanes",
    x_label="Predicted energy response / eV",
    y_label="Reference energy response / eV",
    hoverdata={"Molecule": get_molecule_labels("ALKANES.xyz")},
)
def alkane_energy_responses(energy_response: dict[str, dict]) -> dict[str, list]:
    """
    Get alkane energy responses for parity plot.

    Parameters
    ----------
    energy_response
        Per-dataset reference and predicted energy responses.

    Returns
    -------
    dict[str, list]
        Reference and predicted alkane energy responses for all models.
    """
    return {
        key: datasets["ALKANES.xyz"].tolist()
        for key, datasets in energy_response.items()
        if "ALKANES.xyz" in datasets
    }


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_cumulene_energy_responses.json",
    title="Energy response of cumulenes",
    x_label="Predicted energy response / eV",
    y_label="Reference energy response / eV",
    hoverdata={"Molecule": get_molecule_labels("CUMULENES.xyz")},
)
def cumulene_energy_responses(energy_response: dict[str, dict]) -> dict[str, list]:
    """
    Get cumulene energy responses for parity plot.

    Parameters
    ----------
    energy_response
        Per-dataset reference and predicted energy responses.

    Returns
    -------
    dict[str, list]
        Reference and predicted cumulene energy responses for all models.
    """
    return {
        key: datasets["CUMULENES.xyz"].tolist()
        for key, datasets in energy_response.items()
        if "CUMULENES.xyz" in datasets
    }


@pytest.fixture
def total_mae(energy_responses: dict[str, list]) -> dict[str, float]:
    """
    Get total MAE of all energy responses.

    Parameters
    ----------
    energy_responses
        Reference and predicted energy responses for all structures.

    Returns
    -------
    dict[str, float]
        Dictionary of total MAE values for each model.
    """
    results = {}
    ref = energy_responses.get("ref")
    for model_name in MODELS:
        pred = energy_responses.get(model_name)
        results[model_name] = (
            mae(ref, pred) if ref is not None and pred is not None else None
        )
    return results


@pytest.fixture
def alkane_mae(energy_response: dict[str, dict]) -> dict[str, float]:
    """
    Get MAE of alkane energy responses.

    Parameters
    ----------
    energy_response
        Per-dataset reference and predicted energy responses.

    Returns
    -------
    dict[str, float]
        Dictionary of alkane MAE values for each model.
    """
    results = {}
    for model_name in MODELS:
        ref = energy_response["ref"].get("ALKANES.xyz")
        pred = energy_response[model_name].get("ALKANES.xyz")
        results[model_name] = (
            mae(ref, pred) if ref is not None and pred is not None else None
        )
    return results


@pytest.fixture
def cumulene_mae(energy_response: dict[str, dict]) -> dict[str, float]:
    """
    Get MAE of cumulene energy responses.

    Parameters
    ----------
    energy_response
        Per-dataset reference and predicted energy responses.

    Returns
    -------
    dict[str, float]
        Dictionary of cumulene MAE values for each model, or None if data
        is not available.
    """
    results = {}
    for model_name in MODELS:
        ref = energy_response["ref"].get("CUMULENES.xyz")
        pred = energy_response[model_name].get("CUMULENES.xyz")
        results[model_name] = (
            mae(ref, pred) if ref is not None and pred is not None else None
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "energy_response_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    weights=DEFAULT_WEIGHTS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    total_mae: dict[str, float],
    alkane_mae: dict[str, float],
    cumulene_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all energy response metrics.

    Parameters
    ----------
    total_mae
        Total MAE value for all models.
    alkane_mae
        Alkane MAE value for all models.
    cumulene_mae
        Cumulene MAE value for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Total MAE": total_mae,
        "Alkanes MAE": alkane_mae,
        "Cumulenes MAE": cumulene_mae,
    }


def test_energy_response(
    metrics: dict[str, dict],
    alkane_energy_responses: dict[str, list],
    cumulene_energy_responses: dict[str, list],
) -> None:
    """
    Run energy response analysis.

    Parameters
    ----------
    metrics
        All energy response metric names and values for each model.
    alkane_energy_responses
        Alkane energy responses for parity plot.
    cumulene_energy_responses
        Cumulene energy responses for parity plot.
    """
    return
