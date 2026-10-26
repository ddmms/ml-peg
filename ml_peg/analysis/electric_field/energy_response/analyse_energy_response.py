"""Analyse energy response benchmark."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    load_metrics_config,
    mae,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "electric_field" / "energy_response" / "outputs"
# The mock calculation is the canonical source of structures: it is not a model
# under test, so it cannot be dropped or truncated by a failed model run
MOCK_PATH = CALC_PATH / "mock"
# Declared rather than globbed, so a truncated mock cannot shrink the benchmark
ALKANES = "ALKANES.xyz"
CUMULENES = "CUMULENES.xyz"
DATASETS = (ALKANES, CUMULENES)
OUT_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def pair_by_formula(dataset_path: Path) -> dict[str, tuple[Atoms, Atoms]]:
    """
    Pair zero-field and field structures in one dataset by chemical formula.

    Structures are not stored in a consistent order, and ``config_type`` is shared by
    every structure in a dataset, so the chemical formula is the only reliable key.
    Each formula must appear exactly once with a field and once without; anything
    unpaired or duplicated is reported and dropped rather than mispaired.

    Parameters
    ----------
    dataset_path
        Path to the dataset file to read.

    Returns
    -------
    dict[str, tuple[Atoms, Atoms]]
        Zero-field and field structure for each formula, ordered by increasing
        number of atoms.
    """
    no_field: dict[str, Atoms] = {}
    field: dict[str, Atoms] = {}
    duplicated: set[str] = set()

    for struct in read(dataset_path, index=":"):
        if "external_field" not in struct.info:
            warn(
                f"Structure in {dataset_path.name} has no recorded external_field, "
                "skipping.",
                stacklevel=2,
            )
            continue

        formula = struct.get_chemical_formula()
        structs = field if np.any(struct.info["external_field"]) else no_field

        if formula in structs:
            warn(
                f"Duplicate {formula} structure in {dataset_path.name}, which cannot "
                "be paired unambiguously, skipping.",
                stacklevel=2,
            )
            duplicated.add(formula)
            continue

        structs[formula] = struct

    for formula in sorted(set(no_field) ^ set(field)):
        warn(
            f"{formula} in {dataset_path.name} has no field or zero-field "
            "counterpart, skipping.",
            stacklevel=2,
        )

    # Drop duplicated formulas rather than pairing an arbitrary first structure:
    # constitutional isomers share a formula, so the right partner is not knowable
    paired = (set(no_field) & set(field)) - duplicated

    return {
        formula: (no_field[formula], field[formula])
        for formula in sorted(paired, key=lambda f: (len(no_field[f]), f))
    }


@cache
def get_reference_pairs(dataset: str) -> dict[str, tuple[Atoms, Atoms]]:
    """
    Get paired structures for one dataset from the mock calculation.

    These pairs fix the formula order used for every model, the reference values and
    the hover labels, so all three stay aligned. Taking them from the mock outputs
    rather than from a model keeps an incomplete model run from omitting a formula
    for everyone; each model's own output is then compared against this set.

    Parameters
    ----------
    dataset
        Dataset filename (e.g. ``"ALKANES.xyz"``).

    Returns
    -------
    dict[str, tuple[Atoms, Atoms]]
        Zero-field and field structure for each formula.
    """
    dataset_path = MOCK_PATH / dataset
    if not dataset_path.exists():
        warn(
            f"{dataset_path} does not exist. Please run mock calculation.",
            stacklevel=2,
        )
        return {}
    return pair_by_formula(dataset_path)


@cache
def get_model_datasets(model_name: str) -> frozenset[str]:
    """
    Get the datasets one model has output for.

    Parameters
    ----------
    model_name
        Name of the model to look for output from.

    Returns
    -------
    frozenset[str]
        Dataset filenames this model has written output for.
    """
    return frozenset(
        dataset for dataset in DATASETS if (CALC_PATH / model_name / dataset).exists()
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
        Chemical formula for each paired structure, across the requested datasets.
    """
    datasets = DATASETS if dataset is None else (dataset,)
    return [formula for name in datasets for formula in get_reference_pairs(name)]


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

    for dataset in DATASETS:
        ref_pairs = get_reference_pairs(dataset)
        if not ref_pairs:
            warn(
                f"No paired field structures found in {dataset}, skipping.",
                stacklevel=2,
            )
            continue

        results["ref"][dataset] = np.array(
            [
                abs(field.info["REF_energy"] - no_field.info["REF_energy"])
                for no_field, field in ref_pairs.values()
            ]
        )

        for model_name in MODELS:
            model_datasets = get_model_datasets(model_name)

            # No output at all means the model has not been run, so leave it
            # unreported rather than claiming a result for it
            if not model_datasets:
                continue

            # Record NaN rather than skip: a skipped column is dropped from the score
            if dataset not in model_datasets:
                warn(
                    f"{dataset} missing from {model_name}, so its calculation is "
                    "unfinished. Recording NaN.",
                    stacklevel=2,
                )
                results[model_name][dataset] = np.full(len(ref_pairs), np.nan)
                continue

            pairs = pair_by_formula(CALC_PATH / model_name / dataset)

            # Keep the reference formula order, recording NaN for anything missing,
            # so every model aligns with the reference values and hover labels
            responses = []
            for formula in ref_pairs:
                if formula not in pairs:
                    warn(
                        f"{formula} missing from {model_name}/{dataset}, "
                        "recording NaN.",
                        stacklevel=2,
                    )
                    responses.append(np.nan)
                    continue
                no_field, field = pairs[formula]
                responses.append(
                    abs(field.get_potential_energy() - no_field.get_potential_energy())
                )

            # Record even when all-NaN, so the failure reaches the metrics
            if np.all(np.isnan(responses)):
                warn(
                    f"All energies NaN for {model_name}/{dataset}.",
                    stacklevel=2,
                )

            results[model_name][dataset] = np.array(responses)
            write_app_structs(model_name, dataset, ref_pairs, pairs)

    return results


def write_app_structs(
    model_name: str,
    dataset: str,
    ref_pairs: dict[str, tuple[Atoms, Atoms]],
    pairs: dict[str, tuple[Atoms, Atoms]],
) -> None:
    """
    Write structures to app data, one file per field structure for the viewer.

    Files follow the reference formula order, falling back to the reference structure
    for anything the model is missing, because the app maps a clicked parity point to
    a structure file by index. Existing files are overwritten so that the numbering
    never outlives the order it was written for.

    Parameters
    ----------
    model_name
        Name of the model the structures were calculated with.
    dataset
        Dataset filename (e.g. ``"ALKANES.xyz"``).
    ref_pairs
        Zero-field and field structure for each formula, fixing the order.
    pairs
        Zero-field and field structure for each formula, for this model.
    """
    structs_dir = OUT_PATH / model_name
    structs_dir.mkdir(parents=True, exist_ok=True)

    ordered = [pairs.get(formula, ref_pairs[formula]) for formula in ref_pairs]
    write(structs_dir / dataset, [struct for pair in ordered for struct in pair])

    dataset_stem = Path(dataset).stem
    for i, (_, field) in enumerate(ordered):
        write(structs_dir / f"{dataset_stem}_{i:04d}.xyz", [field])


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
    ref = energy_response["ref"]
    datasets = [dataset for dataset in DATASETS if dataset in ref]

    flattened = {}
    for key, res in energy_response.items():
        if not res:
            flattened[key] = None
            continue

        # Pad any dataset this model is missing, so that every model stays the same
        # length as the reference, and so aligns with the hover labels
        arrays = [
            res[dataset] if dataset in res else np.full(len(ref[dataset]), np.nan)
            for dataset in datasets
        ]
        flattened[key] = np.concatenate(arrays).tolist()
    return flattened


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_alkane_energy_responses.json",
    title="Energy response of alkanes",
    x_label="Predicted energy response / eV",
    y_label="Reference energy response / eV",
    hoverdata={"Molecule": get_molecule_labels(ALKANES)},
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
        key: datasets[ALKANES].tolist()
        for key, datasets in energy_response.items()
        if ALKANES in datasets
    }


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_cumulene_energy_responses.json",
    title="Energy response of cumulenes",
    x_label="Predicted energy response / eV",
    y_label="Reference energy response / eV",
    hoverdata={"Molecule": get_molecule_labels(CUMULENES)},
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
        key: datasets[CUMULENES].tolist()
        for key, datasets in energy_response.items()
        if CUMULENES in datasets
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
        ref = energy_response["ref"].get(ALKANES)
        pred = energy_response[model_name].get(ALKANES)
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
        ref = energy_response["ref"].get(CUMULENES)
        pred = energy_response[model_name].get(CUMULENES)
        results[model_name] = (
            mae(ref, pred) if ref is not None and pred is not None else None
        )
    return results


@pytest.fixture
def missing_mock_datasets() -> tuple[str, ...]:
    """
    Get benchmark datasets the mock calculation has not produced.

    Returns
    -------
    tuple[str, ...]
        Expected dataset filenames absent from the mock outputs.
    """
    return tuple(dataset for dataset in DATASETS if not (MOCK_PATH / dataset).exists())


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
    missing_mock_datasets: tuple[str, ...],
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
    missing_mock_datasets
        Expected dataset filenames absent from the mock outputs.
    """
    # Without every dataset there is no benchmark to score, however well the models
    # that did run performed
    assert not missing_mock_datasets, (
        f"Mock calculation is missing {', '.join(missing_mock_datasets)}. "
        "Please run mock calculation."
    )

    # Element info for the app's periodic table filter. Without it the benchmark is
    # left out of filtering entirely, rather than being filtered on C and H
    write_struct_info(
        data_path=[MOCK_PATH / dataset for dataset in DATASETS],
        out_path=OUT_PATH,
    )
