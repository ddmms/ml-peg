"""Analyse MAD-BENCH predictions against MAD-1.6 r2SCAN reference labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import json
from pathlib import Path
from warnings import warn

from ase import Atoms, units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    build_density_inputs,
    load_metrics_config,
    mae,
    rmse,
    sample_density_grid,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.filters import _model_supported_elements
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "mad_bench" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "mad_bench"

S3_KEY = "inputs/bulk_crystal/mad_bench/mad_bench.zip"
S3_FILENAME = "mad_bench.zip"
STRUCTURES_FILENAME = "mad_bench.xyz"

STRUCTURE_SET = "MAD-1.6 test geometries"

# every DFT reference protocol that may label the fixed structure set
REFERENCE_PROTOCOLS = {
    "mad-1.6-r2scan-fhi-aims": {
        "label": "MAD-1.6 - r²SCAN - FHI-aims",
        "dataset": "MAD-1.6",
        "level_of_theory": "r2SCAN",
        "code": "FHI-aims",
        "forces_key": "r2scan_forces",
        "energy_key": "r2scan_energy",
        "stress_key": "r2scan_stress",
        "units": {"forces": "eV/Å", "energy": "eV", "stress": "eV/Å³"},
    },
}
SCORED_REFERENCE = "mad-1.6-r2scan-fhi-aims"

# stress labels and predictions are in eV per cubic Angstrom and are reported in GPa
STRESS_TO_GPA = 1 / units.GPa
DISPLAY_UNITS = {"forces": "eV/Å", "energy": "eV", "stress": "GPa"}

PRED_FORCES_KEY = "pred_forces"
PRED_STRESS_KEY = "pred_stress"

GEOMETRY_TOLERANCE = 1e-6

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def structure_id(atoms: Atoms) -> tuple[str, int]:
    """
    Get the MAD-1.6 identity of a structure.

    Parameters
    ----------
    atoms
        Benchmark structure.

    Returns
    -------
    tuple[str, int]
        MAD-1.6 subset name and frame index within that subset.
    """
    return atoms.info["subset"], int(atoms.info["frame_id"])


def same_geometry(reference: Atoms, prediction: Atoms) -> bool:
    """
    Check that a prediction was made on the reference geometry and atom order.

    Parameters
    ----------
    reference
        Reference structure.
    prediction
        Structure written by the calculation.

    Returns
    -------
    bool
        Whether species order, periodicity, positions and cell all match.
    """
    return (
        len(reference) == len(prediction)
        and (reference.numbers == prediction.numbers).all()
        and (reference.pbc == prediction.pbc).all()
        and np.allclose(
            reference.positions,
            prediction.positions,
            rtol=0,
            atol=GEOMETRY_TOLERANCE,
        )
        and np.allclose(
            reference.cell.array,
            prediction.cell.array,
            rtol=0,
            atol=GEOMETRY_TOLERANCE,
        )
    )


def align_predictions(
    reference: list[Atoms], predictions: list[Atoms]
) -> list[Atoms | None]:
    """
    Order predicted structures as the reference, rejecting stale outputs.

    Parameters
    ----------
    reference
        Reference structures from the benchmark input file.
    predictions
        Structures written by the calculation for one model.

    Returns
    -------
    list[Atoms | None]
        Prediction for each reference structure, None where missing.

    Raises
    ------
    ValueError
        If identities are duplicated or unknown, or a geometry does not match.
    """
    prediction_ids = [structure_id(atoms) for atoms in predictions]
    duplicates = [key for key, count in Counter(prediction_ids).items() if count > 1]

    if duplicates:
        raise ValueError(f"Duplicate prediction identities {duplicates[:5]}")

    pred_by_id = dict(zip(prediction_ids, predictions, strict=True))
    unknown = set(pred_by_id) - {structure_id(atoms) for atoms in reference}

    if unknown:
        raise ValueError(f"Predictions for unknown structures {sorted(unknown)[:5]}")

    aligned = []

    for ref_atoms in reference:
        pred_atoms = pred_by_id.get(structure_id(ref_atoms))

        if pred_atoms is not None and not same_geometry(ref_atoms, pred_atoms):
            raise ValueError(
                f"Geometry or atom order of {structure_id(ref_atoms)} differs from "
                "the reference"
            )

        aligned.append(pred_atoms)

    return aligned


def prediction_outcome(pred_atoms: Atoms | None, name: str) -> str:
    """
    Get the status the calculation recorded for one property of one structure.

    Parameters
    ----------
    pred_atoms
        Aligned prediction, or None if missing.
    name
        Property name, one of ``forces``, ``energy`` or ``stress``.

    Returns
    -------
    str
        Status recorded by the calculation, ``missing`` if there is no prediction, or
        ``no_status`` if the prediction has no status for this property.
    """
    if pred_atoms is None:
        return "missing"

    return pred_atoms.info.get(f"{name}_status", "no_status")


def property_coverage(
    population: list[tuple[Atoms, Atoms | None]],
    name: str,
    declared_elements: frozenset[str] | None,
) -> dict:
    """
    Count prediction outcomes, separately for structures outside declared coverage.

    Parameters
    ----------
    population
        Reference structures with their aligned predictions.
    name
        Property name, one of ``forces``, ``energy`` or ``stress``.
    declared_elements
        Elements of the training data declared for the model, or None if unknown.

    Returns
    -------
    dict
        Count of each outcome, and the outcomes of structures containing elements
        outside the declared training data, None if no coverage is declared.
    """
    outcomes = Counter()
    outside_declared = Counter()

    for ref_atoms, pred_atoms in population:
        outcome = prediction_outcome(pred_atoms, name)
        outcomes[outcome] += 1

        if declared_elements is None:
            continue

        if not set(ref_atoms.get_chemical_symbols()) <= declared_elements:
            outside_declared[outcome] += 1

    return {
        "outcomes": dict(outcomes),
        "outside_declared_elements": (
            dict(outside_declared) if declared_elements is not None else None
        ),
    }


def collect_property(
    population: list[tuple[Atoms, Atoms | None]],
    name: str,
    get_ref: Callable[[Atoms], np.ndarray],
    get_pred: Callable[[Atoms], np.ndarray],
) -> dict:
    """
    Gather reference and predicted components of one property for one model.

    Parameters
    ----------
    population
        Reference structures with their aligned predictions.
    name
        Property name, one of ``forces`` or ``stress``.
    get_ref
        Function returning the reference values of a structure.
    get_pred
        Function returning the predicted values of a structure.

    Returns
    -------
    dict
        Flattened reference and predicted components, NaN for failed structures,
        and the source index of the structure each component belongs to.
    """
    ref_values = []
    pred_values = []
    source_indices = []

    for ref_atoms, pred_atoms in population:
        ref = np.asarray(get_ref(ref_atoms), dtype=float)

        if prediction_outcome(pred_atoms, name) == "ok":
            pred = np.asarray(get_pred(pred_atoms), dtype=float)
        else:
            pred = np.full(ref.shape, np.nan)

        ref_values.append(ref.ravel())
        pred_values.append(pred.ravel())
        source_indices.append(np.full(ref.size, ref_atoms.info["source_index"]))

    if not ref_values:
        return {"ref": np.array([]), "pred": np.array([]), "source_index": np.array([])}

    return {
        "ref": np.concatenate(ref_values),
        "pred": np.concatenate(pred_values),
        "source_index": np.concatenate(source_indices),
    }


def absolute_errors(ref: np.ndarray, prediction: np.ndarray) -> np.ndarray | None:
    """
    Get absolute errors, or None if any prediction is not finite.

    Parameters
    ----------
    ref
        Reference values.
    prediction
        Predicted values.

    Returns
    -------
    np.ndarray | None
        Absolute errors, None if any error is not finite.
    """
    errors = np.abs(np.asarray(prediction, dtype=float) - np.asarray(ref, dtype=float))

    if errors.size == 0 or not np.isfinite(errors).all():
        return None

    return errors


def p8_error(ref: np.ndarray, prediction: np.ndarray) -> float:
    """
    Get the normalised eighth root of the mean eighth power error.

    Parameters
    ----------
    ref
        Reference values.
    prediction
        Predicted values.

    Returns
    -------
    float
        Normalised L8 error, NaN if any prediction is not finite.
    """
    errors = absolute_errors(ref, prediction)

    if errors is None:
        return np.nan

    largest = errors.max()

    if largest == 0:
        return 0.0

    # scale by the largest error so the eighth power cannot overflow
    return float(largest * np.mean((errors / largest) ** 8) ** 0.125)


def sup_error(ref: np.ndarray, prediction: np.ndarray) -> float:
    """
    Get the maximum absolute error.

    Parameters
    ----------
    ref
        Reference values.
    prediction
        Predicted values.

    Returns
    -------
    float
        Maximum absolute error, NaN if any prediction is not finite.
    """
    errors = absolute_errors(ref, prediction)

    if errors is None:
        return np.nan

    return float(errors.max())


def error_statistics(ref: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """
    Get MAE, RMSE, normalised L8 and maximum absolute error.

    Parameters
    ----------
    ref
        Reference values.
    prediction
        Predicted values.

    Returns
    -------
    dict[str, float]
        Error statistics, each NaN if any prediction is not finite.
    """
    if absolute_errors(ref, prediction) is None:
        return dict.fromkeys(("MAE", "RMSE", "p8", "sup"), np.nan)

    return {
        "MAE": float(mae(ref, prediction)),
        "RMSE": float(rmse(ref, prediction)),
        "p8": p8_error(ref, prediction),
        "sup": sup_error(ref, prediction),
    }


def available_references(structures: list[Atoms]) -> list[str]:
    """
    Get the reference protocols whose force labels cover every structure.

    Parameters
    ----------
    structures
        Benchmark structures with reference labels.

    Returns
    -------
    list[str]
        Keys of the reference protocols present in the input data.
    """
    return [
        key
        for key, protocol in REFERENCE_PROTOCOLS.items()
        if all(protocol["forces_key"] in atoms.arrays for atoms in structures)
    ]


def reference_property_data(
    population: list[tuple[Atoms, Atoms | None]],
    protocol: dict,
    declared_elements: frozenset[str] | None,
) -> dict[str, dict]:
    """
    Compare one model's predictions with the labels of one reference protocol.

    Parameters
    ----------
    population
        Reference structures with their aligned predictions.
    protocol
        Reference protocol from ``REFERENCE_PROTOCOLS``.
    declared_elements
        Elements of the training data declared for the model, or None if unknown.

    Returns
    -------
    dict[str, dict]
        Force components in eV per Angstrom and stress components in GPa with
        coverage, and energy coverage.
    """
    energy_population = [
        pair for pair in population if protocol["energy_key"] in pair[0].info
    ]
    stress_population = [
        pair for pair in population if protocol["stress_key"] in pair[0].info
    ]

    forces = collect_property(
        population,
        "forces",
        lambda atoms: atoms.arrays[protocol["forces_key"]],
        lambda atoms: atoms.arrays[PRED_FORCES_KEY],
    )
    stress = collect_property(
        stress_population,
        "stress",
        lambda atoms: atoms.info[protocol["stress_key"]] * STRESS_TO_GPA,
        lambda atoms: atoms.info[PRED_STRESS_KEY] * STRESS_TO_GPA,
    )

    forces.update(property_coverage(population, "forces", declared_elements))
    stress.update(property_coverage(stress_population, "stress", declared_elements))

    return {
        "forces": forces,
        "stress": stress,
        "energy": property_coverage(energy_population, "energy", declared_elements),
    }


def density_inputs(reference_data: dict[str, dict], name: str) -> dict[str, dict]:
    """
    Get finite components of one property for the density parity plot.

    Parameters
    ----------
    reference_data
        Per-property components and coverage per model for one reference.
    name
        Property name, one of ``forces`` or ``stress``.

    Returns
    -------
    dict[str, dict]
        Density scatter inputs per model.
    """
    plot_stats = {}

    for model_name, data in reference_data.items():
        values = data[name]
        finite = np.isfinite(values["pred"])
        n_failed = sum(values["outcomes"].values()) - values["outcomes"].get("ok", 0)

        plot_stats[model_name] = {
            name: {
                "ref": values["ref"][finite].tolist(),
                "pred": values["pred"][finite].tolist(),
            },
            "excluded": n_failed,
        }

    return build_density_inputs(list(plot_stats), plot_stats, name, metric_fn=mae)


def density_point_sources(values: dict) -> list[int]:
    """
    Get the source structure of each point in a density parity plot.

    Parameters
    ----------
    values
        Reference and predicted components with their source indices.

    Returns
    -------
    list[int]
        Source index of the structure behind each plotted point, in plot order.
    """
    finite = np.isfinite(values["pred"])

    # the same sampling as plot_density_scatter, so the order matches the plot
    sampled_indices, _, _ = sample_density_grid(
        values["ref"][finite], values["pred"][finite]
    )

    return [int(values["source_index"][finite][index]) for index in sampled_indices]


@pytest.fixture
def reference() -> list[Atoms]:
    """
    Load reference structures and write their elements for filtering.

    Returns
    -------
    list[Atoms]
        Benchmark structures with the labels of every reference protocol present.
    """
    data_path = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "mad_bench"
    structures_path = data_path / STRUCTURES_FILENAME

    write_struct_info(data_path=structures_path, out_path=OUT_PATH)

    return read(structures_path, index=":")


@pytest.fixture
def property_data(reference: list[Atoms]) -> dict[str, dict]:
    """
    Get per-property components and coverage for each reference and model.

    Parameters
    ----------
    reference
        Benchmark structures with reference labels.

    Returns
    -------
    dict[str, dict]
        Force and stress components with coverage, and energy coverage, per
        reference protocol and model.
    """
    declared = _model_supported_elements()
    references = available_references(reference)
    results = {key: {} for key in references}

    for model_name in MODELS:
        output_file = CALC_PATH / model_name / STRUCTURES_FILENAME

        if not output_file.exists():
            continue

        try:
            aligned = align_predictions(reference, read(output_file, index=":"))
        except ValueError as err:
            raise ValueError(
                f"Stale or invalid outputs in {output_file}. {err}"
            ) from err

        population = list(zip(reference, aligned, strict=True))

        # the same predictions are compared with every reference protocol
        for key in references:
            protocol = REFERENCE_PROTOCOLS[key]
            model_results = reference_property_data(
                population, protocol, declared.get(model_name)
            )

            for name, data in model_results.items():
                n_total = sum(data["outcomes"].values())
                n_ok = data["outcomes"].get("ok", 0)

                if n_ok < n_total:
                    warn(
                        f"{model_name} has valid {name} for {n_ok} of {n_total} "
                        f"MAD-BENCH structures {data['outcomes']}, so full-set "
                        f"{name} scores against {protocol['label']} are unavailable",
                        stacklevel=2,
                    )

            results[key][model_name] = model_results

    return results


def scored_mae(property_data: dict[str, dict], name: str) -> dict[str, float]:
    """
    Get the MAE of one property against the scored reference for each model.

    Parameters
    ----------
    property_data
        Per-property components and coverage per reference and model.
    name
        Property name, one of ``forces`` or ``stress``.

    Returns
    -------
    dict[str, float]
        MAE per model, NaN if any structure failed or the model was not run.
    """
    scored_data = property_data[SCORED_REFERENCE]
    results = {}

    for model_name in MODELS:
        if model_name not in scored_data:
            results[model_name] = np.nan
            continue

        values = scored_data[model_name][name]
        results[model_name] = mae(values["ref"], values["pred"])

    return results


@pytest.fixture
def force_mae(property_data: dict[str, dict]) -> dict[str, float]:
    """
    Get the force component MAE in eV per Angstrom against the scored reference.

    Parameters
    ----------
    property_data
        Per-property components and coverage per reference and model.

    Returns
    -------
    dict[str, float]
        Force MAE per model, NaN if any structure failed or the model was not run.
    """
    return scored_mae(property_data, "forces")


@pytest.fixture
def stress_mae(property_data: dict[str, dict]) -> dict[str, float]:
    """
    Get the Voigt stress component MAE in GPa against the scored reference.

    Parameters
    ----------
    property_data
        Per-property components and coverage per reference and model.

    Returns
    -------
    dict[str, float]
        Stress MAE per model, NaN if any stress structure failed or the model was
        not run.
    """
    return scored_mae(property_data, "stress")


@pytest.fixture
def density_plots(
    reference: list[Atoms], property_data: dict[str, dict]
) -> dict[str, dict]:
    """
    Write density parity plots and the structure behind each plotted point.

    Parameters
    ----------
    reference
        Benchmark structures with reference labels.
    property_data
        Per-property components and coverage per reference and model.

    Returns
    -------
    dict[str, dict]
        Density scatter inputs per reference protocol, property and model.
    """
    results = {}
    shown_sources = set()

    for key, reference_data in property_data.items():
        protocol = REFERENCE_PROTOCOLS[key]
        results[key] = {}
        point_sources = {}

        for name, label in (("forces", "force"), ("stress", "stress")):
            unit = DISPLAY_UNITS[name]

            plot = plot_density_scatter(
                filename=OUT_PATH / f"figure_{name}_density_{key}.json",
                title=f"MAD-BENCH {name} against {protocol['label']}",
                x_label=f"Reference {label} component / {unit}",
                y_label=f"Predicted {label} component / {unit}",
                annotation_metadata={"excluded": "Failed structures"},
            )(density_inputs)

            results[key][name] = plot(reference_data, name)

            point_sources[name] = {
                model_name: density_point_sources(data[name])
                for model_name, data in reference_data.items()
            }

            for sources in point_sources[name].values():
                shown_sources.update(sources)

        with (OUT_PATH / f"density_structures_{key}.json").open(
            "w", encoding="utf8"
        ) as file:
            json.dump(point_sources, file)

    # write each structure shown by any plot once, geometry only
    structures_dir = OUT_PATH / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    for atoms in reference:
        if atoms.info["source_index"] in shown_sources:
            geometry = Atoms(
                numbers=atoms.numbers,
                positions=atoms.positions,
                cell=atoms.cell,
                pbc=atoms.pbc,
            )
            write(structures_dir / f"{atoms.info['source_index']}.xyz", geometry)

    return results


@pytest.fixture
def diagnostics(
    reference: list[Atoms], property_data: dict[str, dict]
) -> dict[str, dict]:
    """
    Write unscored statistics and per-property coverage for each reference.

    Parameters
    ----------
    reference
        Benchmark structures with reference labels.
    property_data
        Per-property components and coverage per reference and model.

    Returns
    -------
    dict[str, dict]
        Diagnostics per reference protocol, with null statistics where coverage is
        incomplete.
    """
    references = {}

    for key, reference_data in property_data.items():
        protocol = REFERENCE_PROTOCOLS[key]
        models = {}

        for model_name, data in reference_data.items():
            model_results = {"energy": data["energy"]}

            for name in ("forces", "stress"):
                entry = {
                    "outcomes": data[name]["outcomes"],
                    "outside_declared_elements": data[name][
                        "outside_declared_elements"
                    ],
                    "n_components": int(data[name]["ref"].size),
                }

                statistics = error_statistics(data[name]["ref"], data[name]["pred"])

                for statistic, value in statistics.items():
                    entry[statistic] = value if np.isfinite(value) else None

                model_results[name] = entry

            models[model_name] = model_results

        n_energy = sum(protocol["energy_key"] in atoms.info for atoms in reference)
        n_stress = sum(protocol["stress_key"] in atoms.info for atoms in reference)

        references[key] = {
            "label": protocol["label"],
            "dataset": protocol["dataset"],
            "level_of_theory": protocol["level_of_theory"],
            "code": protocol["code"],
            "units": DISPLAY_UNITS,
            "label_units": protocol["units"],
            "n_structures": {
                "forces": len(reference),
                "energy": n_energy,
                "stress": n_stress,
            },
            "models": models,
        }

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    with (OUT_PATH / "mad_bench_diagnostics.json").open("w", encoding="utf8") as file:
        json.dump(
            {
                "structure_set": {
                    "label": STRUCTURE_SET,
                    "n_structures": len(reference),
                },
                "scored_reference": SCORED_REFERENCE,
                "references": references,
            },
            file,
            indent=1,
        )

    return references


@pytest.fixture
@build_table(
    filename=OUT_PATH / "mad_bench_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    force_mae: dict[str, float], stress_mae: dict[str, float]
) -> dict[str, dict]:
    """
    Get the MAD-BENCH metrics shown in the main table.

    Stress MAE has weight 0 in ``metrics.yml``, so it is shown but not scored.

    Parameters
    ----------
    force_mae
        Force MAE per model.
    stress_mae
        Stress MAE per model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {"Force MAE": force_mae, "Stress MAE": stress_mae}


def test_mad_bench(
    metrics: dict[str, dict],
    density_plots: dict[str, dict],
    diagnostics: dict[str, dict],
) -> None:
    """
    Run MAD-BENCH analysis.

    Parameters
    ----------
    metrics
        Scored MAD-BENCH metric names and values for all models.
    density_plots
        Density scatter inputs for force and stress components.
    diagnostics
        Unscored error statistics and coverage per model.
    """
    return
