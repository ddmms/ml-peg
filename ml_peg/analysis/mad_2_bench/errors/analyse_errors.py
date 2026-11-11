"""Analyse the MAD-2 test benchmark."""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path
from typing import Any

from ase.geometry import find_mic
from ase.io import read
import numpy as np
import yaml

from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "mad_2_bench" / "errors" / "outputs"
OUT_PATH = APP_ROOT / "data" / "mad_2_bench"
METADATA_PATH = Path(__file__).with_name("model_metadata.yml")

S3_KEY = "inputs/mad_2_bench/errors/errors.zip"
S3_FILENAME = "errors.zip"

REFERENCES = {
    "mad/r2scan": {"name": "MAD-2 r2SCAN", "file": "mad2_r2scan"},
    "matpes/pbe": {"name": "MatPES PBE", "file": "matpes_pbe"},
    "matpes/r2scan": {"name": "MatPES r2SCAN", "file": "matpes_r2scan"},
    "omat24/static": {"name": "OMat24", "file": "omat24"},
    "alexandria/static": {"name": "sAlex (Alexandria PBE)", "file": "alexandria"},
    "mptrj/static": {"name": "MPtrj", "file": "mptrj"},
}

COMPARISON_CLASSES = ("all", "bulk", "surface", "2D", "molecule", "cluster")

COMMON_COVERAGE = 0.90

MOLECULAR_CLASSES = ("molecule", "cluster")

COMPARABLE = {"same-family", "u-differs"}

UNSUPPORTED_MARKERS = (
    "is not in list",
    "configured to process the following elements",
    "do not know atomic number",
    "does not support the atomic type",
    "covers atomic numbers up to",
    "KeyError",
)

UNITS = {"energy": "meV_per_atom", "forces": "meV_per_A", "stress": "meV_per_A3"}
STATISTICS = ("mae", "rmse", "p95", "max")
ERRORS_STATISTICS = ("mae", "rmse", "p95")

EXCLUSION_LABELS = {
    "potcar-differs": "POTCAR differs",
    "u-differs": "Hubbard U differs",
    "definition-differs": "energy definition differs (TOTEN against sigma->0)",
    "unverified-potcar": "POTCAR of the training protocol unverified",
    "unverified-u": "Hubbard U of the training protocol unverified",
    "unverified-definition": "energy definition of the training protocol unverified",
    "unverified-source": "training energy source unverified",
    "unverified-corrections": (
        "MP2020 corrections of the training energies not documented"
    ),
    "functional-differs": "other functional",
    "incompatible": "other code or energy scale",
    "unresolved": "energy convention not verified yet",
}

ALIGNMENT_TOLERANCE = 1e-4


def load_labels(data_path: Path) -> list[dict[str, Any]]:
    """
    Read the labels of the six references.

    Parameters
    ----------
    data_path
        Extracted inputs folder.

    Returns
    -------
    list[dict[str, Any]]
        One entry per frame and reference.
    """
    protocols = json.loads((data_path / "protocols.json").read_text())
    labels = []

    for reference, spec in REFERENCES.items():
        for atoms in read(data_path / f"{spec['file']}.xyz", ":"):
            info = atoms.info
            symbols = atoms.get_chemical_symbols()
            protocol = protocols.get(reference, {}).get(info["protocol"])
            hubbard_u = {
                element: float(value)
                for element, value in (
                    item.split(":") for item in info.get("hubbard_u", "").split()
                )
            }
            class_name = info["class"]

            labels.append(
                {
                    "reference": reference,
                    "class": class_name,
                    "subset": info["subset"],
                    "identity": info["id"],
                    "frame": (spec["file"], info["id"]),
                    "atoms": atoms,
                    "natoms": len(atoms),
                    "symbols": set(symbols),
                    "energy": float(atoms.get_potential_energy()),
                    "atomization_energy": info.get("atomization_energy"),
                    "forces": atoms.get_forces(),
                    "stress": atoms.get_stress(voigt=True)
                    if class_name == "bulk" and "stress" in atoms.calc.results
                    else None,
                    "potcars": {
                        element: protocol["potcars"][element]
                        for element in dict.fromkeys(symbols)
                    }
                    if protocol
                    else {},
                    "hubbard": {
                        element: hubbard_u.get(element, 0.0)
                        for element in dict.fromkeys(symbols)
                    }
                    if protocol
                    else {},
                    "energy_definition": protocol["energy_definition"]
                    if protocol
                    else None,
                }
            )

    return labels


def load_predictions(calc_path: Path, model: str) -> dict[tuple[str, str], dict]:
    """
    Read the saved predictions of one model.

    Parameters
    ----------
    calc_path
        Folder with one output folder per model.
    model
        Model name.

    Returns
    -------
    dict[tuple[str, str], dict]
        Prediction of each (file, frame id) pair.
    """
    predictions = {}

    for line in (calc_path / model / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        for frame in row["frames"]:
            predictions[tuple(frame)] = row

    return predictions


def load_metadata() -> dict[str, dict]:
    """
    Read the model metadata.

    Returns
    -------
    dict[str, dict]
        Model metadata by model name.
    """
    return yaml.safe_load(METADATA_PATH.read_text()) or {}


def model_record(metadata: dict[str, dict], model: str) -> dict[str, Any]:
    """
    Get the metadata of a model.

    Parameters
    ----------
    metadata
        Model metadata by model name.
    model
        Model name.

    Returns
    -------
    dict[str, Any]
        Display text and scoring rules of the model.
    """
    if model in metadata:
        return metadata[model]

    return {
        "name": model,
        "checkpoint": "not documented",
        "training_data": "not documented",
        "molecular_only": False,
        "supported_elements": None,
        "stress": True,
        "energy_scale_sources": None,
        "energy_corrections": "unknown",
        "mad_energy_target": None,
        "energy_relation": None,
    }


def energy_relation(record: dict[str, Any], reference: str) -> str:
    """
    Get how the energies of a model relate to a reference.

    Parameters
    ----------
    record
        Model metadata.
    reference
        Reference key.

    Returns
    -------
    str
        Relation from the metadata, or "unknown".
    """
    if record["energy_relation"] is None:
        return "unknown"

    return record["energy_relation"][reference]


def protocol_tables(labels: list[dict[str, Any]]) -> tuple:
    """
    Collect the POTCAR, Hubbard U and energy definition facts.

    Parameters
    ----------
    labels
        All labels.

    Returns
    -------
    tuple
        Labels by frame, and POTCARs, elements with U and energy definitions by
        reference.
    """
    by_frame = {}
    potcars = collections.defaultdict(lambda: collections.defaultdict(set))
    with_u = collections.defaultdict(set)
    definitions = collections.defaultdict(set)

    for label in labels:
        if not label["potcars"]:
            continue

        reference = label["reference"]
        by_frame[(reference, label["identity"])] = label
        definitions[reference].add(label["energy_definition"])

        for element, identity in label["potcars"].items():
            potcars[reference][element].add(identity)

        for element, value in label["hubbard"].items():
            if value:
                with_u[reference].add(element)

    return by_frame, potcars, with_u, definitions


def source_status(source: str, label: dict[str, Any], tables: tuple) -> str:
    """
    Check whether a training protocol gives the same energy as a label.

    Parameters
    ----------
    source
        Reference whose protocol the training energies follow.
    label
        Label to compare with.
    tables
        Facts from `protocol_tables`.

    Returns
    -------
    str
        "compatible" or the first reason the energies differ.
    """
    if source == label["reference"]:
        return "compatible"

    by_frame, potcars, with_u, definitions = tables
    twin = by_frame.get((source, label["identity"]))

    if any(name.startswith("unverified") for name in label["potcars"].values()):
        return "unverified-potcar"

    if twin is not None:
        if any(name.startswith("unverified") for name in twin["potcars"].values()):
            return "unverified-potcar"
        if twin["potcars"] != label["potcars"]:
            return "potcar-differs"
        if twin["hubbard"] != label["hubbard"]:
            return "u-differs"

    else:
        for element, identity in label["potcars"].items():
            known = potcars[source].get(element, set())
            if len(known) != 1 or any(name.startswith("unverified") for name in known):
                return "unverified-potcar"
            if identity not in known:
                return "potcar-differs"

        for element, value in label["hubbard"].items():
            if value or element in with_u[source]:
                return "unverified-u"

    if len(definitions[source]) != 1:
        return "unverified-definition"

    if definitions[source] != {label["energy_definition"]}:
        return "definition-differs"

    return "compatible"


def energy_status(
    record: dict[str, Any], label: dict[str, Any], tables: tuple, cache: dict
) -> str:
    """
    Check whether the energy of a model can be compared with a label.

    Parameters
    ----------
    record
        Model metadata.
    label
        Label to compare with.
    tables
        Facts from `protocol_tables`.
    cache
        Source checks already done.

    Returns
    -------
    str
        "compatible" or the reason the energy is not compared.
    """
    if record["energy_relation"] is None:
        return "unverified-source"

    reference = label["reference"]
    relation = energy_relation(record, reference)

    if relation not in COMPARABLE:
        return relation

    if reference == "mad/r2scan":
        return "compatible"

    if record["energy_scale_sources"] is None:
        return "unverified-source"

    if record["energy_corrections"].startswith("unknown"):
        return "unverified-corrections"

    for source in record["energy_scale_sources"]:
        key = (source, id(label))
        if key not in cache:
            cache[key] = source_status(source, label, tables)
        if cache[key] != "compatible":
            return cache[key]

    return "compatible"


def evaluate(
    label: dict[str, Any], prediction: dict | None, record: dict[str, Any], status: str
) -> dict[str, Any]:
    """
    Get the absolute errors of one model on one label.

    Parameters
    ----------
    label
        Label to compare with.
    prediction
        Saved prediction, or None.
    record
        Model metadata.
    status
        Energy comparability of the label.

    Returns
    -------
    dict[str, Any]
        Status and absolute errors in eV units.
    """
    frame = {
        "key": label["frame"],
        "identity": label["identity"],
        "energy_status": status,
        "status": "missing",
        "energy": None,
        "forces": None,
        "stress": None,
    }

    if prediction is None:
        return frame

    supported = record["supported_elements"]
    if supported and not label["symbols"] <= set(supported):
        frame["status"] = "unsupported"
        return frame

    if "error" in prediction:
        unsupported = any(
            marker in prediction["error"] for marker in UNSUPPORTED_MARKERS
        )
        frame["status"] = "unsupported" if unsupported else "failed"
        return frame

    if not (
        np.isfinite(prediction["energy"]) and np.isfinite(prediction["forces"]).all()
    ):
        frame["status"] = "failed"
        return frame

    frame["status"] = "predicted"

    if status == "compatible":
        target = label["energy"]
        if (
            label["reference"] == "mad/r2scan"
            and record["mad_energy_target"] == "atomization_energy"
        ):
            target = label["atomization_energy"]
        frame["energy"] = abs(prediction["energy"] - target) / label["natoms"]

    forces = np.abs(np.asarray(prediction["forces"]) - label["forces"])
    frame["forces"] = forces.ravel().astype(np.float32)

    if (
        label["stress"] is not None
        and "stress" in prediction
        and record["stress"]
        and np.isfinite(label["stress"]).all()
        and np.isfinite(prediction["stress"]).all()
    ):
        stress = np.abs(np.asarray(prediction["stress"]) - label["stress"])
        frame["stress"] = stress.astype(np.float32)

    return frame


def statistics(prop: str, values: np.ndarray) -> dict[str, float | None]:
    """
    Get the mean, root mean square, 95th percentile and maximum.

    Parameters
    ----------
    prop
        Property name.
    values
        Absolute errors in meV units.

    Returns
    -------
    dict[str, float | None]
        Statistics named by property, statistic and unit.
    """
    names = [f"{prop}_{stat}_{UNITS[prop]}" for stat in STATISTICS]

    if not values.size:
        return dict.fromkeys(names)

    values = values.astype(np.float64)

    return dict(
        zip(
            names,
            [
                float(values.mean()),
                float(np.sqrt(np.mean(values**2))),
                float(np.percentile(values, 95)),
                float(values.max()),
            ],
            strict=True,
        )
    )


def property_statistics(
    frames: list[dict[str, Any]], prop: str
) -> dict[str, float | int | None]:
    """
    Get the statistics of one property on chosen frames.

    Parameters
    ----------
    frames
        Errors of one model.
    prop
        Property name.

    Returns
    -------
    dict[str, float | int | None]
        Statistics in meV units and the number of frames used.
    """
    kept = [frame for frame in frames if frame[prop] is not None]

    if prop == "energy":
        values = np.array([frame["energy"] for frame in kept])
    else:
        values = np.concatenate([frame[prop] for frame in kept]) if kept else []

    result = statistics(prop, np.asarray(values) * 1000)
    result[f"{prop}_frames"] = len(kept)

    return result


def in_domain(record: dict[str, Any], class_name: str) -> bool:
    """
    Check whether a structure class is in the domain of a model.

    Parameters
    ----------
    record
        Model metadata.
    class_name
        Structure class.

    Returns
    -------
    bool
        False only for molecule-only models on periodic classes.
    """
    return not record["molecular_only"] or class_name in MOLECULAR_CLASSES


def cohort(
    eligible: int,
    predicted_keys: dict[str, set],
    class_name: str,
    records: dict[str, dict],
) -> tuple[list[str], set]:
    """
    Get the models compared on common frames and the frames they share.

    Parameters
    ----------
    eligible
        Number of labels of the property.
    predicted_keys
        Frames each model predicted.
    class_name
        Structure class.
    records
        Model metadata.

    Returns
    -------
    tuple[list[str], set]
        Models taking part and their shared frames.
    """
    members = [
        model
        for model, keys in predicted_keys.items()
        if in_domain(records[model], class_name)
        and eligible
        and len(keys) >= COMMON_COVERAGE * eligible
    ]

    if not members:
        return [], set()

    return sorted(members), set.intersection(*[predicted_keys[m] for m in members])


def excluded_counts(statuses: collections.Counter) -> dict[str, int]:
    """
    Count the frames whose energy is not compared, by reason.

    Parameters
    ----------
    statuses
        Energy status of every frame.

    Returns
    -------
    dict[str, int]
        Frames by reason, sorted by reason.
    """
    return {
        status: count
        for status, count in sorted(statuses.items())
        if status != "compatible"
    }


def energy_text(statuses: collections.Counter, eligible: int) -> str:
    """
    Describe which frames an energy is compared on.

    Parameters
    ----------
    statuses
        Energy status of every frame.
    eligible
        Number of labels in the selection.

    Returns
    -------
    str
        Text for the table tooltip.
    """
    excluded = ", ".join(
        f"{EXCLUSION_LABELS[status]} {count}"
        for status, count in sorted(
            excluded_counts(statuses).items(), key=lambda item: -item[1]
        )
    )
    compared = f"compared on {statuses['compatible']} of {eligible} frames"

    return f"{compared}, excluded: {excluded}" if excluded else compared


def errors_tables(
    labels: list[dict[str, Any]],
    frames: dict[tuple[int, str], dict],
    records: dict[str, dict],
) -> dict[str, dict]:
    """
    Build the errors table rows of every reference and class.

    Parameters
    ----------
    labels
        All labels.
    frames
        Errors by label index and model.
    records
        Model metadata of the models shown.

    Returns
    -------
    dict[str, dict]
        Rows by selection and frame mode, and selection summaries.
    """
    groups = collections.defaultdict(list)

    for index, label in enumerate(labels):
        for class_name in (label["class"], "all"):
            groups[(label["reference"], class_name)].append(index)

    rows = {}
    summaries = {}

    for (reference, class_name), members in sorted(groups.items()):
        key = f"{reference}|{class_name}|all|all"
        per_model = {model: [frames[(i, model)] for i in members] for model in records}
        eligible = {
            "energy": len(members),
            "forces": len(members),
            "stress": sum(1 for i in members if labels[i]["stress"] is not None),
        }
        predicted_keys = {
            "energy": {
                model: {f["key"] for f in model_frames if f["energy"] is not None}
                for model, model_frames in per_model.items()
                if any(f["energy_status"] == "compatible" for f in model_frames)
            },
            "forces": {
                model: {f["key"] for f in model_frames if f["forces"] is not None}
                for model, model_frames in per_model.items()
            },
            "stress": {
                model: {f["key"] for f in model_frames if f["stress"] is not None}
                for model, model_frames in per_model.items()
            },
        }
        cohorts = {
            prop: cohort(eligible[prop], predicted_keys[prop], class_name, records)
            for prop in UNITS
        }
        summaries[key] = {
            "stress_labels": eligible["stress"],
            "common": {
                prop: {"models": len(models), "frames": len(keys)}
                for prop, (models, keys) in cohorts.items()
            },
        }

        for model, model_frames in per_model.items():
            statuses = collections.Counter(f["energy_status"] for f in model_frames)
            base = {
                "model": model,
                "energies": energy_text(statuses, len(members)),
                "comparable": statuses["compatible"] > 0,
            }
            available = dict(base)
            common = dict(base)

            for prop in UNITS:
                stats = property_statistics(model_frames, prop)
                total = statuses["compatible"] if prop == "energy" else eligible[prop]
                coverage = f"{stats[f'{prop}_frames']} / {total}"
                cohort_models, keys = cohorts[prop]
                shared = (
                    property_statistics(
                        [f for f in model_frames if f["key"] in keys], prop
                    )
                    if model in cohort_models
                    else {}
                )

                for row, values in ((available, stats), (common, shared)):
                    row[f"{prop}_coverage"] = coverage
                    for stat in ERRORS_STATISTICS:
                        row[f"{prop}_{stat}"] = values.get(
                            f"{prop}_{stat}_{UNITS[prop]}"
                        )

            rows.setdefault(f"{key}|available", []).append(available)
            rows.setdefault(f"{key}|common", []).append(common)

    return {"rows": rows, "summaries": summaries}


def comparison_entries(
    labels: list[dict[str, Any]],
    frames: dict[tuple[int, str], dict],
    records: dict[str, dict],
) -> dict[str, Any]:
    """
    Compare every model with the six references on shared frames.

    Parameters
    ----------
    labels
        All labels.
    frames
        Errors by label index and model.
    records
        Model metadata of the models shown.

    Returns
    -------
    dict[str, Any]
        One entry per structure class.
    """
    by_reference = collections.defaultdict(dict)

    for index, label in enumerate(labels):
        if label["identity"] in by_reference[label["reference"]]:
            raise ValueError(f"two labels for {label['identity']} in a reference")
        by_reference[label["reference"]][label["identity"]] = index

    entries = {}

    for class_name in COMPARISON_CLASSES:
        matched = None

        for reference in REFERENCES:
            keys = {
                identity
                for identity, index in by_reference[reference].items()
                if class_name == "all" or labels[index]["class"] == class_name
            }
            matched = keys if matched is None else matched & keys

        matched = sorted(matched)

        if not matched:
            continue

        stress_matched = [
            key
            for key in matched
            if all(
                labels[by_reference[reference][key]]["stress"] is not None
                for reference in REFERENCES
            )
        ]
        eligible = {"energy": matched, "forces": matched, "stress": stress_matched}
        success = {prop: {} for prop in UNITS}

        for model in records:
            cells = {
                key: [frames[(by_reference[ref][key], model)] for ref in REFERENCES]
                for key in matched
            }
            success["forces"][model] = {
                key
                for key in matched
                if all(f["status"] == "predicted" for f in cells[key])
            }
            success["stress"][model] = {
                key
                for key in stress_matched
                if all(f["stress"] is not None for f in cells[key])
            }

            compatible = {
                key: [f for f in cells[key] if f["energy_status"] == "compatible"]
                for key in matched
            }
            if any(compatible.values()):
                success["energy"][model] = {
                    key
                    for key in matched
                    if compatible[key]
                    and all(f["status"] == "predicted" for f in compatible[key])
                }

        cohorts = {
            prop: cohort(len(eligible[prop]), success[prop], class_name, records)
            for prop in UNITS
        }
        models = {}

        for model in records:
            cells = {"properties": {}, "references": {ref: {} for ref in REFERENCES}}

            for prop in UNITS:
                if model not in success[prop]:
                    continue

                members, common_keys = cohorts[prop]
                inside = model in members
                used = common_keys if inside else success[prop][model]
                cells["properties"][prop] = {"in_cohort": inside, "frames": len(used)}

                for reference in REFERENCES:
                    model_frames = [
                        frames[(by_reference[reference][key], model)]
                        for key in sorted(used)
                    ]
                    stats = property_statistics(model_frames, prop)
                    used_frames = stats.pop(f"{prop}_frames")
                    cell = cells["references"][reference]
                    cell.update(stats)

                    if prop == "energy":
                        cell["energy_frames"] = used_frames
                        cell["energy_excluded"] = excluded_counts(
                            collections.Counter(
                                f["energy_status"] for f in model_frames
                            )
                        )

            models[model] = cells

        entries[f"{class_name}|all|full"] = {"models": models}

    return {"entries": entries}


def force_disagreement(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compare the force labels of the references on aligned bulk frames.

    Parameters
    ----------
    labels
        All labels.

    Returns
    -------
    dict[str, Any]
        Mean absolute force difference of every pair of references, in meV/Å.
    """
    bulk = {reference: {} for reference in REFERENCES}

    for label in labels:
        if label["class"] == "bulk":
            bulk[label["reference"]][label["identity"]] = label["atoms"]

    matched = set.intersection(*[set(found) for found in bulk.values()])
    shared = []

    for key in sorted(matched):
        mad_atoms = bulk["mad/r2scan"][key]
        aligned = True

        for reference in REFERENCES:
            atoms = bulk[reference][key]

            if (
                len(atoms) != len(mad_atoms)
                or not np.array_equal(atoms.numbers, mad_atoms.numbers)
                or list(atoms.pbc) != list(mad_atoms.pbc)
                or np.abs(atoms.cell.array - mad_atoms.cell.array).max()
                > ALIGNMENT_TOLERANCE
            ):
                aligned = False
                break

            _, lengths = find_mic(
                atoms.positions - mad_atoms.positions, mad_atoms.cell, mad_atoms.pbc
            )
            if lengths.max() > ALIGNMENT_TOLERANCE:
                aligned = False
                break

        if aligned:
            shared.append(key)

    forces = {
        reference: np.concatenate(
            [bulk[reference][key].get_forces().ravel() for key in shared]
        )
        for reference in REFERENCES
    }
    matrix = {}

    for first, second in itertools.combinations(REFERENCES, 2):
        value = float(np.abs(forces[first] - forces[second]).mean() * 1000)
        matrix.setdefault(first, {})[second] = value
        matrix.setdefault(second, {})[first] = value

    return {"shared_frames": len(shared), "forces_mae_meV_per_A": matrix}


def subset_counts(labels: list[dict[str, Any]]) -> dict[str, dict]:
    """
    Count the labelled frames by reference, class and MAD subset.

    Parameters
    ----------
    labels
        All labels.

    Returns
    -------
    dict[str, dict]
        Frame counts by reference, class and subset.
    """
    counts = {}

    for label in labels:
        for class_name in (label["class"], "all"):
            subsets = counts.setdefault(label["reference"], {}).setdefault(
                class_name, {}
            )
            subsets[label["subset"]] = subsets.get(label["subset"], 0) + 1

    return counts


def placeholder_table(name: str) -> dict[str, Any]:
    """
    Get the empty metrics table the app expects.

    Parameters
    ----------
    name
        Benchmark name.

    Returns
    -------
    dict[str, Any]
        Table without rows, since nothing is scored.
    """
    return {
        "data": [],
        "columns": [{"id": "MLIP", "name": "MLIP"}],
        "thresholds": {
            "none": {"good": 0.0, "bad": 1.0, "unit": "none", "level_of_theory": name}
        },
        "weights": {"none": 0.0},
        "tooltip_header": {},
    }


def write_app_data(
    data_path: Path, calc_path: Path, out_path: Path, models: list[str]
) -> list[str]:
    """
    Score the saved predictions and write the data of both pages.

    Parameters
    ----------
    data_path
        Extracted inputs folder.
    calc_path
        Folder with one output folder per model.
    out_path
        App data folder.
    models
        Models to include, those without predictions are skipped.

    Returns
    -------
    list[str]
        Models included.
    """
    metadata = load_metadata()
    models = [m for m in models if (calc_path / m / "predictions.jsonl").exists()]
    records = {model: model_record(metadata, model) for model in sorted(models)}

    labels = load_labels(data_path)
    tables = protocol_tables(labels)
    cache = {}
    frames = {}

    for model, record in records.items():
        predictions = load_predictions(calc_path, model)
        for index, label in enumerate(labels):
            status = energy_status(record, label, tables, cache)
            frames[(index, model)] = evaluate(
                label, predictions.get(label["frame"]), record, status
            )

    references = {key: spec["name"] for key, spec in REFERENCES.items()}
    entries = {
        model: {
            "name": record["name"],
            "identifier": model,
            "checkpoint": record["checkpoint"],
            "training_data": record["training_data"],
            "molecular": record["molecular_only"],
        }
        for model, record in records.items()
    }

    files = {
        "errors/selections.json": errors_tables(labels, frames, records),
        "errors/references.json": references,
        "errors/models.json": entries,
        "errors/subsets.json": subset_counts(labels),
        "errors/errors_metrics_table.json": placeholder_table("errors"),
        "comparison/comparison.json": comparison_entries(labels, frames, records),
        "comparison/references.json": references,
        "comparison/models.json": entries,
        "comparison/disagreement.json": force_disagreement(labels),
        "comparison/comparison_metrics_table.json": placeholder_table("comparison"),
    }

    for name, content in files.items():
        path = out_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, separators=(",", ":")))

    return list(records)


def test_mad_2_bench_errors() -> None:
    """Write the app data of the MAD-2 errors and comparison pages."""
    data_path = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "mad-2-bench"
    included = write_app_data(data_path, CALC_PATH, OUT_PATH, MODELS)

    assert included, f"no saved predictions in {CALC_PATH}"
