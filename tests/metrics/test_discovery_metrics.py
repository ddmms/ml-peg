"""Tests for materials-discovery metrics and artifact handling."""

from __future__ import annotations

from collections.abc import Callable
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.bulk_crystal.materials_discovery import (
    E_ABOVE_HULL,
    MATERIAL_ID,
    PREDICTED_FORMATION_ENERGY,
    REFERENCE_FORMATION_ENERGY,
    UNIQUE_PROTOTYPE,
    DiscoverySubset,
    align_predictions,
    calc_discovery_metrics,
    classify_stable,
    discovery_subset_indices,
    evaluate_discovery,
    evaluate_discovery_paths,
    stable_metrics,
    validate_prediction_frame,
    validate_reference_frame,
    write_discovery_metrics_json,
)
from ml_peg.data.artifacts import (
    ArtifactRole,
    artifact_filename,
    canonical_scientific_notation,
    parse_artifact_filename,
    read_csv_artifact,
    read_jsonl_artifact,
)

REPORTED_METRICS = set(
    "F1 DAF Precision Recall Accuracy TPR FPR TNR FNR TP FP TN FN MAE RMSE R2 "
    "missing_preds".split()
)
DAF_SUBSETS = (
    DiscoverySubset.unique_prototypes,
    DiscoverySubset.most_stable_10k,
)


def _reference_frame(
    *,
    material_ids: list[str] | None = None,
    each_true: list[float] | None = None,
    formation_energies: list[float] | None = None,
    unique_prototypes: list[bool | float | int] | None = None,
) -> pd.DataFrame:
    """Build a compact valid discovery reference dataframe."""
    resolved_ids = material_ids or ["wbm-0", "wbm-1", "wbm-2", "wbm-3"]
    row_count = len(resolved_ids)
    return pd.DataFrame(
        {
            MATERIAL_ID: resolved_ids,
            E_ABOVE_HULL: each_true or [-1.0, -0.5, 0.5, 1.0][:row_count],
            REFERENCE_FORMATION_ENERGY: formation_energies or [0.0] * row_count,
            UNIQUE_PROTOTYPE: unique_prototypes or [True] * row_count,
        }
    )


def _prediction_frame(
    material_ids: list[str], values: list[float | None]
) -> pd.DataFrame:
    """Build a valid discovery prediction dataframe."""
    return pd.DataFrame({MATERIAL_ID: material_ids, PREDICTED_FORMATION_ENERGY: values})


def test_stable_metrics_exact_values() -> None:
    """Classification and regression metrics preserve exact source semantics."""
    metrics = stable_metrics(
        [-1.0, -0.5, 0.5, 1.0],
        [-0.8, 0.2, -0.1, 0.9],
    )
    expected = {
        "F1": 0.5,
        "DAF": 1.0,
        "Precision": 0.5,
        "Recall": 0.5,
        "Accuracy": 0.5,
        "TPR": 0.5,
        "FPR": 0.5,
        "TNR": 0.5,
        "FNR": 0.5,
        "TP": 1,
        "FP": 1,
        "TN": 1,
        "FN": 1,
        "MAE": 0.4,
        "RMSE": math.sqrt(0.225),
        "R2": 0.64,
    }
    assert metrics == pytest.approx(expected)


def test_stable_metrics_pairs_series_by_position() -> None:
    """Ignore Series labels when pairing true and predicted values."""
    each_true = pd.Series([-1.0, -0.5, 0.5, 1.0], index=[10, 11, 12, 13])
    each_pred = pd.Series([-0.8, -0.2, 0.1, 0.9], index=[13, 12, 11, 10])

    indexed_metrics = stable_metrics(each_true, each_pred)
    positional_metrics = stable_metrics(each_true.tolist(), each_pred.tolist())

    assert indexed_metrics == pytest.approx(positional_metrics)
    assert (
        indexed_metrics["TP"],
        indexed_metrics["FN"],
        indexed_metrics["FP"],
        indexed_metrics["TN"],
    ) == (2, 0, 0, 2)


@pytest.mark.parametrize(
    ("each_true", "each_pred", "fillna", "expected_counts"),
    [
        (
            [-0.1, 0.0, 0.1, np.nan, None],
            [-0.1, 0.0, 0.1, 0.2, -0.2],
            True,
            (2, 0, 0, 1),
        ),
        ([0.0, -0.1], [None, 0.1], True, (0, 2, 0, 0)),
        ([-0.1, 0.1, np.nan], [-0.1, np.nan, -0.2], False, (1, 0, 0, 0)),
    ],
    ids=["nullable-truth", "missing-predictions", "no-fill"],
)
def test_classify_stable_handles_nans(
    each_true: list[float | None],
    each_pred: list[float | None],
    fillna: bool,
    expected_counts: tuple[int, int, int, int],
) -> None:
    """Nullable values retain established classification behavior."""
    masks = classify_stable(each_true, each_pred, fillna=fillna)
    assert tuple(int(mask.sum()) for mask in masks) == expected_counts


@pytest.mark.parametrize(
    ("each_true", "each_pred", "expected"),
    [
        (
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            (0.0, 1.0, 0.0, np.nan, np.nan, np.nan, np.nan),
        ),
        (
            [-0.1, -0.2, -0.3],
            [0.1, 0.2, 0.3],
            (np.nan, np.nan, np.nan, 0.0, 1.0, np.nan, np.nan),
        ),
    ],
    ids=["no-stable-class", "no-unstable-class"],
)
def test_stable_metrics_zero_classes(
    each_true: list[float],
    each_pred: list[float],
    expected: tuple[float, ...],
) -> None:
    """Absent positive or negative classes produce the expected NaN rates."""
    metrics = stable_metrics(each_true, each_pred)
    metric_names = ("Precision", "FPR", "TNR", "Recall", "FNR", "DAF", "F1")
    assert tuple(metrics[name] for name in metric_names) == pytest.approx(
        expected, nan_ok=True
    )


def test_stable_metrics_handles_empty_regression_pairs_without_warnings() -> None:
    """All-missing regression pairs return NaNs without runtime warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = stable_metrics([None, np.nan], [None, np.nan])

    assert all(math.isnan(float(metrics[name])) for name in ("MAE", "RMSE", "R2"))


def test_fillna_changes_classification_but_not_regression_metrics() -> None:
    """Missing-value classification does not alter valid regression pairs."""
    filled = stable_metrics([-0.1, 0.1], [None, 0.2], fillna=True)
    unfilled = stable_metrics([-0.1, 0.1], [None, 0.2], fillna=False)

    assert filled["FN"] == 1
    assert unfilled["FN"] == 0
    assert {
        metric_name: filled[metric_name] for metric_name in ("MAE", "RMSE", "R2")
    } == pytest.approx(
        {
            "MAE": unfilled["MAE"],
            "RMSE": unfilled["RMSE"],
            "R2": unfilled["R2"],
        },
        nan_ok=True,
    )


def test_single_regression_pair_has_nan_r2() -> None:
    """R2 is undefined for one valid prediction pair."""
    assert math.isnan(float(stable_metrics([0.1], [0.2])["R2"]))


@pytest.mark.parametrize(
    ("threshold", "expected_counts"),
    [(-0.1, (1, 0, 0, 2)), (0.0, (2, 0, 0, 1)), (0.1, (3, 0, 0, 0))],
)
def test_classify_stable_thresholds(
    threshold: float, expected_counts: tuple[int, int, int, int]
) -> None:
    """Apply the requested stability threshold to both arrays."""
    masks = classify_stable(
        [-0.2, 0.0, 0.1],
        [-0.2, 0.0, 0.1],
        stability_threshold=threshold,
    )
    assert tuple(int(mask.sum()) for mask in masks) == expected_counts


@pytest.mark.parametrize(
    "invalid_threshold",
    [None, np.nan, np.inf, -np.inf],
    ids=["none", "nan", "positive-infinity", "negative-infinity"],
)
def test_classify_stable_rejects_nonfinite_thresholds(
    invalid_threshold: float | None,
) -> None:
    """Reject missing and non-finite stability thresholds."""
    with pytest.raises(ValueError, match="stability_threshold must be a real number"):
        classify_stable(
            [-0.1, 0.1],
            [-0.1, 0.1],
            stability_threshold=invalid_threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("validator", "missing_column"),
    [
        (validate_reference_frame, UNIQUE_PROTOTYPE),
        (validate_prediction_frame, PREDICTED_FORMATION_ENERGY),
    ],
    ids=["reference", "predictions"],
)
def test_discovery_schema_validation(
    validator: Callable[[pd.DataFrame], None],
    missing_column: str,
) -> None:
    """Schema validators accept indexed IDs and reject missing columns."""
    reference = _reference_frame().set_index(MATERIAL_ID)
    predictions = _prediction_frame(
        reference.index.tolist(), [0.0] * len(reference)
    ).set_index(MATERIAL_ID)
    dataframe = reference if validator is validate_reference_frame else predictions
    validator(dataframe)

    with pytest.raises(ValueError, match="missing required columns"):
        validator(dataframe.drop(columns=missing_column))


@pytest.mark.parametrize(
    ("column", "invalid_values", "match"),
    [
        (
            MATERIAL_ID,
            ["wbm-0", "wbm-0", "wbm-2", "wbm-3"],
            "duplicate.*material_id",
        ),
        (
            MATERIAL_ID,
            [1, "1", 2, 3],
            "duplicate.*material_id.*string conversion",
        ),
        (
            UNIQUE_PROTOTYPE,
            ["False", "True", "False", "True"],
            "values must be boolean",
        ),
        (UNIQUE_PROTOTYPE, [True, None, False, True], "values must be boolean"),
        (E_ABOVE_HULL, [np.inf, -0.5, 0.5, 1.0], "values must be finite"),
        (
            E_ABOVE_HULL,
            pd.array([-1.0, pd.NA, 0.5, 1.0], dtype="Float64"),
            "values must be finite",
        ),
        (
            REFERENCE_FORMATION_ENERGY,
            [0.0, np.nan, 0.0, 0.0],
            "values must be finite",
        ),
    ],
)
def test_discovery_schema_rejects_invalid_values(
    column: str,
    invalid_values: object,
    match: str,
) -> None:
    """Reference fields reject duplicate IDs, non-booleans, and nonfinite energies."""
    reference = _reference_frame()
    reference[column] = invalid_values

    with pytest.raises(ValueError, match=match):
        validate_reference_frame(reference)


def test_discovery_schema_rejects_inconsistent_id_locations() -> None:
    """Material IDs in a column and named index must agree."""
    reference = _reference_frame().set_index(MATERIAL_ID, drop=False)
    reference.index = pd.Index(
        ["other-0", "other-1", "other-2", "other-3"], name=MATERIAL_ID
    )
    with pytest.raises(ValueError, match="inconsistent.*column and index"):
        validate_reference_frame(reference)


@pytest.mark.parametrize(
    "prototype_flags",
    [[1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]],
    ids=["integers", "floats"],
)
def test_discovery_schema_accepts_binary_prototype_flags(
    prototype_flags: list[float | int],
) -> None:
    """CSV schemas accept zero/one prototype flags."""
    reference = _reference_frame(unique_prototypes=prototype_flags)

    validate_reference_frame(reference)
    unique_index = discovery_subset_indices(
        reference,
        pd.Series([0.0] * len(reference), index=reference[MATERIAL_ID]),
    )[DiscoverySubset.unique_prototypes]
    assert unique_index.tolist() == ["wbm-0", "wbm-2"]


def test_artifact_readers_support_gzip_csv_and_jsonl(tmp_path: Path) -> None:
    """Artifact readers infer gzip compression for CSV and JSONL inputs."""
    dataframe = pd.DataFrame({"material_id": ["wbm-0", "wbm-1"], "value": [1, 2]})
    csv_path = tmp_path / "artifact.csv.gz"
    jsonl_path = tmp_path / "artifact.jsonl.gz"
    dataframe.to_csv(csv_path, index=False)
    dataframe.to_json(jsonl_path, orient="records", lines=True, compression="gzip")

    pd.testing.assert_frame_equal(read_csv_artifact(csv_path), dataframe)
    pd.testing.assert_frame_equal(read_jsonl_artifact(jsonl_path), dataframe)


@pytest.mark.parametrize(
    ("value", "expected"), [(1e-5, "1e-5"), ("0.0100", "1e-2"), (2.5, "2.5e0")]
)
def test_canonical_scientific_notation(value: float | str, expected: str) -> None:
    """Scientific notation is normalized for artifact names."""
    assert canonical_scientific_notation(value) == expected


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (ArtifactRole.discovery, "2026-07-18-discovery.csv.gz"),
        (ArtifactRole.geo_opt, "2026-07-18-geo-opt.jsonl.gz"),
        (
            ArtifactRole.geo_opt_analysis,
            "2026-07-18-geo-opt-symprec=1e-5-moyo=0.12.0.csv.gz",
        ),
    ],
)
def test_artifact_names_round_trip(role: ArtifactRole, expected: str) -> None:
    """Dated artifact names round-trip for every role."""
    filename = (
        artifact_filename("2026-07-18", role, symprec=1e-5, moyo_version="0.12.0")
        if role is ArtifactRole.geo_opt_analysis
        else artifact_filename("2026-07-18", role)
    )
    assert filename == expected
    assert parse_artifact_filename(f"/tmp/{filename}") is role


@pytest.mark.parametrize("invalid_date", ["2026-02-30", "2026/07/18"])
def test_artifact_filename_rejects_invalid_dates(invalid_date: str) -> None:
    """Artifact names require real ISO calendar dates."""
    with pytest.raises(ValueError, match="date"):
        artifact_filename(invalid_date, ArtifactRole.discovery)


def test_prediction_alignment_rejects_unknown_ids_and_fills_missing() -> None:
    """Alignment rejects extraneous IDs and inserts NaN for omitted references."""
    reference = _reference_frame()
    reference_ids = reference[MATERIAL_ID].tolist()
    aligned = align_predictions(reference, pd.Series([0.2], index=[reference_ids[1]]))
    assert aligned.index.tolist() == reference_ids
    assert aligned.iloc[1] == pytest.approx(0.2)
    assert aligned.isna().sum() == 3

    with pytest.raises(ValueError, match="unknown material IDs.*unknown"):
        align_predictions(reference, pd.Series([0.2], index=["unknown"]))


def test_dataframe_evaluation_normalizes_material_ids_to_strings() -> None:
    """Match numeric reference IDs with string prediction IDs."""
    reference = _reference_frame()
    reference[MATERIAL_ID] = [1, 2, 3, 4]
    predictions = _prediction_frame(["1", "2", "3", "4"], [-0.8, 0.2, -0.1, 0.9])

    results = evaluate_discovery(reference, predictions)

    assert results[str(DiscoverySubset.full_test_set)]["missing_preds"] == 0


@pytest.mark.parametrize("ranking_case", ["ties", "missing"])
def test_most_stable_10k_ranking(ranking_case: str) -> None:
    """Stable sorting preserves ties and places missing predictions last."""
    material_ids = [f"wbm-{idx}" for idx in range(10_001)]
    reference = _reference_frame(
        material_ids=material_ids,
        each_true=[0.0] * len(material_ids),
        formation_energies=[0.0] * len(material_ids),
        unique_prototypes=[True] * len(material_ids),
    )
    if ranking_case == "ties":
        prediction_values = [0.0] * len(material_ids)
    else:
        prediction_values = [
            *map(float, range(9_999)),
            np.nan,
            np.nan,
        ]
    predictions = pd.Series(prediction_values, index=material_ids)

    ranked_index = discovery_subset_indices(reference, predictions)[
        DiscoverySubset.most_stable_10k
    ]
    assert ranked_index.tolist() == material_ids[:10_000]
    if ranking_case == "missing":
        assert pd.isna(predictions.loc[ranked_index[-1]])


def test_most_stable_ranking_uses_predicted_hull_distance() -> None:
    """Ranking includes DFT hull and formation-energy offsets, not raw predictions."""
    material_ids = ["wbm-a", "wbm-b", "wbm-c", "wbm-d"]
    reference = _reference_frame(
        material_ids=material_ids,
        each_true=[1.0, 0.0, 0.5, 0.2],
        formation_energies=[0.0] * 4,
        unique_prototypes=[True] * 4,
    )
    predictions = pd.Series([-2.0, -0.1, -0.5, -0.2], index=material_ids)

    ranked_index = discovery_subset_indices(reference, predictions)[
        DiscoverySubset.most_stable_10k
    ]

    assert ranked_index.tolist() == material_ids
    assert not ranked_index.equals(predictions.sort_values().index)


def test_subset_metrics_and_daf_override() -> None:
    """Subset selection uses the given DAF prevalence."""
    reference = _reference_frame(
        material_ids=[f"wbm-{idx}" for idx in range(6)],
        each_true=[-0.2, -0.1, 0.1, 0.2, -0.05, 0.3],
        formation_energies=[-1.0, -0.9, -0.8, -0.7, -0.6, -0.5],
        unique_prototypes=[True, True, False, True, True, False],
    )
    predictions = pd.Series(
        [-1.1, -0.7, -0.9, -0.6, np.nan, -0.4],
        index=reference[MATERIAL_ID],
    )
    subset_indices = discovery_subset_indices(reference, predictions)
    metrics = calc_discovery_metrics(
        reference,
        predictions,
        subset_indices=subset_indices,
        uniq_proto_prevalence=0.5,
    )

    assert set(metrics) == set(DiscoverySubset)
    assert subset_indices[DiscoverySubset.most_stable_10k].tolist() == [
        "wbm-0",
        "wbm-1",
        "wbm-3",
        "wbm-4",
    ]
    for subset in DAF_SUBSETS:
        assert metrics[subset]["DAF"] == pytest.approx(
            metrics[subset]["Precision"] / 0.5
        )
    full_metrics = metrics[DiscoverySubset.full_test_set]
    full_prevalence = 3 / 6
    assert full_metrics["DAF"] == pytest.approx(
        full_metrics["Precision"] / full_prevalence
    )


def test_calc_discovery_metrics_rejects_incomplete_subset_indices() -> None:
    """Require all three subset indices when callers provide them."""
    reference = _reference_frame()
    predictions = pd.Series([0.0] * len(reference), index=reference[MATERIAL_ID])

    with pytest.raises(ValueError, match="Missing required discovery subsets"):
        calc_discovery_metrics(
            reference,
            predictions,
            subset_indices={DiscoverySubset.full_test_set: pd.Index([])},
        )


def test_evaluation_masks_outliers_then_rounds_to_three_decimals() -> None:
    """Evaluation masks errors above 5 eV and rounds inputs before metrics."""
    reference = _reference_frame(
        material_ids=["wbm-a", "wbm-b", "wbm-c"],
        each_true=[0.0004, -0.2, 0.2],
        formation_energies=[-1.0, 0.0, 0.0],
        unique_prototypes=[True, True, True],
    )
    predictions = _prediction_frame(
        ["wbm-a", "wbm-b", "wbm-c"],
        [-0.9996, 5.0001, 5.0],
    )

    results = evaluate_discovery(reference, predictions)
    full_metrics = results[str(DiscoverySubset.full_test_set)]
    assert set(full_metrics) == REPORTED_METRICS
    assert {
        metric_name: full_metrics[metric_name]
        for metric_name in ("TP", "FN", "FP", "TN", "missing_preds")
    } == {"TP": 1, "FN": 1, "FP": 0, "TN": 1, "missing_preds": 1}
    assert {name: full_metrics[name] for name in ("MAE", "RMSE", "R2")} == {
        "MAE": 2.5,
        "RMSE": 3.536,
        "R2": -1249.0,
    }


def test_synthetic_daf_uses_prepared_rounded_hull_labels() -> None:
    """Synthetic-mode prevalence follows the documented rounded evaluation labels."""
    reference = _reference_frame(
        material_ids=["wbm-a", "wbm-b"],
        each_true=[0.0004, 0.1],
        formation_energies=[0.0, 0.0],
        unique_prototypes=[True, True],
    )
    predictions = _prediction_frame(["wbm-a", "wbm-b"], [0.0004, 0.1])

    results = evaluate_discovery(reference, predictions)

    assert results[str(DiscoverySubset.unique_prototypes)]["DAF"] == 2.0


def test_canonical_evaluation_requires_explicit_unrounded_prevalence() -> None:
    """Leaderboard mode requires unrounded unique-prototype prevalence."""
    reference = _reference_frame()
    predictions = _prediction_frame(
        reference[MATERIAL_ID].tolist(), [-0.8, 0.2, -0.1, 0.9]
    )
    with pytest.raises(ValueError, match="requires explicit unrounded"):
        evaluate_discovery(reference, predictions, canonical=True)

    results = evaluate_discovery(
        reference,
        predictions,
        canonical=True,
        uniq_proto_prevalence=0.25,
    )
    for subset in DAF_SUBSETS:
        assert results[str(subset)]["DAF"] == 4.0


def test_path_evaluation_is_json_safe_and_writes_strict_json(
    tmp_path: Path,
) -> None:
    """Gzip path evaluation replaces NaNs by null and writes strict JSON."""
    reference = _reference_frame(
        material_ids=["007", "008", "009", "010"],
        each_true=[-0.4, -0.3, -0.2, -0.1],
    )
    predictions = _prediction_frame(
        reference[MATERIAL_ID].tolist(), [1.0, 1.0, 1.0, None]
    )
    reference_path = tmp_path / "reference.csv.gz"
    prediction_path = tmp_path / "predictions.csv.gz"
    output_path = tmp_path / "metrics.json"
    reference.to_csv(reference_path, index=False)
    predictions.to_csv(prediction_path, index=False)

    results = evaluate_discovery_paths(reference_path, prediction_path)
    assert results[str(DiscoverySubset.full_test_set)]["Precision"] is None
    assert results[str(DiscoverySubset.full_test_set)]["missing_preds"] == 1
    json.dumps(results, allow_nan=False)

    write_discovery_metrics_json(results, output_path)
    with open(output_path, encoding="utf-8") as file:
        assert json.load(file) == results


def test_evaluators_treat_infinite_predictions_as_missing() -> None:
    """Raw and JSON-safe evaluators sanitize infinite predictions."""
    reference = _reference_frame()
    predictions = pd.Series([np.inf, -np.inf, 0.5, 1.0], index=reference[MATERIAL_ID])

    results = evaluate_discovery(reference, predictions, max_error_threshold=None)
    full_metrics = results[str(DiscoverySubset.full_test_set)]
    assert full_metrics["missing_preds"] == 2
    json.dumps(results, allow_nan=False)

    metrics = calc_discovery_metrics(reference, predictions)
    assert metrics[DiscoverySubset.full_test_set]["FN"] == 2
