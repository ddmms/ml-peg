"""Tests for geometry-optimization metrics, schemas, and I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.bulk_crystal.geo_opt.analyse_geo_opt import (
    _json_safe_mapping,
    _json_safe_records,
)
from ml_peg.analysis.bulk_crystal.geo_opt.io import (
    read_analysis_csv,
    read_geo_opt_records,
    write_analysis_csv,
    write_geo_opt_jsonl,
)
from ml_peg.analysis.bulk_crystal.geo_opt.metrics import (
    N_STRUCTURES,
    N_SYM_OPS_MAE,
    SYMMETRY_DECREASE,
    SYMMETRY_INCREASE,
    SYMMETRY_MATCH,
    calc_geo_opt_metrics,
)
from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    ANGLE_TOLERANCE,
    COMPARISON_FIELDS,
    CONVERGED,
    ENERGY,
    HALL_NUM,
    HALL_SYMBOL,
    INTERNATIONAL_SPG_NAME,
    MATERIAL_ID,
    MAX_PAIR_DIST,
    N_ROT_SYMS,
    N_STEPS,
    N_SYM_OPS,
    N_SYM_OPS_DIFF,
    N_TRANS_SYMS,
    SITE_SYMMETRY_SYMBOLS,
    SPG_NUM,
    SPG_NUM_DIFF,
    STRUCTURE,
    STRUCTURE_RMSD_VS_DFT,
    SYMPREC,
    WYCKOFF_SYMBOLS,
    validate_analysis_dataframe,
    validate_geo_opt_dataframe,
)

pytestmark = pytest.mark.framework("matbench-discovery")


def _geo_opt_record(
    material_id: str = "wbm-1",
    **overrides: object,
) -> dict[str, object]:
    """Return one valid geo-opt record with optional overrides."""
    record: dict[str, object] = {
        MATERIAL_ID: material_id,
        STRUCTURE: {"lattice": {"matrix": []}, "sites": []},
        ENERGY: -1.25,
        CONVERGED: True,
        N_STEPS: 12,
    }
    record.update(overrides)
    return record


def _analysis_dataframe() -> pd.DataFrame:
    """Return one valid per-structure symmetry comparison table."""
    return pd.DataFrame(
        {
            MATERIAL_ID: ["wbm-1"],
            SPG_NUM: [221],
            HALL_NUM: [517],
            INTERNATIONAL_SPG_NAME: ["P m -3 m"],
            SITE_SYMMETRY_SYMBOLS: [["m-3m"]],
            WYCKOFF_SYMBOLS: [["a"]],
            N_SYM_OPS: [48],
            N_ROT_SYMS: [48],
            N_TRANS_SYMS: [48],
            HALL_SYMBOL: ["-P 4 2 3"],
            SYMPREC: [1e-2],
            ANGLE_TOLERANCE: [None],
            SPG_NUM_DIFF: [0],
            N_SYM_OPS_DIFF: [0],
            STRUCTURE_RMSD_VS_DFT: [0.0],
            MAX_PAIR_DIST: [0.0],
        }
    )


@pytest.mark.parametrize(
    "non_finite_value",
    [np.float32(np.nan), np.float64(np.inf), np.float32(-np.inf)],
    ids=["float32-nan", "float64-positive-inf", "float32-negative-inf"],
)
def test_json_safe_mapping_sanitizes_numpy_non_finite_scalars(
    non_finite_value: np.floating,
) -> None:
    """Convert non-finite NumPy floating scalars to JSON nulls."""
    result = _json_safe_mapping({"metric": non_finite_value})

    assert result == {"metric": None}
    assert json.dumps(result, allow_nan=False) == '{"metric": null}'


def test_json_safe_records_preserves_float_precision() -> None:
    """Preserve benchmark-scale float precision in JSON result records."""
    value = 1.2345678901234

    assert _json_safe_records(pd.DataFrame({"value": [value]}))[0]["value"] == value


@pytest.mark.parametrize(
    (
        "rmsd_values",
        "spg_num_diffs",
        "n_sym_ops_diffs",
        "expected_metrics",
    ),
    [
        ([0.1] * 3, [0, 0, 0], [0, 0, 0], (0.1, 0, 0, 1, 0, 3)),
        ([0.1] * 3, [-1, -2, -1], [-2, -4, -2], (0.1, 8 / 3, 1, 0, 0, 3)),
        ([0.1] * 3, [1, 2, 1], [2, 4, 2], (0.1, 8 / 3, 0, 0, 1, 3)),
        (
            [0.1] * 3,
            [0, -1, 1],
            [0, -2, 2],
            (0.1, 4 / 3, 1 / 3, 1 / 3, 1 / 3, 3),
        ),
        ([0.1] * 3, [1, -1, 0], [0, 0, 0], (0.1, 0, 0, 1 / 3, 0, 3)),
        ([0.1] * 3, [0, np.nan, 1], [0, np.nan, 2], (0.1, 1, 0, 0.5, 0.5, 2)),
        (
            [0.1, "0.2", "invalid", None],
            [0, 1, -1, np.nan],
            [0, 2, -4, np.nan],
            (0.575, 2, 1 / 3, 1 / 3, 1 / 3, 3),
        ),
        (
            [None, 0.2],
            [np.nan, np.nan],
            [np.nan, np.nan],
            (0.6, np.nan, np.nan, np.nan, np.nan, 0),
        ),
    ],
)
def test_calc_geo_opt_metrics(
    rmsd_values: list[object],
    spg_num_diffs: list[float],
    n_sym_ops_diffs: list[float],
    expected_metrics: tuple[float, ...],
) -> None:
    """Calculate all metrics across valid, invalid, and missing inputs."""
    dataframe = pd.DataFrame(
        {
            STRUCTURE_RMSD_VS_DFT: rmsd_values,
            SPG_NUM_DIFF: spg_num_diffs,
            N_SYM_OPS_DIFF: n_sym_ops_diffs,
        }
    )
    metrics = calc_geo_opt_metrics(dataframe)
    actual_metrics = (
        metrics[STRUCTURE_RMSD_VS_DFT],
        metrics[N_SYM_OPS_MAE],
        metrics[SYMMETRY_DECREASE],
        metrics[SYMMETRY_MATCH],
        metrics[SYMMETRY_INCREASE],
        metrics[N_STRUCTURES],
    )
    assert actual_metrics == pytest.approx(expected_metrics, nan_ok=True)


@pytest.mark.parametrize(
    "invalid_rmsd",
    [np.inf, -np.inf, -0.1],
    ids=["positive-infinity", "negative-infinity", "negative-value"],
)
def test_calc_geo_opt_metrics_penalizes_invalid_numeric_rmsd(
    invalid_rmsd: float,
) -> None:
    """Replace invalid numeric RMSDs with the 1.0 matching penalty."""
    dataframe = pd.DataFrame(
        {
            STRUCTURE_RMSD_VS_DFT: [invalid_rmsd, 0.2],
            SPG_NUM_DIFF: [0, 0],
            N_SYM_OPS_DIFF: [0, 0],
        }
    )

    metrics = calc_geo_opt_metrics(dataframe)

    assert metrics[STRUCTURE_RMSD_VS_DFT] == pytest.approx(0.6)


def test_calc_geo_opt_metrics_rejects_missing_columns() -> None:
    """Reject metric tables lacking a required comparison field."""
    with pytest.raises(ValueError, match="n_sym_ops_diff"):
        calc_geo_opt_metrics(
            pd.DataFrame(
                {
                    STRUCTURE_RMSD_VS_DFT: [0.0],
                    SPG_NUM_DIFF: [0],
                }
            )
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({STRUCTURE: []}, "structure must be a dictionary"),
        ({ENERGY: float("inf")}, "energy must be finite"),
        ({CONVERGED: 1}, "converged must be boolean"),
        ({N_STEPS: -1}, "n_steps must be a non-negative integer"),
        ({MATERIAL_ID: ""}, "material_id must be a non-empty string"),
    ],
)
def test_validate_geo_opt_dataframe_rejects_invalid_records(
    overrides: dict[str, object], message: str
) -> None:
    """Reject invalid record values with field-specific errors."""
    with pytest.raises(ValueError, match=message):
        validate_geo_opt_dataframe(pd.DataFrame([_geo_opt_record(**overrides)]))


def test_validate_geo_opt_dataframe_rejects_duplicate_ids() -> None:
    """Reject duplicate material identifiers."""
    dataframe = pd.DataFrame([_geo_opt_record(), _geo_opt_record()])

    with pytest.raises(ValueError, match="Duplicate material_id"):
        validate_geo_opt_dataframe(dataframe)


def test_validate_analysis_dataframe_rejects_schema_failures() -> None:
    """Reject missing and non-numeric per-structure analysis fields."""
    missing_comparison = _analysis_dataframe().drop(columns=COMPARISON_FIELDS[0])
    with pytest.raises(ValueError, match=COMPARISON_FIELDS[0]):
        validate_analysis_dataframe(missing_comparison)

    non_numeric = _analysis_dataframe().astype({N_SYM_OPS: object})
    non_numeric.loc[0, N_SYM_OPS] = "many"
    with pytest.raises(ValueError, match="must be numeric"):
        validate_analysis_dataframe(non_numeric)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (HALL_SYMBOL, 123, "must contain strings"),
        (SYMPREC, np.inf, "numeric and finite"),
        (ANGLE_TOLERANCE, np.inf, "numeric and finite"),
        (SPG_NUM, 1.5, "must contain integers"),
        (N_SYM_OPS, -1, "must be non-negative"),
    ],
)
def test_validate_analysis_dataframe_rejects_impossible_values(
    field: str, value: object, match: str
) -> None:
    """Reject semantically impossible symmetry-analysis values."""
    dataframe = _analysis_dataframe()
    dataframe[field] = value

    with pytest.raises(ValueError, match=match):
        validate_analysis_dataframe(dataframe)


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_geo_opt_jsonl_round_trip(tmp_path: Path, suffix: str) -> None:
    """Round-trip records through compressed and plain JSONL."""
    output_path = tmp_path / f"geo-opt{suffix}"
    records = [
        _geo_opt_record("001", energy=1.2345678901234),
        _geo_opt_record("wbm-2", n_steps=0),
    ]

    write_geo_opt_jsonl(records, output_path)
    loaded_records = read_geo_opt_records(output_path)

    assert loaded_records == records
    assert isinstance(loaded_records[0][STRUCTURE], dict)


def test_geo_opt_jsonl_writer_omits_noncanonical_columns(tmp_path: Path) -> None:
    """JSONL output excludes unrelated DataFrame columns."""
    output_path = tmp_path / "geo-opt.jsonl"
    dataframe = pd.DataFrame([_geo_opt_record() | {"extra": "discard me"}])

    write_geo_opt_jsonl(dataframe, output_path)
    stored = pd.read_json(output_path, lines=True)

    assert list(stored.columns) == [
        MATERIAL_ID,
        STRUCTURE,
        ENERGY,
        CONVERGED,
        N_STEPS,
    ]


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_analysis_csv_round_trip(tmp_path: Path, suffix: str) -> None:
    """Round-trip JSON-encoded list columns in analysis CSVs."""
    output_path = tmp_path / f"geo-opt-analysis{suffix}"
    dataframe = _analysis_dataframe()
    dataframe[MATERIAL_ID] = "001"

    write_analysis_csv(dataframe, output_path)
    loaded = read_analysis_csv(output_path)

    assert loaded.index.tolist() == ["001"]
    assert loaded.index.name == MATERIAL_ID
    assert loaded.loc["001", SPG_NUM] == 221
    assert loaded.loc["001", STRUCTURE_RMSD_VS_DFT] == pytest.approx(0.0)
    assert loaded.loc["001", INTERNATIONAL_SPG_NAME] == "P m -3 m"
    assert loaded.loc["001", HALL_SYMBOL] == "-P 4 2 3"
    assert loaded.loc["001", SITE_SYMMETRY_SYMBOLS] == ["m-3m"]
    assert loaded.loc["001", WYCKOFF_SYMBOLS] == ["a"]
    assert loaded.loc["001", ANGLE_TOLERANCE] is None
    assert set(loaded.columns) == set(_analysis_dataframe().columns) - {MATERIAL_ID}


def test_read_analysis_csv_rejects_malformed_list_cells(tmp_path: Path) -> None:
    """Raise a schema error for malformed list values."""
    output_path = tmp_path / "bad-analysis.csv"
    dataframe = _analysis_dataframe()
    dataframe[SITE_SYMMETRY_SYMBOLS] = "["
    dataframe.to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="Invalid list encoding"):
        read_analysis_csv(output_path)
