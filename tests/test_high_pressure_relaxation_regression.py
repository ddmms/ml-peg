"""Regression checks for high-pressure relaxation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PRESSURE_LABELS = ["P000", "P025", "P050", "P075", "P100", "P125", "P150"]
FIRST_N = 5
VOLUME_ATOL = 0.001
ENERGY_ATOL = 0.0001

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_ROOT = REPO_ROOT / "tests" / "data" / "high_pressure_relaxation"
OUTPUT_ROOT = (
    REPO_ROOT
    / "ml_peg"
    / "calcs"
    / "bulk_crystal"
    / "high_pressure_relaxation"
    / "outputs"
)

TEST_CASES = [
    {
        "name": "mace-mpa-0",
        "output_model": "mace-mpa-0",
        "data_file": TEST_DATA_ROOT / "mace-mpa-0_first5.csv",
    },
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[case["name"] for case in TEST_CASES])
@pytest.mark.parametrize("pressure_label", PRESSURE_LABELS)
def test_high_pressure_relaxation_regression(
    case: dict[str, Path | str],
    pressure_label: str,
) -> None:
    """Check entries against stored regression baselines."""
    data_path = Path(case["data_file"])
    results_path = (
        OUTPUT_ROOT / str(case["output_model"]) / f"results_{pressure_label}.csv"
    )

    assert data_path.exists(), f"Missing test data file: {data_path}"
    assert results_path.exists(), f"Missing results file: {results_path}"

    expected = pd.read_csv(data_path)
    expected_pressure = expected[expected["pressure_label"] == pressure_label]

    assert len(expected_pressure) == FIRST_N

    results = pd.read_csv(results_path).head(FIRST_N)
    if len(results) < FIRST_N:
        pytest.skip(
            f"{case['output_model']} only has {len(results)} rows for {pressure_label}."
        )

    assert expected_pressure["mat_id"].tolist() == results["mat_id"].tolist()
    np.testing.assert_allclose(
        results["pred_volume_per_atom"].to_numpy(),
        expected_pressure["volume"].to_numpy(),
        rtol=0.0,
        atol=VOLUME_ATOL,
    )
    np.testing.assert_allclose(
        results["pred_energy_per_atom"].to_numpy(),
        expected_pressure["energy_per_atom"].to_numpy(),
        rtol=0.0,
        atol=ENERGY_ATOL,
    )
