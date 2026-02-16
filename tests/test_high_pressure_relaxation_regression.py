"""Regression checks for high-pressure relaxation outputs."""

from __future__ import annotations

from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_peg.calcs.bulk_crystal.high_pressure_relaxation.calc_high_pressure_relaxation import (  # noqa: E501
    PRESSURE_LABELS,
    PRESSURES,
    load_structures,
    relax_with_pressure,
)
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

N_TEST_STRUCTURES = 3
VOLUME_ATOL = 0.001
ENERGY_ATOL = 0.0001

TEST_DATA_DIR = Path(__file__).parent / "data" / "high_pressure_relaxation"
MODEL_NAME = "mace-mpa-0"
REFERENCE_FILE = TEST_DATA_DIR / f"{MODEL_NAME}_first{N_TEST_STRUCTURES}.csv"


@pytest.fixture(scope="module")
def regression_results() -> pd.DataFrame:
    """Run relaxation on first N test structures for all pressures."""
    models = load_models(current_models)
    model = models[MODEL_NAME]
    calc = model.get_calculator()

    all_results = []
    for pressure_gpa, pressure_label in zip(PRESSURES, PRESSURE_LABELS, strict=False):
        structures = load_structures(
            pressure_label, n_structures=N_TEST_STRUCTURES, random_select=False
        )

        for struct_data in structures:
            atoms = struct_data["atoms"].copy()
            atoms.calc = copy(calc)

            relaxed, converged, enthalpy_per_atom = relax_with_pressure(
                atoms, pressure_gpa
            )

            pred_volume = (
                relaxed.get_volume() / len(relaxed) if relaxed is not None else None
            )

            all_results.append(
                {
                    "pressure_label": pressure_label,
                    "mat_id": struct_data["mat_id"],
                    "volume": pred_volume,
                    "energy_per_atom": enthalpy_per_atom,
                }
            )

    return pd.DataFrame(all_results)


@pytest.mark.parametrize("pressure_label", PRESSURE_LABELS)
def test_high_pressure_relaxation_regression(
    regression_results: pd.DataFrame,
    pressure_label: str,
) -> None:
    """Run calculations and check against stored regression baselines."""
    if not REFERENCE_FILE.exists():
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        regression_results.to_csv(REFERENCE_FILE, index=False)
        pytest.skip(f"Generated baseline {REFERENCE_FILE.name}; re-run to validate")

    expected = pd.read_csv(REFERENCE_FILE)
    expected_p = expected[expected["pressure_label"] == pressure_label]
    results_p = regression_results[
        regression_results["pressure_label"] == pressure_label
    ]

    assert len(expected_p) == N_TEST_STRUCTURES
    assert expected_p["mat_id"].tolist() == results_p["mat_id"].tolist()
    np.testing.assert_allclose(
        results_p["volume"].to_numpy(),
        expected_p["volume"].to_numpy(),
        rtol=0.0,
        atol=VOLUME_ATOL,
    )
    np.testing.assert_allclose(
        results_p["energy_per_atom"].to_numpy(),
        expected_p["energy_per_atom"].to_numpy(),
        rtol=0.0,
        atol=ENERGY_ATOL,
    )
