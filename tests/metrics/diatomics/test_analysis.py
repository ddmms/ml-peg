"""Integration tests for ml-peg diatomic analysis entry points."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.physicality.diatomics.analyse_diatomics import (
    DEFAULT_THRESHOLDS,
    aggregate_model_metrics,
    collect_metrics,
    evaluate_mbd_diatomic_metrics,
    write_mbd_diatomic_metrics,
)
from ml_peg.analysis.physicality.diatomics.metrics import DIATOMIC_METRIC_KEYS


def _integration_dataframe() -> pd.DataFrame:
    """Build smooth homo- and heteronuclear current-format curves."""
    distances = np.linspace(0.3, 3.0, 20)
    energies = 100 * (distances - 1.0) ** 2 - 2
    force_parallel = -200 * (distances - 1.0)
    dataframe = pd.DataFrame(
        {
            "element_1": "H",
            "distance": distances,
            "energy": energies,
            "force_parallel": force_parallel,
        }
    )
    return pd.concat(
        [
            dataframe.assign(pair=pair, element_2=element_2)
            for pair, element_2 in (("H-H", "H"), ("H-He", "He"))
        ],
        ignore_index=True,
    )


def _write_reference(path: Path) -> None:
    """Write a matching gzipped PBE curve for integration tests."""
    dataframe = _integration_dataframe().query("pair == 'H-H'")
    forces = np.zeros((len(dataframe), 2, 3))
    forces[:, 0, 0] = -dataframe["force_parallel"]
    forces[:, 1, 0] = dataframe["force_parallel"]
    curve = {
        "distances": dataframe["distance"].tolist(),
        "energies": dataframe["energy"].tolist(),
        "forces": forces.tolist(),
    }
    with gzip.open(path, mode="wt", encoding="utf-8") as file:
        json.dump({"PBE": {"H-H": curve}}, file)


def test_separate_mbd_evaluation_is_json_safe_and_homonuclear(
    tmp_path: Path,
) -> None:
    """Separate result reports 12 homonuclear metrics without legacy weighting."""
    reference_path = tmp_path / "reference.json.gz"
    _write_reference(reference_path)
    pair_data = {"test-model": _integration_dataframe()}
    result = evaluate_mbd_diatomic_metrics(
        pair_data,
        reference_path=reference_path,
        interpolate=200,
    )

    assert result["weighted_in_legacy_score"] is False
    assert set(result["metric_names"]) == DIATOMIC_METRIC_KEYS
    model_results = result["models"]
    assert isinstance(model_results, dict)
    assert set(model_results["test-model"]["elements"]) == {"H"}
    assert set(model_results["test-model"]["means"]) == DIATOMIC_METRIC_KEYS
    json.dumps(result, allow_nan=False)

    output_path = tmp_path / "mbd-metrics.json"
    written = write_mbd_diatomic_metrics(
        output_path,
        pair_data,
        reference_path=reference_path,
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == written


def test_mbd_evaluation_uses_pair_labels_for_homonuclear_filtering(
    tmp_path: Path,
) -> None:
    """Ignore inconsistent element columns and classify pairs from their labels."""
    reference_path = tmp_path / "reference.json.gz"
    _write_reference(reference_path)
    dataframe = _integration_dataframe()
    dataframe.loc[dataframe["pair"] == "H-H", "element_2"] = "He"

    result = evaluate_mbd_diatomic_metrics(
        {"test-model": dataframe}, reference_path=reference_path
    )

    assert set(result["models"]["test-model"]["elements"]) == {"H"}


def test_legacy_metric_regression_remains_unchanged() -> None:
    """Legacy five-metric homo-plus-hetero aggregation retains pinned values."""
    pair_dataframe = pd.DataFrame(
        {
            "pair": ["H-H"] * 5 + ["H-He"] * 5,
            "distance": [1, 2, 3, 4, 5] * 2,
            "energy": [4, 1, 0, 1, 4] * 2,
            "force_parallel": [-2, -1, 0, 1, 2] * 2,
        }
    )
    expected = {
        "Force flips": 1.0,
        "Energy minima": 1.0,
        "Energy inflections": 0.0,
        "ρ(E, repulsion)": -1.0,
        "ρ(E, attraction)": 1.0,
    }

    assert list(DEFAULT_THRESHOLDS) == list(expected)
    assert aggregate_model_metrics(pair_dataframe) == pytest.approx(expected)
    collected = collect_metrics({"test-model": pair_dataframe})
    collected_record = collected.to_dict(orient="records")[0]
    assert collected_record.pop("Model") == "test-model"
    assert collected_record == pytest.approx(expected)
