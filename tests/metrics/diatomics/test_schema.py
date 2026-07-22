"""Tests for diatomic schemas and local data adapters."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.physicality.diatomics.metrics import (
    DiatomicCurve,
    DiatomicCurves,
    load_dft_reference_curves,
    load_mbd_json,
    load_ml_peg_curves,
)


def _curve_payload() -> dict[str, object]:
    """Return a minimal two-point MBD curve payload."""
    return {
        "energies": [0.2, 0.0],
        "forces": [
            [[0.1, 0, 0], [-0.1, 0, 0]],
            [[0.0, 0, 0], [0.0, 0, 0]],
        ],
    }


def _ml_peg_dataframe(
    *rows: tuple[object, float, float, float],
) -> pd.DataFrame:
    """Build an ml-peg dataframe from pair, distance, energy, and force rows."""
    return pd.DataFrame(rows, columns=("pair", "distance", "energy", "force_parallel"))


def test_diatomic_classes_parse_and_reshape() -> None:
    """Typed classes convert arrays and reshape legacy flattened forces."""
    distances = [1.0, 2.0]
    energies = [0.1, 0.2]
    flattened_forces = np.arange(12).reshape(1, 4, 3)

    curve = DiatomicCurve(distances, energies, flattened_forces)
    assert curve.forces.shape == (2, 2, 3)
    assert {
        type(curve.distances),
        type(curve.energies),
        type(curve.forces),
    } == {np.ndarray}

    payload = {
        "distances": distances,
        "homo-nuclear": {"H-H": _curve_payload()},
        "hetero_nuclear": {"H-He": _curve_payload()},
    }
    curves = DiatomicCurves.from_dict(payload)
    assert list(curves.homo_nuclear) == ["H"]
    assert list(curves.hetero_nuclear) == ["H-He"]
    np.testing.assert_array_equal(curves.homo_nuclear["H"].distances, distances)


@pytest.mark.parametrize(
    ("override", "error_match"),
    [
        ({"energies": [0.0]}, "distance and energy counts differ"),
        ({"forces": np.zeros((1, 2, 3))}, "distance and force counts differ"),
        ({"forces": np.zeros((2, 1, 3))}, "forces must have shape"),
        ({"distances": [[1.0], [2.0]]}, "distances must have shape"),
        ({"energies": [[0.0], [1.0]]}, "energies must have shape"),
    ],
)
def test_diatomic_curve_rejects_invalid_shapes_and_counts(
    override: dict[str, object],
    error_match: str,
) -> None:
    """DiatomicCurve rejects malformed arrays and sample-count mismatches."""
    arguments: dict[str, object] = {
        "distances": [1.0, 2.0],
        "energies": [0.0, 1.0],
        "forces": np.zeros((2, 2, 3)),
    }
    with pytest.raises(ValueError, match=error_match):
        DiatomicCurve(**(arguments | override))


@pytest.mark.parametrize(
    "bad_distances",
    [[0.5, 1.5], [1.0, 1.0], [2.0, 1.0]],
    ids=["off-grid", "duplicate", "reordered"],
)
def test_mbd_schema_rejects_invalid_curve_grids(
    bad_distances: list[float],
) -> None:
    """MBD curves must use ordered subsets of the top-level grid."""
    curve_payload = _curve_payload() | {"distances": bad_distances}
    payload = {
        "distances": [1.0, 2.0],
        "homo-nuclear": {"H-H": curve_payload},
    }
    with pytest.raises(ValueError, match="must be an ordered subset"):
        DiatomicCurves.from_dict(payload)


def test_json_and_gzip_loaders(tmp_path: Path) -> None:
    """MBD JSON and gzipped DFT references load into the typed schema."""
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps(
            {
                "distances": [0.7, 1.0],
                "homo-nuclear": {"H-H": _curve_payload()},
            }
        ),
        encoding="utf-8",
    )
    prediction_curves = load_mbd_json(prediction_path)
    assert list(prediction_curves.homo_nuclear) == ["H"]

    reference_path = tmp_path / "reference.json.gz"
    with gzip.open(reference_path, mode="wt", encoding="utf-8") as file:
        json.dump({"PBE": {"H-H": _curve_payload() | {"distances": [0.7, 1.0]}}}, file)
    reference_curves = load_dft_reference_curves(ref_path=reference_path)
    np.testing.assert_array_equal(
        reference_curves.homo_nuclear["H"].energies,
        [0.2, 0.0],
    )


def test_ml_peg_dataframe_and_csv_force_adapter(tmp_path: Path) -> None:
    """CSV adapter reconstructs x-axis forces with the expected atom signs."""
    dataframe = _ml_peg_dataframe(
        ("H-H", 2, 0, 2.5),
        ("H-H", 1, 1, -1.5),
        ("H-He", 1, 2, 4),
        ("H-He", 2, 1, -3),
    )
    csv_path = tmp_path / "diatomics.csv"
    dataframe.to_csv(csv_path, index=False)

    for source in (dataframe, csv_path):
        curves = load_ml_peg_curves(source)
        assert list(curves.homo_nuclear) == ["H"]
        assert list(curves.hetero_nuclear) == ["H-He"]
        h_curve = curves.homo_nuclear["H"]
        np.testing.assert_array_equal(h_curve.distances, [1.0, 2.0])
        np.testing.assert_array_equal(h_curve.forces[:, 0, 0], [1.5, -2.5])
        np.testing.assert_array_equal(h_curve.forces[:, 1, 0], [-1.5, 2.5])
        np.testing.assert_array_equal(h_curve.forces[:, :, 1:], 0)

    homonuclear_only = load_ml_peg_curves(dataframe, include_heteronuclear=False)
    assert list(homonuclear_only.homo_nuclear) == ["H"]
    assert homonuclear_only.hetero_nuclear == {}


@pytest.mark.parametrize(
    ("dataframe", "error_match"),
    [
        (pd.DataFrame({"pair": ["H-H"]}), "Missing ml-peg diatomics columns"),
        (_ml_peg_dataframe(("H2", 1, 0, 0)), "pair labels must have form"),
    ],
)
def test_ml_peg_adapter_rejects_schema_errors(
    dataframe: pd.DataFrame,
    error_match: str,
) -> None:
    """ml-peg adapter reports missing columns and malformed pair labels."""
    with pytest.raises(ValueError, match=error_match):
        load_ml_peg_curves(dataframe)


@pytest.mark.parametrize(
    "pair_values",
    [[None, "H-H"], [None, None]],
    ids=["partly-null", "entirely-null"],
)
def test_ml_peg_adapter_rejects_null_pair_labels(
    pair_values: list[str | None],
) -> None:
    """Pass null pair labels through grouping so schema validation rejects them."""
    dataframe = pd.DataFrame(
        {
            "pair": pair_values,
            "distance": range(1, len(pair_values) + 1),
            "energy": 0.0,
            "force_parallel": 0.0,
        }
    )

    with pytest.raises(ValueError, match="pair labels must have form"):
        load_ml_peg_curves(dataframe)


@pytest.mark.parametrize(
    ("energies", "error_match"),
    [
        ([0.0, 0.0], "duplicate distance values"),
        ([0.0, 1.0], "conflicting samples"),
    ],
    ids=["identical", "conflicting"],
)
def test_ml_peg_adapter_rejects_duplicate_distances(
    energies: list[float], error_match: str
) -> None:
    """Reject duplicate distances instead of retaining an arbitrary sample."""
    dataframe = _ml_peg_dataframe(
        ("H-H", 1, energies[0], 0),
        ("H-H", 1, energies[1], 0),
    )
    with pytest.raises(ValueError, match=error_match):
        load_ml_peg_curves(dataframe)
