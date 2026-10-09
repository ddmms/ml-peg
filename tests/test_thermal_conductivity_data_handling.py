"""Check conductivity data handling independently of scientific calculations."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).parents[1]
CALC_PATH = ROOT / "ml_peg/calcs/bulk_crystal/thermal_conductivity"
ANALYSIS_PATH = (
    ROOT
    / "ml_peg/analysis/bulk_crystal/thermal_conductivity"
    / "analyse_thermal_conductivity.py"
)


def _load_functions(path: Path, names: set[str], namespace: dict) -> None:
    """Load actual functions without optional Phono3py or import-time downloads."""
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for function in functions:
        function.decorator_list = []
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture
def conductivity_functions(tmp_path):
    """Load data-handling functions with isolated paths and numeric dependencies."""
    namespace = {
        "Path": Path,
        "h5py": h5py,
        "np": np,
        "pd": pd,
        "warnings": warnings,
        "json": json,
        "go": go,
        "MODELS": ["model"],
        "OUT_PATH": tmp_path,
        "PBE_DATA_PATH": tmp_path,
    }
    _load_functions(
        CALC_PATH / "thermal_conductivity.py",
        {
            "dict_to_hdf5",
            "hdf5_to_dict",
            "load_hdf5_subdir_dicts",
            "calculate_kappa_avg",
        },
        namespace,
    )
    namespace["tc"] = SimpleNamespace(
        TCKeys=SimpleNamespace(
            mat_id="material_id",
            kappa_tot_rta="kappa_tot_rta",
            kappa_tot_avg="kappa_tot_avg",
            has_imag_ph_modes="has_imag_ph_modes",
            final_spg_num="final_spg_num",
            init_spg_num="initial_spg_num",
            spg_num="spg_num",
            mode_weights="mode_weights",
            mode_kappa_tot_avg="mode_kappa_tot_avg",
            srd="srd",
            sre="sre",
            srme="srme",
            true_kappa_tot_avg="true_kappa_tot_avg",
        ),
        **{
            name: namespace[name]
            for name in (
                "dict_to_hdf5",
                "hdf5_to_dict",
                "load_hdf5_subdir_dicts",
                "calculate_kappa_avg",
            )
        },
    )
    _load_functions(
        CALC_PATH / "data/collect_ref_kappas.py",
        {"collect_reference_kappas"},
        namespace,
    )
    _load_functions(
        ANALYSIS_PATH,
        {
            "_add_missing_error_rows",
            "_first_value",
            "status_parity",
            "calc_kappa_metrics_from_dfs",
            "calc_kappa_srme_dataframes",
            "calc_kappa_srme",
        },
        namespace,
    )
    return namespace


@pytest.mark.parametrize("prefix", ["kappa", "fast_kappa"])
@pytest.mark.parametrize("complete", [False, True])
def test_reference_collection_preserves_incomplete_aggregates(
    tmp_path, conductivity_functions, prefix, complete
) -> None:
    """Both reference formats survive missing materials and rebuild when complete."""
    aggregate = f"{prefix}s"
    json_path = tmp_path / f"{aggregate}.json.gz"
    hdf5_path = tmp_path / f"{aggregate}.hdf5"
    json_path.write_bytes(b"previous JSON aggregate")
    hdf5_path.write_bytes(b"previous HDF5 aggregate")

    for index, material_id in enumerate(["mp-1", "mp-2"]):
        directory = tmp_path / material_id
        directory.mkdir()
        if index == 0 or complete:
            with h5py.File(directory / f"{prefix}.hdf5", "w") as handle:
                conductivity_functions["dict_to_hdf5"](
                    {"kappa_tot_avg": [index + 1.0]}, handle
                )

    conductivity_functions["collect_reference_kappas"](prefix, aggregate)

    if complete:
        with h5py.File(hdf5_path) as handle:
            assert set(handle) == {"mp-1", "mp-2"}
        assert set(pd.read_json(json_path)["material_id"]) == {"mp-1", "mp-2"}
    else:
        assert json_path.read_bytes() == b"previous JSON aggregate"
        assert hdf5_path.read_bytes() == b"previous HDF5 aggregate"


@pytest.mark.parametrize(
    "prediction_ids", [["mp-2", "mp-1", "extra"], ["mp-1", "extra"]]
)
def test_prediction_alignment_discards_unknown_ids(
    conductivity_functions, prediction_ids
) -> None:
    """Extra predictions cannot reach reference lookup, even with no missing IDs."""
    predictions = pd.DataFrame(
        {"kappa_tot_rta": np.arange(len(prediction_ids))}, index=prediction_ids
    )
    reference = pd.DataFrame(index=["mp-1", "mp-2"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        aligned = conductivity_functions["_add_missing_error_rows"](
            predictions, reference, "model"
        )

    assert aligned.index.tolist() == reference.index.tolist()
    assert (
        aligned.loc["mp-1", "kappa_tot_rta"] == predictions.loc["mp-1", "kappa_tot_rta"]
    )
    if "mp-2" not in prediction_ids:
        assert pd.isna(aligned.loc["mp-2", "kappa_tot_rta"])


@pytest.mark.parametrize("valid_point", [False, True])
def test_status_parity_excludes_nonfinite_points_and_limits(
    tmp_path, conductivity_functions, valid_point
) -> None:
    """Status figures contain valid JSON, finite coordinates, and fallback limits."""
    predictions = [np.inf, np.nan, 2.0, -1.0]
    reference = [1.0, 2.0, np.inf, 3.0]
    expected_ids = []
    if valid_point:
        predictions.append(4.0)
        reference.append(5.0)
        expected_ids = [["mp-4"]]
    index = [f"mp-{i}" for i in range(len(predictions))]
    stats = {
        "ref": pd.DataFrame({"kappa_tot_avg": reference}, index=index),
        "model": pd.DataFrame(
            {"kappa_tot_avg": predictions, "has_imag_ph_modes": False}, index=index
        ),
    }
    if not valid_point:
        stats["ref"]["kappa_tot_avg"] = np.nan
        stats["model"].loc["mp-2", "kappa_tot_avg"] = -2.0

    conductivity_functions["status_parity"](stats)

    def reject_nonfinite(value):
        raise AssertionError(f"Non-finite value in JSON: {value}")

    figure = json.loads(
        (tmp_path / "figure_status_parity.json").read_text(),
        parse_constant=reject_nonfinite,
    )["model"]
    assert figure["data"][0]["customdata"] == expected_ids
    expected_limits = [1.0, 5.0] if valid_point else [1e-3, 1.0]
    assert figure["data"][-1]["x"] == expected_limits
    assert figure["data"][-1]["y"] == expected_limits


def test_nan_mode_scores_receive_existing_failure_penalty(
    conductivity_functions,
) -> None:
    """Failed mode scores count in the aggregate while valid scores stay unchanged."""
    index = ["good", "bad-mode"]
    reference = pd.DataFrame(
        {
            "kappa_tot_avg": [np.array([1.0]), np.array([1.0])],
            "mode_kappa_tot_avg": [np.ones((1, 2)), np.ones((1, 2))],
        },
        index=index,
    )
    predictions = pd.DataFrame(
        {
            "kappa_tot_rta": [np.ones((1, 6)), np.ones((1, 6))],
            "mode_weights": [np.ones(2), np.ones(2)],
            "mode_kappa_tot_avg": [np.ones((1, 2)), np.array([[np.nan, 1.0]])],
        },
        index=index,
    )

    scored = conductivity_functions["calc_kappa_metrics_from_dfs"](
        predictions, reference
    )

    assert scored["srme"].tolist() == [0.0, 2.0]
    assert scored["srme"].mean() == 1.0
    assert scored["sre"].tolist() == [0.0, 0.0]
