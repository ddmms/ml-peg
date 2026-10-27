"""Check conductivity data handling independently of scientific calculations."""

from __future__ import annotations

import ast
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

from ase import Atoms
from ase.io import read, write
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ml_peg import analysis, models
from ml_peg.analysis.utils.decorators import merge_saved_models
from ml_peg.app.utils.plot_helpers import build_violin_distribution

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
        "merge_saved_models": merge_saved_models,
        "build_violin_distribution": build_violin_distribution,
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
            "failure",
            "instability",
            "_first_value",
            "status_parity",
            "_write_srme_violin",
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


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("plot", ["srme", "fast_srme", "status"])
def test_conductivity_plot_bundles_respect_update_mode(
    tmp_path, monkeypatch, fake_models, conductivity_functions, update, plot
) -> None:
    """Updates preserve other models' figures; ordinary runs replace the bundle."""
    funcs = conductivity_functions
    funcs["MODELS"] = ["model_1", "model_2"]
    index = ["mp-1", "mp-2"]
    stats = {"ref": pd.DataFrame({"kappa_tot_avg": [1.0, 2.0]}, index=index)}
    for model, values in (("model_1", [0.2, 0.4]), ("model_2", [0.6, 0.8])):
        stats[model] = pd.DataFrame(
            {
                "kappa_tot_avg": [3.0, 4.0],
                "has_imag_ph_modes": False,
                "srme": values,
                "fast_srme": values,
            },
            index=index,
        )

    filename = "figure_status_parity.json" if plot == "status" else f"{plot}.json"

    def write_plots():
        if plot == "status":
            funcs["status_parity"](stats)
        else:
            funcs["_write_srme_violin"](stats, plot, filename, plot)

    write_plots()
    original = json.loads((tmp_path / filename).read_text())
    funcs["MODELS"] = ["model_2"]
    monkeypatch.setattr(models, "current_models", "model_2")
    monkeypatch.setattr(analysis, "update_results", update)
    stats["model_2"]["kappa_tot_avg"] = [10.0, 20.0]
    stats["model_2"]["srme"] = [0.1, 0.2]
    stats["model_2"]["fast_srme"] = [0.1, 0.2]
    write_plots()
    saved = json.loads((tmp_path / filename).read_text())

    assert list(saved) == (["model_1", "model_2"] if update else ["model_2"])
    assert saved["model_2"] != original["model_2"]
    if update:
        assert saved["model_1"] == original["model_1"]


@pytest.mark.parametrize("separate_fc2", [False, True])
def test_reference_supercells_survive_extxyz_round_trip(separate_fc2) -> None:
    """Both shared and separate fc2 supercells retain valid matrices on disk."""
    # Evaluate the production metadata expression without running downloads or
    # reference calculations from this standalone generation script.
    path = CALC_PATH / "data/reference_generation.py"
    assignment = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "info_dict"
            for target in node.targets
        )
    )
    fc3 = np.diag([2, 2, 2])
    fc2 = np.diag([4, 4, 4]) if separate_fc2 else None
    namespace = {
        "ph3": SimpleNamespace(
            phonon_supercell_matrix=fc2,
            supercell_matrix=fc3,
            primitive_matrix=np.eye(3),
            symmetry=SimpleNamespace(dataset=SimpleNamespace(number=1)),
        ),
        "tc": SimpleNamespace(TCKeys=SimpleNamespace(mat_id="material_id")),
        "name": "Si",
        "mat_id": "mp-1",
        "original_qmesh": [9, 9, 9],
        "fast_qmesh": [3, 3, 3],
    }
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    atoms = Atoms("Si", cell=np.eye(3), pbc=True)
    atoms.info.update(namespace["info_dict"])
    stream = StringIO()
    write(stream, atoms, format="extxyz")
    stream.seek(0)
    restored = read(stream, format="extxyz")
    np.testing.assert_array_equal(
        restored.info["fc2_supercell"], fc2 if separate_fc2 else fc3
    )
    np.testing.assert_array_equal(restored.info["fc3_supercell"], fc3)


def test_all_failed_materials_receive_failure_penalty(conductivity_functions) -> None:
    """Missing phonon status counts as failure without asserting imaginary modes."""
    reference = pd.DataFrame(index=["mp-1", "mp-2"])
    predictions = pd.DataFrame({"error": ["failed", "failed"]}, index=reference.index)
    aligned = conductivity_functions["_add_missing_error_rows"](
        predictions, reference, "model"
    )
    stats = {"ref": reference, "model": aligned}
    assert conductivity_functions["failure"](stats) == {"model": 1.0}
    assert conductivity_functions["instability"](stats) == {"model": 0.0}
    assert "has_imag_ph_modes" not in predictions


def test_existing_phonon_status_is_preserved(conductivity_functions) -> None:
    """Known stable and unstable results remain distinct from missing results."""
    reference = pd.DataFrame(index=["mp-1", "mp-2", "mp-3"])
    predictions = pd.DataFrame(
        {"has_imag_ph_modes": [False, True, np.nan]}, index=reference.index
    )
    aligned = conductivity_functions["_add_missing_error_rows"](
        predictions, reference, "model"
    )
    stats = {"ref": reference, "model": aligned}
    assert conductivity_functions["failure"](stats)["model"] == pytest.approx(1 / 3)
    assert conductivity_functions["instability"](stats)["model"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("existing_alias", [False, True])
@pytest.mark.parametrize("missing_plugin", [False, True])
def test_wte_registration_cleans_up_grid_alias(existing_alias, missing_plugin):
    """Register explicitly without replacing or leaking the legacy grid module."""
    grid = object()
    previous = object()
    alias = "phono3py.phonon.grid"
    modules = {alias: previous} if existing_alias else {}
    calls = []

    def import_plugin(name):
        if name == "phonopy.phonon.grid":
            return grid
        assert name == "wte"
        assert modules[alias] is (previous if existing_alias else grid)
        if missing_plugin:
            raise ModuleNotFoundError(name="wte")
        return SimpleNamespace(register=lambda: calls.append("registered"))

    namespace = {
        "sys": SimpleNamespace(modules=modules),
        "import_module": import_plugin,
    }
    _load_functions(CALC_PATH / "thermal_conductivity.py", {"register_wte"}, namespace)
    if missing_plugin:
        with pytest.raises(ModuleNotFoundError, match="uv sync"):
            namespace["register_wte"]()
    else:
        namespace["register_wte"]()
        assert calls == ["registered"]
    assert modules == ({alias: previous} if existing_alias else {})


def test_wte_mode_outputs_preserve_total_conductivity():
    """Use the plugin API while retaining mode contributions and output keys."""
    from copy import deepcopy

    keys = SimpleNamespace(
        **{
            name: name
            for name in (
                "kappa_tot_rta",
                "kappa_p_rta",
                "kappa_c",
                "mode_weights",
                "q_points",
                "ph_freqs",
                "mode_kappa_tot_rta",
            )
        }
    )
    population = np.ones((1, 1, 1, 2, 6))
    coherence = np.zeros((1, 1, 1, 2, 2, 6))
    coherence[:, :, :, 0, 1] = 0.25
    coherence[:, :, :, 1, 0] = 0.25
    kappa = SimpleNamespace(
        kappa_TOT_RTA=np.full((1, 1, 6), 1.25),
        kappa_P_RTA=np.ones((1, 1, 6)),
        kappa_C=np.full((1, 1, 6), 0.25),
        mode_kappa=population,
        mode_kappa_C=coherence,
        mode_heat_capacities=np.ones((1, 1, 2)),
        grid_weights=np.array([2]),
        qpoints=np.zeros((1, 3)),
        frequencies=np.ones((1, 2)),
    )
    calls = []
    ph3 = SimpleNamespace(
        thermal_conductivity=kappa,
        init_phph_interaction=lambda **kw: calls.append(("init", kw)),
        run_thermal_conductivity=lambda **kw: calls.append(("run", kw)),
    )
    namespace = {
        "np": np,
        "deepcopy": deepcopy,
        "warnings": warnings,
        "TCKeys": keys,
        "register_wte": lambda: calls.append(("register", {})),
    }
    _load_functions(
        CALC_PATH / "thermal_conductivity.py",
        {"calculate_conductivity", "calc_mode_kappa_tot"},
        namespace,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result_ph3, result, result_kappa = namespace["calculate_conductivity"](
            ph3,
            [300],
            boundary_mfp=123,
            log_level=0,
        )
    assert result_ph3 is ph3
    assert result_kappa is kappa
    assert [name for name, _ in calls] == ["register", "init", "run"]
    assert calls[-1][1] == {
        "transport_type": "WTE",
        "temperatures": [300],
        "is_isotope": True,
        "boundary_mfp": 123,
        "log_level": 0,
    }
    np.testing.assert_allclose(result["mode_kappa_tot_rta"], 1.25)
    np.testing.assert_allclose(
        result["mode_kappa_tot_rta"].sum(axis=(1, 2)) / 2,
        result["kappa_tot_rta"],
    )
