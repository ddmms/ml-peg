"""Focused AM26 component and static-figure tests without browser or model runs."""

from __future__ import annotations

import ast
from copy import deepcopy
import io
import json
from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from dash.dcc import Graph
from dash.development.base_component import Component
import numpy as np
import pytest
from test_am26_analysis import (
    memory_files as memory_files,
)
from test_am26_analysis import (
    output_frames as output_frames,
)
from test_am26_analysis import (
    pipeline as pipeline,
)
from test_am26_analysis import (
    reference_frames as reference_frames,
)
from test_am26_analysis import (
    xyz_bytes,
)

from ml_peg import analysis, models
from ml_peg.analysis.amorphous_materials.AM26 import analyse_AM26 as am26
from ml_peg.analysis.amorphous_materials.AM26.heatmaps import (
    FIGURE_FILENAMES,
    build_heatmap,
    write_heatmaps,
)
from ml_peg.app.amorphous_materials.AM26 import app_AM26 as app_module
from ml_peg.app.base_app import BaseApp
from ml_peg.app.build_app import get_all_tests


def walk_components(component):
    """Traverse the actual BaseApp layout without starting a Dash server."""
    if not isinstance(component, Component):
        return
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, (tuple, list)):
        for child in children:
            yield from walk_components(child)
    elif isinstance(children, Component):
        yield from walk_components(children)


def graphs(app):
    """Find both saved figures placed beneath the standard table."""
    return {c.id: c for c in walk_components(app.layout) if isinstance(c, Graph)}


@pytest.fixture
def app_data(pipeline, reference_frames, monkeypatch):
    """Generate a Phase 3 table/bundle with zero, >100%, and unavailable cells."""
    summary = am26.reference_summary(reference_frames)
    results = {
        model: am26.unavailable_result(summary, "missing_output")
        for model in ("model_1", "model_2")
    }
    record = results["model_1"]
    record["status"] = "aligned"
    energy = [0.0, 25.0, 50.0, 100.0, 163.08216905423208]
    force = [0.0, 20.0, 40.0, 60.0, 80.0]
    for index, system in enumerate(am26.SYSTEM_ORDER):
        values = record["systems"][system]
        values.update(
            energy_rmse=energy[index] / 100 * values["energy_std"],
            force_rmse=force[index] / 100 * values["force_std"],
            valid_energy_count=2,
            failed_energy_count=0,
            valid_force_count=2,
            failed_force_count=0,
        )
        values["Energy RelRMSE"] = energy[index]
        values["Force RelRMSE"] = force[index]
    record["overall"] = {
        "Mean Energy RelRMSE": float(np.mean(energy)),
        "Mean Force RelRMSE": float(np.mean(force)),
    }
    am26.write_metrics_table(results)
    am26.write_reference_info()
    bundle = am26.write_diagnostics(results, reference_frames)
    write_heatmaps(bundle, am26.OUT_PATH)
    monkeypatch.setattr(app_module, "DATA_PATH", am26.OUT_PATH)
    return pipeline, bundle


def test_normal_app_discovery_table_and_info(app_data):
    """AM26 is discovered in its existing category without a registry edit."""
    apps, layouts, tables, _ = get_all_tests(
        category="amorphous_materials", test="AM26"
    )
    app = apps["AM26"]
    assert isinstance(app, app_module.AM26App)
    assert isinstance(app, BaseApp)
    assert layouts["amorphous_materials"]["AM26"] is app.layout
    assert tables["amorphous_materials"]["AM26"] is app.table
    assert app.metrics == ["Mean Energy RelRMSE", "Mean Force RelRMSE"]
    assert [c["name"] for c in app.table.columns] == [
        "MLIP",
        "Score",
        "Mean Energy RelRMSE [%]",
        "Mean Force RelRMSE [%]",
    ]
    assert app.elements == {"C", "Si", "O", "Ge", "Sb", "Te", "Li", "P", "S"}
    assert len(app.info["elements"]) == 10
    for metric in app.metrics:
        assert app.table.thresholds[metric]["good"] == 0.0
        assert app.table.thresholds[metric]["bad"] == 100.0
        assert app.table.thresholds[metric]["level_of_theory"] == "PBE"
    assert app.register_callbacks() is None


@pytest.mark.parametrize("metric", ["Energy RelRMSE", "Force RelRMSE"])
def test_saved_heatmap_values_and_column_order(app_data, metric):
    """Saved figures match Phase 3 values and appear through extra_components."""
    files, _ = app_data
    bundle = json.loads(files[am26.OUT_PATH / "system_metrics.json"])
    app = app_module.get_app()
    graph_id = (
        "AM26-energy-relrmse-heatmap"
        if metric.startswith("Energy")
        else "AM26-force-relrmse-heatmap"
    )
    graph = graphs(app)[graph_id]
    heatmap = next(t for t in graph.figure["data"] if t["type"] == "heatmap")
    assert heatmap["x"] == ["a-C", "a-Si", "a-SiO2", "a-GST", "a-LiPS"]
    assert heatmap["y"] == list(bundle["models"])
    for row, model in zip(heatmap["z"], heatmap["y"], strict=True):
        expected = [
            bundle["models"][model]["systems"][s][metric] for s in am26.SYSTEM_ORDER
        ]
        assert row == expected
    assert "Model: %{y}" in heatmap["hovertemplate"]
    assert "Material system: %{x}" in heatmap["hovertemplate"]
    assert "%{z:.15g}" in heatmap["hovertemplate"]
    assert len(graphs(app)) == 2


def test_zero_missing_and_above_100_are_distinct(app_data):
    """Saturate colours at the bad threshold without clipping or zero-filling z."""
    app = app_module.get_app()
    figure = graphs(app)["AM26-energy-relrmse-heatmap"].figure
    heatmap = next(t for t in figure["data"] if t["type"] == "heatmap")
    assert heatmap["zmin"] == 0.0
    assert heatmap["zmax"] == 100.0
    assert heatmap["z"][0][0] == 0.0
    assert heatmap["z"][0][4] == 163.08216905423208
    assert heatmap["z"][0][4] > heatmap["zmax"]
    assert "%{z" in heatmap["hovertemplate"]
    assert heatmap["colorbar"]["ticktext"][-1] == "≥100"
    assert heatmap["z"][1] == [None] * 5
    assert heatmap["hoverongaps"] is False
    assert heatmap["connectgaps"] is False
    missing = next(t for t in figure["data"] if t["type"] == "scatter")
    assert missing["text"] == ["N/A"] * 5
    assert missing["y"] == ["model_2"] * 5
    assert "unavailable or failed" in missing["hovertemplate"]
    assert figure["layout"]["plot_bgcolor"] == "#eeeeee"
    assert any("not truncated" in a["text"] for a in figure["layout"]["annotations"])


@pytest.mark.parametrize("failure", ["Energy", "Force", "both"])
def test_independent_missing_properties_do_not_become_zero(
    app_data, reference_frames, failure
):
    """Failed properties remain gaps while the other property stays available."""
    _, bundle = app_data
    record = bundle["models"]["model_1"]
    # Keep another model available: BaseApp deliberately hides the entire results
    # area when every model lacks both overall metrics.
    bundle["models"]["model_2"] = deepcopy(record)
    affected = ("Energy", "Force") if failure == "both" else (failure,)
    for prop in affected:
        record["systems"]["silica"][f"{prop} RelRMSE"] = np.nan
        record["overall"][f"Mean {prop} RelRMSE"] = np.nan
        key = "energy" if prop == "Energy" else "force"
        record["systems"]["silica"][f"failed_{key}_count"] = 1
    am26.write_metrics_table(bundle["models"])
    merged = am26.write_diagnostics(bundle["models"], reference_frames)
    write_heatmaps(merged, am26.OUT_PATH)
    app = app_module.get_app()
    for prop in ("Energy", "Force"):
        figure = graphs(app)[f"AM26-{prop.lower()}-relrmse-heatmap"].figure
        heatmap = next(t for t in figure["data"] if t["type"] == "heatmap")
        value = heatmap["z"][0][2]
        if prop in affected:
            assert value is None
        else:
            assert value == record["systems"]["silica"][f"{prop} RelRMSE"]
    rows = [r for r in app.table.data if r["id"] == "model_1"]
    if failure == "both":
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]["Score"] == "NaN"
        assert rows[0][f"Mean {failure} RelRMSE"] == "NaN"
        surviving = "Force" if failure == "Energy" else "Energy"
        metric = f"Mean {surviving} RelRMSE"
        assert rows[0][metric] == pytest.approx(record["overall"][metric])


def test_all_unavailable_results_use_standard_no_results_layout(
    app_data, reference_frames
):
    """Keep BaseApp's no-results notice when no model has a scored result."""
    files, _ = app_data
    summary = am26.reference_summary(reference_frames)
    results = {
        model: am26.unavailable_result(summary, "missing_output")
        for model in ("model_1", "model_2")
    }
    am26.write_metrics_table(results)
    bundle = am26.write_diagnostics(results, reference_frames)
    write_heatmaps(bundle, am26.OUT_PATH)
    app = app_module.get_app()
    assert app.table.data == []
    assert graphs(app) == {}
    assert any(
        getattr(c, "children", None) == "No results found"
        for c in walk_components(app.layout)
    )
    for filename in FIGURE_FILENAMES.values():
        figure = json.loads(files[am26.OUT_PATH / filename])
        heatmap = next(t for t in figure["data"] if t["type"] == "heatmap")
        assert heatmap["z"] == [[None] * 5, [None] * 5]


def test_absent_system_entry_is_a_gap(app_data):
    """An omitted system record is unavailable rather than an invented zero."""
    _, bundle = app_data
    bundle["models"]["model_1"]["systems"] = {"carbon": {"Energy RelRMSE": 0.0}}
    figure = build_heatmap(bundle, "Energy RelRMSE")
    assert list(figure.data[0].z[0]) == [0.0, None, None, None, None]


def test_empty_model_bundle_has_no_fabricated_values():
    """No recorded models produces an explicit empty figure, without zero rows."""
    figure = build_heatmap({"models": {}}, "Energy RelRMSE")
    assert len(figure.data) == 0
    assert any("No AM26 system results" in a.text for a in figure.layout.annotations)


@pytest.mark.parametrize("missing", ["energy", "force", "both", "info"])
def test_missing_plot_or_info_uses_standard_behaviour(app_data, monkeypatch, missing):
    """Keep the table available when optional figure/info files are absent."""
    paths = set()
    if missing in {"energy", "both"}:
        paths.add(am26.OUT_PATH / FIGURE_FILENAMES["Energy RelRMSE"])
    if missing in {"force", "both"}:
        paths.add(am26.OUT_PATH / FIGURE_FILENAMES["Force RelRMSE"])
    if missing == "info":
        paths.add(am26.OUT_PATH / "info.json")
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if p in paths else exists(p))
    if missing == "info":
        with pytest.warns(UserWarning, match="info.json does not exist"):
            app = app_module.get_app()
        assert app.info is None
    else:
        app = app_module.get_app()
        for prop in ("energy", "force"):
            graph = graphs(app)[f"AM26-{prop}-relrmse-heatmap"]
            assert (graph.figure is None) == (missing in {prop, "both"})
    assert app.metrics == ["Mean Energy RelRMSE", "Mean Force RelRMSE"]


def test_missing_table_is_skipped_by_normal_registration(app_data, monkeypatch):
    """Follow get_all_tests' existing warning/skip path for absent table data."""
    monkeypatch.setattr(app_module, "DATA_PATH", am26.OUT_PATH / "not-generated")
    with pytest.warns(UserWarning, match="Unable to load layout for AM26"):
        apps, _, _, _ = get_all_tests(category="amorphous_materials", test="AM26")
    assert "AM26" not in apps


def test_heatmap_generation_does_not_change_scored_table(app_data):
    """Diagnostic rendering never changes stored metric values or scores."""
    files, bundle = app_data
    path = am26.OUT_PATH / "am26_metrics_table.json"
    before = files[path]
    write_heatmaps(bundle, am26.OUT_PATH)
    app_module.get_app()
    assert files[path] == before


def test_app_has_no_scientific_or_custom_filtering_code():
    """App code loads saved artifacts and introduces no model/filter callbacks."""
    source = ast.parse(Path(app_module.__file__).read_text())
    imports = [
        node.module for node in ast.walk(source) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(
        module and ("analysis" in module or "calcs" in module or "models" in module)
        for module in imports
    )
    functions = {
        node.name for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
    }
    assert functions == {"get_app", "register_callbacks"}


def test_incremental_analysis_rebuilds_heatmaps_from_merged_bundle(
    pipeline, reference_frames, output_frames, monkeypatch
):
    """The rendering hook preserves unselected model rows during --update."""
    am26.run_analysis()
    paths = [am26.OUT_PATH / filename for filename in FIGURE_FILENAMES.values()]
    before = [json.loads(pipeline[path]) for path in paths]
    changed = read(
        io.StringIO(xyz_bytes(output_frames).decode()), index=":", format="extxyz"
    )
    for atoms in changed:
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=atoms.info["ref_energy"] + len(atoms) * 0.7,
            forces=atoms.arrays["ref_forces"] + 0.7,
        )
    pipeline[am26.CALC_PATH / "model_2" / "am26_results.extxyz"] = xyz_bytes(changed)
    monkeypatch.setattr(models, "current_models", "model_2")
    monkeypatch.setattr(analysis, "update_results", True)
    monkeypatch.setattr(am26, "MODELS", ["model_2"])
    am26.run_analysis()
    for old, path in zip(before, paths, strict=True):
        new = json.loads(pipeline[path])
        old_heatmap = next(t for t in old["data"] if t["type"] == "heatmap")
        new_heatmap = next(t for t in new["data"] if t["type"] == "heatmap")
        assert new_heatmap["y"] == ["model_1", "model_2"]
        assert new_heatmap["z"][0] == old_heatmap["z"][0]
        assert new_heatmap["z"][1] != old_heatmap["z"][1]
