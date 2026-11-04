"""Synthetic AM26 analysis, alignment, metadata, and standard table/update tests."""

from __future__ import annotations

import builtins
from fnmatch import fnmatch
from hashlib import sha256
import io
import json
from pathlib import Path

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
import numpy as np
import pytest

from ml_peg import analysis, models
from ml_peg.analysis.amorphous_materials.AM26 import analyse_AM26 as am26
from ml_peg.analysis.utils import decorators
from ml_peg.app.utils.load import rebuild_table
from ml_peg.calcs.amorphous_materials.AM26.calc_AM26 import preserve_reference


def xyz_bytes(frames):
    """Serialize synthetic frames in memory using the actual Phase 2 format."""
    stream = io.StringIO()
    write(stream, frames, format="extxyz")
    return stream.getvalue().encode()


@pytest.fixture
def reference_frames():
    """Two unequal-sized frames per system, containing all nine AM26 elements."""
    frames = []
    for system, formula in zip(
        am26.SYSTEM_ORDER, ("C", "Si", "SiO2", "GeSbTe3", "LiPS"), strict=True
    ):
        for repeat in (1, 2):
            symbols = Atoms(formula).get_chemical_symbols() * repeat
            n_atoms = len(symbols)
            atoms = Atoms(
                symbols,
                positions=np.arange(3 * n_atoms).reshape(-1, 3) / 8,
                cell=[8, 9, 10],
                pbc=True,
            )
            per_atom = float(2 * (repeat - 1))
            atoms.info.update(
                label=f"{system}-{repeat}",
                system=system,
                energy_per_at=per_atom,
                virial=np.eye(3),
                cell_origin=np.zeros(3),
            )
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=n_atoms * per_atom,
                forces=np.tile([1.0, -1.0, 0.0], (n_atoms, 1)),
            )
            frames.append(atoms)
    return frames


@pytest.fixture
def output_frames(reference_frames):
    """Supply stored Phase 2 fields with a distinct raw bias for each system."""
    outputs = []
    for index, atoms in enumerate(reference_frames):
        bias = (am26.SYSTEM_ORDER.index(atoms.info["system"]) + 1) / 10
        output = preserve_reference(atoms, index)
        output.info.update(energy_failed=False, forces_failed=False)
        output.calc = SinglePointCalculator(
            output,
            energy=atoms.get_potential_energy() + len(atoms) * bias,
            forces=atoms.get_forces() + bias,
        )
        outputs.append(output)
    return outputs


@pytest.fixture
def memory_files(monkeypatch):
    """Exercise real JSON/ASE/helper I/O without creating or deleting test files."""
    root = Path(__file__).parents[1] / ".am26-analysis-memory"
    files = {}
    directories = {root}
    original_open = builtins.open
    original_exists = Path.exists
    original_glob = Path.glob
    original_mkdir = Path.mkdir
    original_is_file = Path.is_file

    def is_virtual(path):
        return isinstance(path, (str, Path)) and Path(path).is_relative_to(root)

    def opening(file, mode="r", *args, **kwargs):
        if not is_virtual(file):
            return original_open(file, mode, *args, **kwargs)
        path = Path(file)
        if "r" in mode and path not in files:
            raise FileNotFoundError(path)
        contents = b"" if "w" in mode else files.get(path, b"")
        stream = io.BytesIO(contents) if "b" in mode else io.StringIO(contents.decode())
        original_close = stream.close

        def close():
            if not stream.closed and any(flag in mode for flag in ("w", "a", "+")):
                data = stream.getvalue()
                files[path] = data.encode() if isinstance(data, str) else data
            original_close()

        stream.close = close
        return stream

    def exists(path):
        if is_virtual(path):
            return (
                path in files
                or path in directories
                or any(path in f.parents for f in files)
            )
        return original_exists(path)

    def glob(path, pattern, **kwargs):
        if is_virtual(path):
            return iter(
                sorted(
                    f for f in files if f.parent == path and fnmatch(f.name, pattern)
                )
            )
        return original_glob(path, pattern, **kwargs)

    def mkdir(path, *args, **kwargs):
        if is_virtual(path):
            directories.add(path)
            return None
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", opening)
    monkeypatch.setattr(io, "open", opening)
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(
        Path, "is_file", lambda p: p in files if is_virtual(p) else original_is_file(p)
    )
    return root, files


@pytest.fixture
def pipeline(monkeypatch, fake_models, memory_files, reference_frames, output_frames):
    """Use the standard fake registry and real helpers over in-memory files."""
    root, files = memory_files
    reference_path = root / "reference" / "data" / "am26.extxyz"
    files[reference_path] = xyz_bytes(reference_frames)
    monkeypatch.setattr(
        am26, "get_reference_path", lambda data_path=None: reference_path
    )
    monkeypatch.setattr(
        am26, "REFERENCE_SHA256", sha256(files[reference_path]).hexdigest()
    )
    monkeypatch.setattr(
        am26,
        "load_reference_structures",
        lambda data_path=None: read(reference_path, index=":", format="extxyz"),
    )
    monkeypatch.setattr(am26, "MODELS", ["model_1", "model_2"])
    monkeypatch.setattr(am26, "get_model_names", decorators.get_model_names)
    monkeypatch.setattr(am26, "CALC_PATH", root / "calculations")
    monkeypatch.setattr(am26, "OUT_PATH", root / "analysis")
    for model in am26.MODELS:
        files[am26.CALC_PATH / model / "am26_results.extxyz"] = xyz_bytes(output_frames)
    return files


def test_phase2_fields_raw_bias_and_five_system_metrics(
    reference_frames, output_frames
):
    """Consume stored fields and preserve raw per-atom/component errors."""
    result = am26.analyse_output(reference_frames, output_frames)
    assert result["status"] == "aligned"
    assert tuple(result["systems"]) == am26.SYSTEM_ORDER
    for index, system in enumerate(am26.SYSTEM_ORDER):
        values = result["systems"][system]
        bias = (index + 1) / 10
        assert values["energy_rmse"] == pytest.approx(bias)
        assert values["force_rmse"] == pytest.approx(bias)
        assert values["Energy RelRMSE"] == pytest.approx(bias * 100)
        assert values["Force RelRMSE"] == pytest.approx(bias / np.sqrt(2 / 3) * 100)
        assert values["energy_std"] == pytest.approx(1.0)
        assert values["force_std"] == pytest.approx(np.sqrt(2 / 3))
        assert values["expected_structure_count"] == 2
        assert values["valid_energy_count"] == values["valid_force_count"] == 2
        assert values["failed_energy_count"] == values["failed_force_count"] == 0
    assert result["overall"]["Mean Energy RelRMSE"] == pytest.approx(30)
    assert result["overall"]["Mean Force RelRMSE"] == pytest.approx(30 / np.sqrt(2 / 3))


@pytest.mark.parametrize("property_name", ["energy", "forces"])
@pytest.mark.parametrize("flag_only", [False, True])
def test_independent_failures_propagate(
    reference_frames, output_frames, property_name, flag_only
):
    """Failed flags and non-finite predictions cannot yield subset scores."""
    output = output_frames[4]  # First silica frame.
    if flag_only:
        output.info[f"{property_name}_failed"] = True
    elif property_name == "energy":
        output.calc.results["energy"] = np.nan
    else:
        output.calc.results["forces"][0, 0] = np.nan
    result = am26.analyse_output(reference_frames, output_frames)
    metric = "Energy" if property_name == "energy" else "Force"
    count_key = "energy" if property_name == "energy" else "force"
    other = "Force" if metric == "Energy" else "Energy"
    assert np.isnan(result["systems"]["silica"][f"{metric} RelRMSE"])
    assert np.isnan(result["overall"][f"Mean {metric} RelRMSE"])
    assert np.isfinite(result["overall"][f"Mean {other} RelRMSE"])
    assert result["systems"]["silica"][f"valid_{count_key}_count"] == 1
    assert result["systems"]["silica"][f"failed_{count_key}_count"] == 1


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing",
        "duplicate",
        "reorder",
        "index",
        "label",
        "system",
        "atom_count",
        "composition",
        "positions",
        "cell",
        "pbc",
        "ref_energy",
        "ref_energy_per_atom",
        "energy_per_at",
        "ref_forces_shape",
        "ref_forces_values",
        "virial",
        "cell_origin",
    ],
)
def test_alignment_mismatches_are_rejected(reference_frames, output_frames, mismatch):
    """Reject corrupted identity/reference data without reordering or shrinking it."""
    if mismatch == "missing":
        output_frames = output_frames[:-1]
    elif mismatch == "duplicate":
        output_frames[1] = output_frames[0]
    elif mismatch == "reorder":
        output_frames[0], output_frames[1] = output_frames[1], output_frames[0]
    elif mismatch == "index":
        output_frames[0].info["input_index"] = 9
    elif mismatch in {"label", "system"}:
        output_frames[0].info[mismatch] = "incorrect"
    elif mismatch == "atom_count":
        output_frames[4] = output_frames[4][:-1]
    elif mismatch == "composition":
        output_frames[0].numbers[0] = 14
    elif mismatch == "positions":
        output_frames[0].positions[0, 0] += 0.25
    elif mismatch == "cell":
        output_frames[0].cell[0, 0] += 1
    elif mismatch == "pbc":
        output_frames[0].pbc[0] = False
    elif mismatch in {"ref_energy", "ref_energy_per_atom", "energy_per_at"}:
        output_frames[0].info[mismatch] += 1
    elif mismatch == "ref_forces_shape":
        output_frames[0].arrays["ref_forces"] = np.zeros((1, 2))
    elif mismatch == "ref_forces_values":
        output_frames[0].arrays["ref_forces"][0, 0] += 1
    else:
        output_frames[0].info[mismatch].flat[0] += 1
    with pytest.raises(ValueError):
        am26.analyse_output(reference_frames, output_frames)


def test_existing_invalid_model_gets_nan_not_a_success_score(pipeline, output_frames):
    """Bad files invalidate the model while other models remain analysable."""
    pipeline[am26.CALC_PATH / "model_1" / "am26_results.extxyz"] = xyz_bytes(
        output_frames[:-1]
    )
    with pytest.warns(UserWarning, match="model_1: invalid output"):
        am26.run_analysis()
    table = json.loads(pipeline[am26.OUT_PATH / "am26_metrics_table.json"])
    rows = {r["id"]: r for r in table["data"]}
    assert rows["model_1"]["Score"] == "NaN"
    assert np.isnan(rows["model_1"]["Mean Energy RelRMSE"])
    assert np.isfinite(rows["model_2"]["Score"])
    bundle = json.loads(pipeline[am26.OUT_PATH / "system_metrics.json"])
    invalid = bundle["models"]["model_1"]
    assert invalid["status"] == "invalid_output"
    assert invalid["error"]
    assert invalid["systems"]["carbon"]["valid_energy_count"] is None


def test_standard_table_config_and_visible_columns(pipeline):
    """Use exactly two equally weighted PBE metrics and the standard unit labels."""
    am26.run_analysis()
    path = am26.OUT_PATH / "am26_metrics_table.json"
    saved = json.loads(pipeline[path])
    assert [c["id"] for c in saved["columns"]] == [
        "MLIP",
        "Score",
        "Mean Energy RelRMSE",
        "Mean Force RelRMSE",
    ]
    assert saved["weights"] == dict.fromkeys(am26.METRIC_NAMES, 1.0)
    for metric in am26.METRIC_NAMES:
        assert saved["thresholds"][metric] == {
            "good": 0.0,
            "bad": 100.0,
            "unit": "%",
            "level_of_theory": "PBE",
        }
        assert saved["metric_levels_of_theory"][metric] == "PBE"
    table = rebuild_table(path, id="AM26-test-table")
    assert [c["name"] for c in table.columns] == [
        "MLIP",
        "Score",
        "Mean Energy RelRMSE [%]",
        "Mean Force RelRMSE [%]",
    ]
    expected_score = ((1 - 30 / 100) + (1 - (30 / np.sqrt(2 / 3)) / 100)) / 2
    assert saved["data"][0]["Score"] == pytest.approx(expected_score)


def test_canonical_info_and_diagnostics_without_model_outputs(pipeline, monkeypatch):
    """Generate all nine elements even when no selected model has an output file."""
    monkeypatch.setattr(am26, "CALC_PATH", am26.CALC_PATH / "not-run")
    am26.run_analysis()
    info = json.loads(pipeline[am26.OUT_PATH / "info.json"])
    assert {e for frame in info["elements"] for e in frame} == {
        "C",
        "Si",
        "O",
        "Ge",
        "Sb",
        "Te",
        "Li",
        "P",
        "S",
    }
    assert len(info["elements"]) == len(info["label"]) == 10
    assert info["input_index"] == list(range(10))
    assert info["system"] == [s for s in am26.SYSTEM_ORDER for _ in range(2)]
    table = json.loads(pipeline[am26.OUT_PATH / "am26_metrics_table.json"])
    assert all(row["Mean Energy RelRMSE"] is None for row in table["data"])
    bundle = json.loads(pipeline[am26.OUT_PATH / "system_metrics.json"])
    assert all(data["status"] == "missing_output" for data in bundle["models"].values())


def test_diagnostic_artifact_values_counts_and_units(pipeline):
    """Retain the complete unscored five-system representation for later heatmaps."""
    am26.run_analysis()
    bundle = json.loads(pipeline[am26.OUT_PATH / "system_metrics.json"])
    assert bundle["schema_version"] == 1
    assert bundle["system_order"] == list(am26.SYSTEM_ORDER)
    assert bundle["system_labels"] == am26.SYSTEM_LABELS
    assert bundle["reference"] == {
        "sha256": am26.REFERENCE_SHA256,
        "structure_count": 10,
    }
    assert bundle["units"]["energy_rmse"] == "eV/atom"
    assert bundle["units"]["force_rmse"] == "eV/Å"
    assert bundle["units"]["RelRMSE"] == "%"
    for data in bundle["models"].values():
        assert data["status"] == "aligned"
        assert data["overall"]["Mean Energy RelRMSE"] == pytest.approx(30)
        assert set(data["systems"]) == set(am26.SYSTEM_ORDER)
        for record in data["systems"].values():
            assert set(record) == {
                "energy_rmse",
                "force_rmse",
                "energy_std",
                "force_std",
                "Energy RelRMSE",
                "Force RelRMSE",
                "expected_structure_count",
                "valid_energy_count",
                "failed_energy_count",
                "valid_force_count",
                "failed_force_count",
            }
            assert (
                record["expected_structure_count"] == record["valid_energy_count"] == 2
            )


@pytest.mark.parametrize("property_name", ["energy", "forces"])
def test_failed_property_table_and_json_semantics(
    pipeline, output_frames, property_name
):
    """A failed mean poisons the standard score without removing the other metric."""
    frame = output_frames[4]
    frame.info[f"{property_name}_failed"] = True
    pipeline[am26.CALC_PATH / "model_1" / "am26_results.extxyz"] = xyz_bytes(
        output_frames
    )
    am26.run_analysis()
    table = json.loads(pipeline[am26.OUT_PATH / "am26_metrics_table.json"])
    row = next(r for r in table["data"] if r["id"] == "model_1")
    metric = "Energy" if property_name == "energy" else "Force"
    other = "Force" if metric == "Energy" else "Energy"
    assert np.isnan(row[f"Mean {metric} RelRMSE"])
    assert np.isfinite(row[f"Mean {other} RelRMSE"])
    assert row["Score"] == "NaN"
    bundle = json.loads(pipeline[am26.OUT_PATH / "system_metrics.json"])
    data = bundle["models"]["model_1"]
    assert data["overall"][f"Mean {metric} RelRMSE"] is None
    key = "energy" if metric == "Energy" else "force"
    assert data["systems"]["silica"][f"failed_{key}_count"] == 1


def test_incremental_update_preserves_table_and_diagnostics(
    pipeline, output_frames, monkeypatch
):
    """Use existing table and model-bundle updates to retain unselected models."""
    am26.run_analysis()
    before_table = json.loads(pipeline[am26.OUT_PATH / "am26_metrics_table.json"])
    before_bundle = json.loads(pipeline[am26.OUT_PATH / "system_metrics.json"])
    changed = read(
        io.StringIO(xyz_bytes(output_frames).decode()), index=":", format="extxyz"
    )
    for frame in changed:
        frame.calc = SinglePointCalculator(
            frame,
            energy=frame.info["ref_energy"] + len(frame) * 0.7,
            forces=frame.arrays["ref_forces"] + 0.7,
        )
    pipeline[am26.CALC_PATH / "model_2" / "am26_results.extxyz"] = xyz_bytes(changed)
    monkeypatch.setattr(models, "current_models", "model_2")
    monkeypatch.setattr(analysis, "update_results", True)
    monkeypatch.setattr(am26, "MODELS", ["model_2"])
    am26.run_analysis()
    after_table = json.loads(pipeline[am26.OUT_PATH / "am26_metrics_table.json"])
    after_bundle = json.loads(pipeline[am26.OUT_PATH / "system_metrics.json"])
    assert after_bundle["models"]["model_1"] == before_bundle["models"]["model_1"]
    assert after_bundle["models"]["model_2"]["overall"][
        "Mean Energy RelRMSE"
    ] == pytest.approx(70)
    before_rows = {r["id"]: r for r in before_table["data"]}
    after_rows = {r["id"]: r for r in after_table["data"]}
    assert after_rows["model_1"] == before_rows["model_1"]
    assert after_rows["model_2"]["Mean Energy RelRMSE"] == pytest.approx(70)


def test_analysis_rejects_live_calculators(reference_frames, output_frames):
    """Never invoke a live calculator in place of reading stored predictions."""
    output_frames[0].calc = object()
    with pytest.raises(ValueError, match="stored single-point"):
        am26.analyse_output(reference_frames, output_frames)
