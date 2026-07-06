"""Tests for the completion marker migration script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_peg import models
from ml_peg.calcs.utils import completion

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "migrate_completion_markers.py"
_spec = importlib.util.spec_from_file_location(
    "migrate_completion_markers", SCRIPT_PATH
)
migrate_markers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_markers)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """Fake calc directory, checkpoint, models file and cached data file."""
    calc_dir = tmp_path / "calc_fake"
    calc_dir.mkdir()
    (calc_dir / "calc_fake.py").write_text("A = 1\n")

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("weights")

    data_file = tmp_path / "data.zip"
    data_file.write_text("data")

    models_yml = tmp_path / "models.yml"
    models_yml.write_text(
        f"model-x:\n  kwargs:\n    model_path: {checkpoint}\n"
        "model-plain:\n  class_name: mace\n"
    )
    monkeypatch.setattr(models, "models_file", models_yml)

    return SimpleNamespace(
        calcs_dir=tmp_path,
        calc_dir=calc_dir,
        out_path=calc_dir / "outputs",
        checkpoint=checkpoint,
        data_file=data_file,
        models_yml=models_yml,
    )


def write_marker(
    calc_dir: Path, model_name: str, fingerprint: str, data_files: list[str]
) -> Path:
    """
    Write a completion marker with a single test entry.

    Parameters
    ----------
    calc_dir
        Fake benchmark directory to write the marker under.
    model_name
        Name of the model the marker belongs to.
    fingerprint
        Fingerprint to store in the entry.
    data_files
        Data file paths to store in the entry.

    Returns
    -------
    Path
        Path of the written marker file.
    """
    marker = calc_dir / "outputs" / model_name / completion.MARKER_FILENAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    content = {"test_fake": {"fingerprint": fingerprint, "data_files": data_files}}
    marker.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf8")
    return marker


def test_old_marker_migrated_with_backup(scene):
    """Test old-style markers are migrated and backed up exactly."""
    legacy = migrate_markers.legacy_fingerprint(scene.calc_dir, "model-x")
    marker = write_marker(scene.calc_dir, "model-x", legacy, [str(scene.data_file)])
    original = marker.read_text()

    new = completion.calc_fingerprint(scene.calc_dir, "model-x")
    assert not completion.is_complete(scene.out_path, "model-x", "test_fake", new)

    counts = migrate_markers.migrate(scene.calcs_dir)
    assert counts["migrated"] == 1
    assert counts["stale"] == 0

    assert completion.is_complete(scene.out_path, "model-x", "test_fake", new)
    content = json.loads(marker.read_text())
    assert content["test_fake"]["data_files"] == [str(scene.data_file)]
    backup = marker.parent / migrate_markers.BACKUP_FILENAME
    assert backup.read_text() == original


def test_second_run_is_noop_and_keeps_backup(scene):
    """Test a second run changes nothing and never clobbers the backup."""
    legacy = migrate_markers.legacy_fingerprint(scene.calc_dir, "model-x")
    marker = write_marker(scene.calc_dir, "model-x", legacy, [])
    backup = marker.parent / migrate_markers.BACKUP_FILENAME
    backup.write_text("pre-existing backup")

    migrate_markers.migrate(scene.calcs_dir)
    assert backup.read_text() == "pre-existing backup"

    migrated = marker.read_text()
    counts = migrate_markers.migrate(scene.calcs_dir)
    assert counts["migrated"] == 0
    assert counts["current"] == 1
    assert marker.read_text() == migrated
    assert backup.read_text() == "pre-existing backup"


def test_stale_entry_left_untouched(scene):
    """Test entries matching neither scheme are left stale to rerun."""
    marker = write_marker(scene.calc_dir, "model-x", "bogus", [])
    original = marker.read_text()

    counts = migrate_markers.migrate(scene.calcs_dir)
    assert counts["stale"] == 1
    assert counts["migrated"] == 0
    assert marker.read_text() == original
    assert not (marker.parent / migrate_markers.BACKUP_FILENAME).exists()

    new = completion.calc_fingerprint(scene.calc_dir, "model-x")
    assert not completion.is_complete(scene.out_path, "model-x", "test_fake", new)


def test_model_without_local_files_is_noop(scene):
    """Test markers of models with no local files are already current."""
    fingerprint = completion.calc_fingerprint(scene.calc_dir, "model-plain")
    assert fingerprint == migrate_markers.legacy_fingerprint(
        scene.calc_dir, "model-plain"
    )
    marker = write_marker(scene.calc_dir, "model-plain", fingerprint, [])
    original = marker.read_text()

    counts = migrate_markers.migrate(scene.calcs_dir)
    assert counts["current"] == 1
    assert counts["migrated"] == 0
    assert marker.read_text() == original
    assert not (marker.parent / migrate_markers.BACKUP_FILENAME).exists()


def test_main_models_file_option(scene, monkeypatch, capsys):
    """Test --models-file selects the model definitions like pytest's option."""
    legacy = migrate_markers.legacy_fingerprint(scene.calc_dir, "model-x")
    marker = write_marker(scene.calc_dir, "model-x", legacy, [])

    # Without the right models file, model-x's config is empty and the marker
    # cannot migrate
    monkeypatch.setattr(models, "models_file", scene.calcs_dir / "missing.yml")
    monkeypatch.setattr(migrate_markers, "CALCS_DIR", scene.calcs_dir)

    migrate_markers.main(["--models-file", str(scene.models_yml)])
    output = capsys.readouterr().out
    assert str(scene.models_yml) in output
    assert "1 entr(y/ies) migrated" in output
    assert json.loads(marker.read_text())["test_fake"][
        "fingerprint"
    ] == completion.calc_fingerprint(scene.calc_dir, "model-x")


def test_corrupt_marker_reported_not_fatal(scene, capsys):
    """Test corrupt markers are reported while other markers still migrate."""
    broken = scene.calc_dir / "outputs" / "model-broken" / completion.MARKER_FILENAME
    broken.parent.mkdir(parents=True)
    broken.write_text("not json")

    legacy = migrate_markers.legacy_fingerprint(scene.calc_dir, "model-x")
    write_marker(scene.calc_dir, "model-x", legacy, [])

    counts = migrate_markers.migrate(scene.calcs_dir)
    assert counts["unreadable"] == 1
    assert counts["migrated"] == 1
    assert broken.read_text() == "not json"
    assert "model-broken" in capsys.readouterr().out
