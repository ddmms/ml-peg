"""End-to-end runner tests against a fake gpumd, plus the inference.json schema."""

from __future__ import annotations

import json
from pathlib import Path

import ase.io
import jsonschema
import mlpeg_gpumd
import numpy as np
import pytest

SCHEMA = json.loads((Path(__file__).parent / "inference_schema.json").read_text())


def run(spec, run_id, out):
    """
    Invoke the runner through its CLI entry point.

    Parameters
    ----------
    spec
        Path to ``spec.json``.
    run_id
        Run identifier.
    out
        Output directory.

    Returns
    -------
    dict
        The parsed ``inference.json``.
    """
    argv = [
        "--spec",
        str(spec),
        "--run-id",
        run_id,
        "--precision",
        "mixed",
        "--out",
        str(out),
    ]
    assert mlpeg_gpumd.main(argv) == 0
    result = json.loads((out / "inference.json").read_text())
    jsonschema.validate(result, SCHEMA)
    return result


def test_ok_run(fake_engine, campaign, tmp_path, sio2_atoms):
    """A full run gives status ok, ordered timestamps, energies and validation."""
    out = tmp_path / "out"
    result = run(campaign, "sio2_quartz-n768-r1", out)
    assert result["status"] == "ok", result["error"]
    assert result["error"] is None
    assert result["model"]["name"] == "nep89"
    assert result["implementation"]["engine"] == "GPUMD"
    assert result["system"] == {"material": "sio2_quartz", "n_atoms": 768}
    assert result["steps"] == {"warmup": 100, "measured": 1000}

    ts = result["timestamps_unix"]
    assert ts["start"] < ts["setup_end"] < ts["warmup_end"] < ts["measure_end"]
    assert ts["measure_end"] - ts["warmup_end"] == pytest.approx(1.52943)
    assert ts["warmup_end"] - ts["setup_end"] == pytest.approx(0.201768)
    assert "warning" not in result["self_reported"]

    assert result["energy"]["total_start_eV"] == pytest.approx(-5973.596225614)
    assert result["energy"]["total_end_eV"] == pytest.approx(-5973.593674397)
    assert result["self_reported"]["gpumd_speed_atom_step_per_s"] == [
        380635.0,
        502147.0,
    ]
    assert result["implementation"]["settings"]["timestep_fs"] == 1.0
    assert result["implementation"]["settings"]["MN_radial"] == 274.0

    frame = ase.io.read(out / "validation.extxyz")
    np.testing.assert_allclose(frame.positions, sio2_atoms.positions, atol=1e-12)
    assert frame.get_potential_energy() == pytest.approx(-123.456)
    forces = frame.get_forces()
    assert forces.shape == (768, 3)
    assert forces[10] == pytest.approx([0.010, -0.020, 0.5])


def test_no_validation_file_when_not_requested(fake_engine, campaign, tmp_path):
    """Sizes without validate: true write no validation.extxyz."""
    spec = json.loads(campaign.read_text())
    spec["systems"][0]["sizes"][0]["validate"] = False
    campaign.write_text(json.dumps(spec))
    out = tmp_path / "out"
    assert run(campaign, "sio2_quartz-n768-r1", out)["status"] == "ok"
    assert not (out / "validation.extxyz").exists()


def test_oom(fake_engine, campaign, tmp_path):
    """A CUDA OOM yields status oom and exit 0."""
    fake_engine.setenv("FAKE_GPUMD_MODE", "oom")
    result = run(campaign, "sio2_quartz-n768-r1", tmp_path / "out")
    assert result["status"] == "oom"
    assert "out of memory" in result["error"]


def test_gpumd_error(fake_engine, campaign, tmp_path):
    """Another GPUMD failure yields status error with the stderr tail."""
    fake_engine.setenv("FAKE_GPUMD_MODE", "error")
    result = run(campaign, "sio2_quartz-n768-r1", tmp_path / "out")
    assert result["status"] == "error"
    assert "number of steps should be positive" in result["error"]
    assert "exited with 1" in result["error"]


def test_multigpu_is_error(fake_engine, campaign, tmp_path):
    """GPUMD seeing more than one GPU would use multi-GPU NEP: reject."""
    fake_engine.setenv("FAKE_GPUMD_MODE", "multigpu")
    result = run(campaign, "sio2_quartz-n768-r1", tmp_path / "out")
    assert result["status"] == "error"
    assert "2 GPUs" in result["error"]


def test_timestamp_warning(fake_engine, campaign, tmp_path):
    """A gap between the runs beyond 1 s is flagged in self_reported."""
    fake_engine.setenv("FAKE_GPUMD_MODE", "slow_gap")
    result = run(campaign, "sio2_quartz-n768-r1", tmp_path / "out")
    assert result["status"] == "ok"
    assert "warning" in result["self_reported"]


def test_unsupported_elements(fake_engine, campaign, tmp_path):
    """Elements absent from nep.txt yield unsupported_elements and exit 0."""
    result = run(campaign, "nacl_rocksalt-n768-r1", tmp_path / "out")
    assert result["status"] == "unsupported_elements"
    assert "Cl" in result["error"] and "Na" in result["error"]


def test_missing_structure_is_error(fake_engine, campaign, tmp_path):
    """A missing input file still produces inference.json with status error."""
    result = run(campaign, "sio2_quartz-n769-r1", tmp_path / "out")
    assert result["status"] == "error"


def test_unknown_size_is_error(fake_engine, campaign, tmp_path):
    """A run ID not in the spec still produces inference.json."""
    result = run(campaign, "sio2_quartz-n5-r1", tmp_path / "out")
    assert result["status"] == "error"
    assert "n_atoms=5" in result["error"]


def test_sha256_mismatch_is_error(fake_engine, campaign, tmp_path):
    """A structure that does not match its spec hash is not run."""
    spec = json.loads(campaign.read_text())
    spec["systems"][0]["sizes"][0]["sha256"] = "0" * 64
    campaign.write_text(json.dumps(spec))
    result = run(campaign, "sio2_quartz-n768-r1", tmp_path / "out")
    assert result["status"] == "error"
    assert "sha256 mismatch" in result["error"]


def test_schema_rejects_incomplete_ok():
    """The schema itself catches an ok result without timestamps."""
    bad = {
        "schema_version": "1.0",
        "run_id": "x-n1-r1",
        "status": "ok",
        "error": None,
        "model": {"name": "nep89", "weights_sha256": "a" * 64, "precision": "mixed"},
        "implementation": {
            "engine": "GPUMD",
            "engine_version": "5.9.1",
            "repo": "r",
            "commit": "c",
            "settings": {},
        },
        "system": {"material": "x", "n_atoms": 1},
        "timestamps_unix": {
            "start": 0.0,
            "setup_end": None,
            "warmup_end": None,
            "measure_end": None,
        },
        "steps": {"warmup": 100, "measured": 1000},
        "energy": {"total_start_eV": 0.0, "total_end_eV": 0.0},
        "self_reported": {},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, SCHEMA)


def test_scratch_on_tmp_is_removed(fake_engine, campaign, tmp_path):
    """The scratch lives on /tmp when it fits and is removed afterwards."""
    out = tmp_path / "out"
    result = run(campaign, "sio2_quartz-n768-r1", out)
    assert result["status"] == "ok"
    assert result["self_reported"]["scratch_dir"].startswith(str(mlpeg_gpumd.TMP_DIR))
    assert not any(mlpeg_gpumd.TMP_DIR.iterdir())
    assert sorted(p.name for p in out.iterdir()) == [
        "inference.json",
        "validation.extxyz",
    ]


@pytest.mark.parametrize("mode", ["ok", "oom"])
def test_scratch_falls_back_to_out(fake_engine, campaign, tmp_path, mode):
    """A too-small /tmp moves the scratch under the out dir, cleaned up after."""
    fake_engine.setenv("FAKE_GPUMD_MODE", mode)
    fake_engine.setattr(mlpeg_gpumd, "SCRATCH_BYTES_PER_ATOM", 1 << 40)
    out = tmp_path / "out"
    result = run(campaign, "sio2_quartz-n768-r1", out)
    assert result["status"] == mode
    assert mlpeg_gpumd.OUT_SCRATCH_NAME in result["self_reported"]["scratch_dir"]
    assert not (out / mlpeg_gpumd.OUT_SCRATCH_NAME).exists()
