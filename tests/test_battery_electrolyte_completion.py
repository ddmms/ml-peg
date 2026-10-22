"""Completion and missing-output regression checks for electrolyte densities."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
import numpy as np
import pytest

from ml_peg import app
from ml_peg.analysis.utils import decorators, utils
from ml_peg.models import get_models


@pytest.fixture
def analysis(monkeypatch, tmp_path):
    """Load analysis without requiring downloaded benchmark inputs."""
    monkeypatch.setattr(app, "APP_ROOT", tmp_path / "app")
    info = {
        "filenames": ["complete", "missing"],
        "exp_density": [1.0, 2.0],
        "salt": ["", ""],
        "solvent": ["", ""],
        "category": ["solvent", "solvent"],
        "concentration_M": [0, 0],
    }
    monkeypatch.setattr(utils, "get_struct_info", lambda **kwargs: info)
    monkeypatch.setattr(utils, "build_dispersion_name_map", lambda *args: {})
    monkeypatch.setattr(get_models, "get_model_names", lambda *args: ["test-model"])
    monkeypatch.setattr(decorators, "get_model_names", lambda *args: ["test-model"])
    path = Path(__file__).parents[1] / (
        "ml_peg/analysis/molecular_dynamics/battery_electrolyte_densities/"
        "analyse_battery_electrolyte_densities.py"
    )
    spec = importlib.util.spec_from_file_location("electrolyte_analysis_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CALC_PATH", tmp_path)
    monkeypatch.setattr(module, "EXPECTED_FRAMES", 4)
    monkeypatch.setattr(module, "EQUILIB_TIME_PS", 0.1)
    return module


def write_trajectory(path, frames=4, invalid=False):
    """Write a small trajectory with distinct equilibration/production volumes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(path, "w") as traj:
        for index in range(frames):
            atoms = Atoms("He", cell=[5 if index < 2 else 10] * 3, pbc=True)
            energy = np.nan if invalid and index == 0 else 0.0
            atoms.calc = SinglePointCalculator(
                atoms, energy=energy, forces=np.zeros((1, 3))
            )
            traj.write(atoms)


def test_complete_production_window(analysis, tmp_path):
    """Discard the boundary frame and average only the full production window."""
    path = tmp_path / "full.traj"
    write_trajectory(path)
    expected = 1e24 / units.mol * Atoms("He").get_masses().sum() / 1000
    assert analysis.compute_density(path) == pytest.approx(expected)


@pytest.mark.parametrize("frames", [0, 2, 3, 5])
def test_wrong_frame_count_is_not_scored(analysis, tmp_path, frames):
    """Neither truncated nor unexpectedly extended runs receive density results."""
    path = tmp_path / "partial.traj"
    write_trajectory(path, frames)
    assert np.isnan(analysis.compute_density(path))


def test_nonfinite_equilibration_is_not_stable(analysis, tmp_path):
    """An invalid saved state cannot be hidden by equilibration removal."""
    path = tmp_path / "invalid.traj"
    write_trajectory(path, invalid=True)
    assert np.isnan(analysis.compute_density(path))


def test_failure_status_overrides_complete_frames(analysis, tmp_path):
    """A failure after the final write must still invalidate the trajectory."""
    path = tmp_path / "failed.traj"
    write_trajectory(path)
    path.with_suffix(".status.json").write_text(json.dumps({"complete": False}))
    assert np.isnan(analysis.compute_density(path))


@pytest.mark.parametrize("exists", [False, True])
def test_unreadable_outputs(analysis, tmp_path, exists):
    """Missing and corrupt outputs warn and produce NaN."""
    path = tmp_path / "bad.traj"
    if exists:
        path.write_text("corrupt")
    with pytest.warns(UserWarning, match="Cannot score"):
        assert np.isnan(analysis.compute_density(path))


def test_missing_system_preserves_alignment_and_stability(analysis, tmp_path):
    """One of two systems yields 50% completion but no aggregate accuracy score."""
    path = tmp_path / "test-model" / "complete.traj"
    write_trajectory(path)
    with pytest.warns(UserWarning, match="Cannot score"):
        densities = inspect.unwrap(analysis.densities)()
    assert densities["ref"] == [1.0, 2.0]
    assert len(densities["test-model"]) == 2
    assert np.isfinite(densities["test-model"][0])
    assert np.isnan(densities["test-model"][1])
    assert inspect.unwrap(analysis.stable_runs)(densities) == {"test-model": 50.0}
    for function in (analysis.get_mae, analysis.get_rmse, analysis.get_mape):
        assert np.isnan(inspect.unwrap(function)(densities)["test-model"])
    assert (analysis.OUT_PATH / "test-model" / "complete.xyz").exists()
    assert not (analysis.OUT_PATH / "test-model" / "missing.xyz").exists()


@pytest.fixture
def calculation(monkeypatch):
    """Load a short real MD protocol without constructing model wrappers."""
    monkeypatch.setattr(get_models, "load_models", lambda *args: {})
    path = Path(__file__).parents[1] / (
        "ml_peg/calcs/molecular_dynamics/battery_electrolyte_densities/"
        "calc_battery_electrolyte_densities.py"
    )
    spec = importlib.util.spec_from_file_location("electrolyte_calc_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "NUM_MD_STEPS", 4)
    monkeypatch.setattr(module, "LOG_INTERVAL", 2)
    monkeypatch.setattr(module, "TIMESTEP", 0.01 * units.fs)
    return module


def test_resume_has_no_duplicate_frame_and_corruption_restarts(calculation, tmp_path):
    """Real ASE restart reaches the expected length and preserves corrupt output."""
    from ase.build import bulk
    from ase.calculators.lj import LennardJones

    atoms = bulk("Ar", cubic=True, a=5.26)
    atoms.info["exp_temperature"] = 298.2
    calc = LennardJones(epsilon=0.01, sigma=3.4)
    full = tmp_path / "full.traj"
    calculation.run_npt(atoms.copy(), calc, full)
    assert json.loads(full.with_suffix(".status.json").read_text())["complete"]
    with Trajectory(full) as traj:
        assert len(traj) == 3
        frames = list(traj)
    resume = tmp_path / "resume.traj"
    with Trajectory(resume, "w") as traj:
        for frame in frames[:2]:
            traj.write(frame)
    calculation.run_npt(atoms.copy(), calc, resume)
    with Trajectory(resume) as traj:
        assert len(traj) == 3
    assert json.loads(resume.with_suffix(".status.json").read_text())["complete"]
    corrupt = tmp_path / "corrupt.traj"
    corrupt.write_text("invalid trajectory")
    with pytest.warns(UserWarning, match="Cannot resume"):
        calculation.run_npt(atoms.copy(), calc, corrupt)
    backups = list(tmp_path.glob("corrupt.traj.invalid-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "invalid trajectory"
    with Trajectory(corrupt) as traj:
        assert len(traj) == 3


def test_md_exception_is_persisted(calculation, tmp_path, monkeypatch):
    """An MD exception leaves explicit failure status for later analysis."""
    from ase.build import bulk
    from ase.calculators.lj import LennardJones

    def fail(*args, **kwargs):
        raise RuntimeError("test MD failure")

    monkeypatch.setattr(calculation.IsotropicMTKNPT, "run", fail)
    atoms = bulk("Ar", cubic=True, a=5.26)
    atoms.info["exp_temperature"] = 298.2
    path = tmp_path / "failed.traj"
    with pytest.warns(UserWarning, match="Error running MD"):
        calculation.run_npt(atoms, LennardJones(), path)
    status = json.loads(path.with_suffix(".status.json").read_text())
    assert status == {"complete": False, "error": "test MD failure"}


def test_exported_metrics_preserve_failure(analysis, tmp_path):
    """A finite completion percentage must not hide missing accuracy in the table."""
    write_trajectory(tmp_path / "test-model" / "complete.traj")
    with pytest.warns(UserWarning, match="Cannot score"):
        densities = inspect.unwrap(analysis.densities)()
    mae = inspect.unwrap(analysis.get_mae)(densities)
    rmse = inspect.unwrap(analysis.get_rmse)(densities)
    mape = inspect.unwrap(analysis.get_mape)(densities)
    stability = inspect.unwrap(analysis.stable_runs)(densities)
    analysis.metrics.__wrapped__(mae, rmse, mape, stability)
    table_path = analysis.OUT_PATH / "battery_electrolyte_densities_metrics_table.json"
    table = json.loads(table_path.read_text())
    completion = json.loads((analysis.OUT_PATH / "figure_completion.json").read_text())
    assert completion["data"][0]["x"] == ["complete", "missing"]
    assert completion["data"][0]["y"] == [100, 0]
    row = table["data"][0]
    assert row["Stable runs"] == 50.0
    assert row["Score"] == "NaN"
