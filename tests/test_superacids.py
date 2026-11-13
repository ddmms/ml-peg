"""Unit tests for superacids benchmark utilities."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import Trajectory
import numpy as np
import pytest

from ml_peg.analysis.superacids.HF_SbF5_density.analyse_HF_SbF5_density import (
    compute_density,
)
from ml_peg.analysis.superacids.HF_structure.analyse_HF_structure import (
    compute_r_factor,
)
from ml_peg.calcs.superacids.HF_SbF5_density.calc_HF_SbF5_density import (
    OUT_FREQ,
    read_restart,
)


def write_trajectory(traj_path: Path, separations: tuple[float, ...]) -> None:
    """
    Write a trajectory of two atoms with a total mass of 100 amu.

    Parameters
    ----------
    traj_path
        Path to write the trajectory to.
    separations
        Separation between the two atoms in each frame, in Angstrom.
    """
    with Trajectory(str(traj_path), "w") as traj:
        for separation in separations:
            atoms = Atoms(
                "H2",
                positions=[[0, 0, 0], [separation, 0, 0]],
                cell=[10, 10, 10],
                pbc=True,
            )
            atoms.set_masses([50.0, 50.0])
            traj.write(atoms)


def test_read_restart_without_trajectory(tmp_path: Path) -> None:
    """Test nothing is restarted from if the trajectory does not exist."""
    atoms, nsteps = read_restart(tmp_path / "does_not_exist.traj")

    assert atoms is None
    assert nsteps == 0


def test_read_restart_returns_last_frame(tmp_path: Path) -> None:
    """Test the last frame of the trajectory, and its step, are returned."""
    traj_path = tmp_path / "X_0.traj"
    write_trajectory(traj_path, (1.0, 2.0, 3.0))

    atoms, nsteps = read_restart(traj_path)

    # Frames are written at steps 0, OUT_FREQ and 2 * OUT_FREQ
    assert nsteps == 2 * OUT_FREQ
    assert atoms.get_distance(0, 1) == pytest.approx(3.0)


def test_read_restart_ignores_corrupt_trajectory(tmp_path: Path) -> None:
    """Test an unreadable trajectory is warned about, rather than raising."""
    traj_path = tmp_path / "X_0.traj"
    traj_path.write_bytes(b"not a trajectory")

    with pytest.warns(UserWarning, match="unreadable trajectory"):
        atoms, nsteps = read_restart(traj_path)

    assert atoms is None
    assert nsteps == 0


def test_compute_density(tmp_path: Path) -> None:
    """Test density is computed from the production half of the volumes."""
    traj_path = tmp_path / "X_0.traj"
    write_trajectory(traj_path, (1.0,))

    # Equilibration at 1000 A^3, production at 200 A^3
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text(
        "# step  volume_A3\n0  1000\n200  1000\n400  200\n600  200\n"
    )

    # 100 amu * 1.66053906660e-24 g/amu / (200 A^3 * 1e-24 cm^3/A^3)
    assert compute_density(traj_path, volume_path) == pytest.approx(0.830269533)


def test_compute_density_too_few_samples(tmp_path: Path) -> None:
    """Test NaN is returned if the production window is too short."""
    traj_path = tmp_path / "X_0.traj"
    write_trajectory(traj_path, (1.0,))

    # Half of two samples is one sample, fewer than MIN_SAMPLES
    volume_path = tmp_path / "volume.dat"
    volume_path.write_text("# step  volume_A3\n0  1000\n200  200\n")

    assert np.isnan(compute_density(traj_path, volume_path))


def test_compute_r_factor_identical_curves() -> None:
    """Test the R-factor of a structure factor against itself is zero."""
    q = np.arange(1.0, 4.01, 0.05)
    sq = np.exp(-q)

    assert compute_r_factor(q, sq, q, sq) == pytest.approx(0.0)


def test_compute_r_factor_constant_offset() -> None:
    """Test the R-factor of a structure factor offset by a known amount."""
    q = np.array([1.0, 2.0, 3.0, 4.0])
    sq_ref = np.ones(4)
    sq_calc = np.full(4, 1.5)

    # sum|1 - 1.5| = 2, sum|1| = 4
    assert compute_r_factor(q, sq_ref, q, sq_calc) == pytest.approx(0.5)


def test_compute_r_factor_without_overlap() -> None:
    """Test NaN is returned if the two structure factors do not overlap."""
    q_ref = np.array([1.0, 2.0, 3.0, 4.0])
    q_calc = np.array([10.0, 11.0, 12.0])

    assert np.isnan(compute_r_factor(q_ref, np.ones(4), q_calc, np.ones(3)))


@pytest.mark.parametrize("final_step, complete", [(600, False), (200000, True)])
def test_density_scoring_requires_complete_run(
    tmp_path, monkeypatch, final_step, complete
):
    """Partial density runs cannot receive an aggregate score."""
    import inspect

    from ml_peg.analysis.superacids.HF_SbF5_density import (
        analyse_HF_SbF5_density as analysis,
    )

    monkeypatch.setattr(analysis, "CALC_PATH", tmp_path)
    monkeypatch.setattr(analysis, "MODELS", ["test-model"])
    for system in analysis.SYSTEMS:
        folder = tmp_path / "test-model" / system
        folder.mkdir(parents=True)
        write_trajectory(folder / f"{system}.traj", (1.0,))
        (folder / "volume.dat").write_text(
            f"# step volume_A3\n0 1000\n200 1000\n400 200\n{final_step} 200\n"
        )
    densities = inspect.unwrap(analysis.densities)()
    errors = inspect.unwrap(analysis.density_errors)(densities)
    if complete:
        assert np.isfinite(errors["test-model"])
    else:
        assert errors["test-model"] is None
        # Partial data remain usable for diagnostic plots.
        time, values = analysis.compute_density_series(
            folder / f"{system}.traj", folder / "volume.dat"
        )
        assert len(values) == 4
        assert time[-1] == pytest.approx(0.3)


def test_structure_factor_rejects_incomplete_run(tmp_path):
    """Reject short MD before invoking the expensive structure-factor analysis."""
    from ml_peg.analysis.superacids.HF_structure.analyse_HF_structure import compute_sq

    traj_path = tmp_path / "NPT.traj"
    write_trajectory(traj_path, (1.0,) * 38)
    with pytest.raises(ValueError, match="Only 38 of 1001 expected frames"):
        compute_sq(traj_path, tmp_path / "sq")


def test_scatter_single_highlight_and_reference_line(tmp_path):
    """Shared shading is drawn once and existing reference lines remain visible."""
    import json

    from ml_peg.analysis.utils.decorators import plot_scatter

    output = tmp_path / "plot.json"

    @plot_scatter(
        filename=output,
        highlight_range={"Production": [50, 100]},
        horizontal_lines=[{"y": 10, "name": "Target", "color": "black"}],
    )
    def curves():
        return {"first": [[0, 100], [1, 2]], "second": [[0, 100], [2, 3]]}

    curves()
    figure = json.loads(output.read_text())
    shapes = figure["layout"]["shapes"]
    assert sum(shape["type"] == "rect" for shape in shapes) == 1
    assert sum(shape["type"] == "line" for shape in shapes) == 1
    assert figure["layout"]["yaxis"]["range"][1] > 10


@pytest.mark.parametrize("benchmark", ["HF_SbF5_density", "HF_structure"])
def test_charge_spin_on_fresh_and_restarted_md(tmp_path, monkeypatch, benchmark):
    """Calculators receive Python integer charge and spin on both MD paths."""
    import importlib

    from ase.calculators.lj import LennardJones
    from ase.io import write

    calc_module = importlib.import_module(
        f"ml_peg.calcs.superacids.{benchmark}.calc_{benchmark}"
    )

    class CheckedCalculator(LennardJones):
        def calculate(self, atoms=None, *args, **kwargs):
            assert type(atoms.info["charge"]) is int
            assert type(atoms.info["spin"]) is int
            assert (atoms.info["charge"], atoms.info["spin"]) == (0, 1)
            return super().calculate(atoms, *args, **kwargs)

    class Model:
        def get_calculator(self, precision):
            assert precision == "low"
            return CheckedCalculator()

        def add_d3_calculator(self, calc):
            return calc

    monkeypatch.setattr(calc_module, "OUT_PATH", tmp_path / "outputs")
    monkeypatch.setattr(calc_module, "N_MIN_STEPS", 1)
    monkeypatch.setattr(calc_module, "N_NPT_STEPS", 2)
    monkeypatch.setattr(calc_module, "OUT_FREQ", 1)
    monkeypatch.setattr(calc_module, "download_s3_data", lambda **kwargs: tmp_path)
    input_dir = tmp_path / benchmark
    if benchmark == "HF_SbF5_density":
        input_dir /= "X_0"
    input_dir.mkdir(parents=True)
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[15] * 3, pbc=True)
    write(input_dir / "start.xyz", atoms)
    (input_dir / "SQ_EXP.dat").write_text("1 1\n2 1\n")
    args = (("test-model", Model()),)
    if benchmark == "HF_SbF5_density":
        run = calc_module.test_hf_sbf5_density
        args += ("X_0",)
        traj = tmp_path / "outputs/test-model/X_0/X_0.traj"
    else:
        run = calc_module.test_hf_structure
        traj = tmp_path / "outputs/test-model/NPT.traj"
    run(*args)
    monkeypatch.setattr(calc_module, "N_NPT_STEPS", 4)
    run(*args)
    with Trajectory(str(traj)) as frames:
        assert len(frames) == 5


@pytest.mark.parametrize(
    "values, expected",
    [([1.0, 2.0, 3.0], 0.0), ([1.0, np.nan, 3.0], 1 / 3), ([np.nan] * 3, 1.0)],
)
def test_density_failure_fraction(tmp_path, monkeypatch, values, expected):
    """Failure counts all compositions and leaves unrun models unscored."""
    import inspect

    from ml_peg.analysis.superacids.HF_SbF5_density import (
        analyse_HF_SbF5_density as analysis,
    )

    monkeypatch.setattr(analysis, "CALC_PATH", tmp_path)
    monkeypatch.setattr(analysis, "MODELS", ["run", "absent"])
    (tmp_path / "run").mkdir()
    result = inspect.unwrap(analysis.failure)({"run": values, "absent": [np.nan] * 3})
    assert result == {"run": pytest.approx(expected), "absent": None}
    table = inspect.unwrap(analysis.metrics)({"run": None}, result)
    assert table["Failure"] == result
    assert table["MAPE"]["run"] is None


@pytest.mark.parametrize("value, expected", [(0.1, 0.0), (None, 1.0), (np.nan, 1.0)])
def test_structure_failure_fraction(tmp_path, monkeypatch, value, expected):
    """Unusable S(q) results count as failures, while absent models do not."""
    import inspect

    from ml_peg.analysis.superacids.HF_structure import analyse_HF_structure as analysis

    monkeypatch.setattr(analysis, "CALC_PATH", tmp_path)
    monkeypatch.setattr(analysis, "MODELS", ["run", "absent"])
    (tmp_path / "run").mkdir()
    errors = {
        "S(q) R-factor": {"run": value, "absent": None},
        "First Peak Position Error": {"run": 0.1, "absent": None},
    }
    result = inspect.unwrap(analysis.failure)(errors)
    assert result == {"run": expected, "absent": None}
    table = inspect.unwrap(analysis.metrics)(errors, result)
    assert table["Failure"] == result
    assert "Failure" not in errors


def test_scatter_hlines_compatibility(tmp_path):
    """Keep the shared hlines argument usable alongside horizontal_lines."""
    import json

    from ml_peg.analysis.utils.decorators import plot_scatter

    output = tmp_path / "hlines.json"

    @plot_scatter(
        filename=output,
        hlines={"Target": 3.0},
        horizontal_lines=[{"y": 4.0, "name": "Other target"}],
    )
    def curves():
        return {"model": [[0, 1], [1, 2]]}

    curves()
    figure = json.loads(output.read_text())
    lines = [shape for shape in figure["layout"]["shapes"] if shape["type"] == "line"]
    assert {line["y0"] for line in lines} == {3.0, 4.0}
    assert any(
        annotation["text"] == "Target" for annotation in figure["layout"]["annotations"]
    )
