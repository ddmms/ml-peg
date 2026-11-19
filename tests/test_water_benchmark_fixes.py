"""Regression checks for water benchmark failure handling and interactions."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.utils import aml_md_analysis as aml
from ml_peg.analysis.utils import md_water_analysis as md
from ml_peg.analysis.utils.dipoles import get_z_dipoles_frames


def test_signed_score_and_comparison_data():
    """Signed cancellation is bounded and comparison data retains the curves."""
    ref = {"dipole": (np.array([0, 1]), np.array([-1.0, 1.0]))}
    pred = {"dipole": (np.array([0, 1]), np.array([1.0, -1.0]))}
    error = aml.compute_all_errors_vacf(ref, pred)["dipole"][2]
    assert aml.error_score_percentage(error) == 0
    zero = {"dipole": (np.array([0, 1]), np.zeros(2))}
    assert aml.compute_all_errors_vacf(zero, zero)["dipole"][2] == 0
    bundle = md.build_bar_data(
        {"m": [0]},
        {"ref": ref, "m": pred},
        ["m"],
        metric_key="dipole",
        metric_label="Dipole",
        ylabel="Dipole",
        xlabel="z",
        errors_fn=aml.compute_all_errors_vacf,
        ref_key=lambda _: "ref",
    )
    point = bundle["models"]["m"]["metrics"]["dipole"][0]
    assert point["data"]["ref"] == [-1.0, 1.0]
    assert point["data"]["pred"] == [1.0, -1.0]
    assert "figure" not in point["data"]


def test_periodic_water_neighbours():
    """Neighbour lists recover O-H bonds across the periodic boundary."""
    atoms = Atoms(
        "OH2",
        positions=[[0.1, 0, 1], [3.2, 0, 1], [0.1, 0.9, 1]],
        cell=[4, 4, 4],
        pbc=True,
    )
    totals, points = get_z_dipoles_frames([atoms], surface_idx_1=0, surface_idx_2=1)
    assert len(points) == 1
    assert np.all(np.isfinite(totals))


def test_density_profile_accepts_validated_frames():
    """The density computation accepts the trajectory returned by validation."""
    from ml_peg.analysis.surfaces.copper_water_interface.density_profile import (
        compute_density_profiles,
    )

    atoms = Atoms("CuOH", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    profiles = compute_density_profiles([atoms], surface_idx_1=0, surface_idx_2=1)
    assert profiles["O"][1].sum() == pytest.approx(1)
    assert profiles["H"][1].sum() == pytest.approx(1)


@pytest.mark.parametrize("metric", ["rdf", "vdos", "vacf", "density"])
def test_cell_to_bar_preserves_click_payload(tmp_path, metric):
    """Correct hover labels without changing scores or comparison click payloads."""
    import json

    from ml_peg.analysis.utils.decorators import cell_to_bar

    path = tmp_path / "bar.json"
    payload = {"figure": {"data": []}, "reference": [1, 2]}

    @cell_to_bar(filename=path, x_label="Species", y_label="Score")
    def build():
        return {
            "models": {
                "model": {
                    "metrics": {
                        metric: [{"label": "O-H", "value": 87.5, "data": payload}]
                    }
                }
            }
        }

    build()
    result = json.loads(path.read_text())
    trace = result["models"]["model"]["figures"][metric]["data"][0]
    assert trace["x"] == ["O-H"]
    assert trace["y"] == [87.5]
    assert trace["customdata"] == [["O-H", payload]]
    assert trace["hovertemplate"] == "<b>Label: </b>%{x}<br><b>Score: </b>%{y}<br>"


def test_water_slab_dipole_normalisation():
    """Keep the slab benchmark's charge, sign and area normalisation unchanged."""
    from ml_peg.analysis.utils.dipoles import get_z_dipoles

    atoms = Atoms(
        "OH2",
        positions=[[1, 1, 5], [1.8, 1, 5.5], [1, 1.8, 5.5]],
        cell=[4, 5, 12],
        pbc=True,
    )
    translated = atoms.copy()
    translated.translate([0, 0, 3])
    np.testing.assert_allclose(
        get_z_dipoles([atoms, translated], q=0.6), [0.015, 0.015]
    )


def test_molecular_dipole_profile_with_periodic_bond():
    """Check signed bond vectors, area and surface-relative oxygen height."""
    atoms = Atoms(
        "CuOH2",
        positions=[[2, 2, 2], [0.1, 1, 5], [7.2, 1, 5.2], [0.1, 1.8, 5.5]],
        cell=[8, 8, 12],
        pbc=True,
    )
    total, profile = get_z_dipoles_frames(
        [atoms], q=0.6, surface_idx_1=0, surface_idx_2=1
    )
    expected = 0.3 * (0.2 + 0.5) / 64
    np.testing.assert_allclose(total, [expected])
    np.testing.assert_allclose(profile, [[3, expected]])


@pytest.mark.parametrize(
    "module_name",
    [
        "ml_peg.app.aqueous_solutions.bulk_water.app_bulk_water",
        "ml_peg.app.aqueous_solutions.ice.app_ice",
        "ml_peg.app.surfaces.copper_water_interface.app_copper_water_interface",
    ],
)
def test_restored_comparison_renderers(module_name):
    """Each restored callback renders the curve payload without a shared helper."""
    from importlib import import_module

    module = import_module(module_name)
    data = {
        "x_values": [0, 1],
        "ref": [1, 2],
        "pred": [1, 1.5],
        "error": [0, 0.5],
        "xlabel": "Distance (nm)",
        "ylabel": "g(r)",
    }
    result = module.render_subplot_component(
        {"points": [{"customdata": ["O-H", data]}]}
    )
    assert result.children[1].src.startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    ("module_name", "benchmark"),
    [
        ("aqueous_solutions.bulk_water.calc_water", "bulk_water"),
        ("aqueous_solutions.ice.calc_ice", "ice"),
        (
            "surfaces.copper_water_interface.calc_copper_water_interface",
            "copper_water_interface",
        ),
    ],
)
def test_failed_md_then_successful_retry(module_name, benchmark, tmp_path, monkeypatch):
    """Keep the failure marker during each attempt and clear it after success."""
    from importlib import import_module
    from unittest.mock import Mock

    calc = import_module(f"ml_peg.calcs.{module_name}")
    marker = tmp_path / "model" / "md-failed"
    atoms = Atoms("CuOH", info={"fix_indices": [0]})
    monkeypatch.setattr(calc, "OUT_PATH", tmp_path)
    monkeypatch.setattr(calc, "download_s3_data", lambda **kwargs: tmp_path)
    monkeypatch.setattr(calc, "read", lambda path: atoms.copy())
    dynamics = Mock()
    monkeypatch.setattr(calc, "NVT", lambda **kwargs: dynamics)
    outcomes = iter([RuntimeError("interrupted MD"), None])

    def run():
        assert marker.exists()
        if error := next(outcomes):
            raise error

    dynamics.run.side_effect = run
    calculate = getattr(calc, f"test_{benchmark}")
    mlip = ("model", Mock())
    with pytest.warns(UserWarning, match="interrupted MD"):
        calculate(mlip)
    assert marker.exists()
    calculate(mlip)
    assert not marker.exists()


def test_corrupt_and_valid_production_frames(tmp_path):
    """A malformed frame is skipped and the next model is readable."""
    from ase.io import write

    from ml_peg.analysis.surfaces.copper_water_interface.density_profile import (
        read_production_frames,
    )

    bad, good = tmp_path / "bad", tmp_path / "good"
    bad.mkdir()
    good.mkdir()
    (bad / "md-traj.extxyz").write_text("2\n\nO 0 0 0\n")
    write(good / "md-traj.extxyz", Atoms("OH2"))
    with pytest.warns(UserWarning, match="Unable to read"):
        assert not read_production_frames(bad, 1)
    assert len(read_production_frames(good, 1)) == 1
    assert not read_production_frames(good, 2)


def test_incremental_bar_preserves_other_model(tmp_path, monkeypatch):
    """Update one model's bar while keeping another model's saved figure."""
    import json

    from ml_peg.analysis.utils import decorators

    filename = tmp_path / "bars.json"
    monkeypatch.setattr(decorators.analysis, "update_results", True)
    monkeypatch.setattr(decorators, "get_model_names", lambda: ["a", "b"])
    selected = "a"
    monkeypatch.setattr(decorators, "get_updated_models", lambda: {selected})

    @decorators.cell_to_bar(filename=filename)
    def build(score):
        point = {"label": "O-H", "value": score, "data": {}}
        return {"models": {selected: {"metrics": {"rdf": [point]}}}}

    build(10)
    selected = "b"
    saved_b = build(20)["models"]["b"]
    selected = "a"
    build(30)
    saved = json.loads(filename.read_text())["models"]
    assert saved["b"] == saved_b
    assert saved["a"]["figures"]["rdf"]["data"][0]["y"] == [30]


@pytest.mark.parametrize(("stop_step", "damage"), [(6, False), (14, False), (14, True)])
def test_resume_water_md_across_walltime(tmp_path, stop_step, damage):
    """Resume before or after equilibration with continuous production sampling."""
    from ase.calculators.emt import EMT
    from ase.io import read
    from janus_core.calculations.md import NVT

    from ml_peg.calcs.utils.water_md import water_md_restart

    def dynamics(directory, target):
        directory.mkdir(exist_ok=True)
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[10] * 3)
        atoms.calc = EMT()
        return NVT(
            **water_md_restart(atoms, directory, 8, target),
            steps=target,
            temp=330,
            timestep=0.5,
            stats_every=1,
            traj_every=2,
            restart_every=4,
            file_prefix=directory / "md",
            track_carbon=False,
        )

    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    dynamics(complete, 20).run()
    dynamics(resumed, stop_step).run()
    # Also exercise recovery from a partially written next frame.
    if stop_step >= 8:
        with (resumed / "md-traj.extxyz").open("a") as stream:
            stream.write("2\n")
    if damage:
        broken = resumed / "md-res-12.extxyz"
        broken.write_bytes(broken.read_bytes()[:200])
        with pytest.warns(UserWarning, match="Skipping unreadable checkpoint"):
            replay = dynamics(resumed, stop_step)
        assert replay.offset == 8
        replay.run()
        assert read(broken).info["step"] == 12

    restart = dynamics(resumed, 20)
    assert restart.offset == stop_step - 2
    checkpoint = read(resumed / f"md-res-{restart.offset}.extxyz")
    np.testing.assert_allclose(restart.struct.get_momenta(), checkpoint.get_momenta())
    restart.run()

    def saved_steps(directory):
        return [frame.info["step"] for frame in read(directory / "md-traj.extxyz", ":")]

    assert saved_steps(resumed) == saved_steps(complete) == list(range(8, 21, 2))
    assert np.loadtxt(resumed / "md-stats.dat")[:, 0].tolist() == list(range(21))


def test_dipole_extraction_failure_is_per_model(tmp_path, monkeypatch):
    """A model can fail dipole extraction even when its trajectory is readable."""
    import pickle
    from runpy import run_module

    from ase.io import write

    from ml_peg.calcs.utils import utils

    data = tmp_path / "copper_water_interface"
    data.mkdir()
    np.save(data / "ref_dipole_data.npy", [0.0])
    (data / "dipole_profile_reference.pkl").write_bytes(pickle.dumps({}))
    monkeypatch.setattr(utils, "download_s3_data", lambda **kwargs: tmp_path)
    module = run_module(
        "ml_peg.analysis.surfaces.copper_water_interface.analyse_copper_water_interface"
    )
    slab = Atoms("Cu120", positions=np.indices((6, 5, 4)).reshape(3, -1).T * 3)
    water = Atoms("OH2", positions=[[1, 1, 13], [1.8, 1, 13.4], [1, 1.8, 13.4]])
    good = slab + water
    good.set_cell([20, 20, 30])
    for model, atoms in (("bad", slab), ("good", good)):
        (tmp_path / model).mkdir()
        write(tmp_path / model / "md-traj.extxyz", atoms)

    stdev = module["stdev_dipole_z_deviation"].__wrapped__
    profile = module["created_dipole_profile"].__wrapped__
    stdev.__globals__.update(
        MODELS=["bad", "good"], CALC_PATH=tmp_path, RUN_STEPS=1, TRAJ_EVERY=1
    )
    with pytest.warns(UserWarning, match="z-dipoles for bad"):
        result = stdev()
    assert np.isnan(result["stdev_dipole_z_deviation"]["bad"])
    assert result["stdev_dipole_z_deviation"]["good"] == 0.0
    assert set(result["raw_dipoles"]) == {"ref", "good"}
    assert np.isfinite(result["raw_dipoles"]["good"]).all()
    with pytest.warns(UserWarning, match="dipole profile for bad"):
        profiles = profile()
    assert profiles["bad"] is md.FAILED
    z, integrated = profiles["good"]["dipole"]
    assert len(z) == len(integrated) > 0
    assert np.isfinite(integrated).all()
