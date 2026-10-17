"""Tests for the optimised inference speed benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.spacegroup import crystal
import numpy as np
import pytest

from ml_peg.analysis.molecular_dynamics.optimized_inference_speed.aggregate import (
    analyse_campaign,
    derived_quantities,
)
from ml_peg.analysis.molecular_dynamics.optimized_inference_speed.validate import (
    check_drift,
    check_elements,
    check_identity,
    check_schema,
    check_static,
    check_timing,
    get_supported_elements,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.build import (
    check_labels,
    parse_def_labels,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.contract import (
    CONTRIB_ROOT,
    SPEC_TEMPLATE,
    InferenceResult,
    build_invocation,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.run import (
    load_container_models,
    plan_runs,
    run_campaign,
)
from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.stage import (
    choose_supercell,
    sha256_file,
    stage_campaign,
)
from ml_peg.models.mock import MockCalculator

TARGETS = [70, 250, 600, 1200]
MOCK_SHA = load_container_models()["_mock"]["weights_sha256"]
NEP89_FILE = Path.home() / ".cache/ml_peg/models/nep/nep89_20250409.txt"


def quartz() -> Atoms:
    """
    Build a 9-atom alpha-quartz cell.

    Returns
    -------
    Atoms
        Alpha-quartz.
    """
    return crystal(
        ["Si", "O"],
        basis=[(0.4697, 0, 0), (0.4135, 0.2669, 0.1191)],
        spacegroup=152,
        cellpar=[4.916, 4.916, 5.405, 90, 90, 120],
    )


@pytest.fixture
def template(tmp_path: Path) -> Path:
    """
    Write a spec template with small size ladders.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    Returns
    -------
    Path
        Template path.
    """
    spec = json.loads(SPEC_TEMPLATE.read_text())
    for system in spec["systems"]:
        system["sizes"] = [{"n_atoms": n} for n in TARGETS]
    path = tmp_path / "template.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """
    Write a quartz base cell, leaving the other materials missing.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    Returns
    -------
    Path
        Directory with ``sio2_quartz.extxyz``.
    """
    path = tmp_path / "base"
    path.mkdir()
    write(path / "sio2_quartz.extxyz", quartz())
    return path


@pytest.fixture
def campaign(tmp_path: Path, base_dir: Path, template: Path) -> Path:
    """
    Stage a small campaign.

    Parameters
    ----------
    tmp_path
        Temporary directory.
    base_dir
        Directory with base cells.
    template
        Spec template.

    Returns
    -------
    Path
        Campaign directory.
    """
    with pytest.warns(UserWarning):
        stage_campaign(base_dir, tmp_path / "campaign", template=template)
    return tmp_path / "campaign"


def make_result(**overrides) -> dict:
    """
    Build a valid ``inference.json`` for the mock.

    Parameters
    ----------
    **overrides
        Top-level fields to replace.

    Returns
    -------
    dict
        Result dictionary.
    """
    result = {
        "schema_version": "1.0",
        "run_id": "sio2_quartz-n72-r1",
        "status": "ok",
        "error": None,
        "model": {"name": "mock", "weights_sha256": MOCK_SHA, "precision": "fp32"},
        "implementation": {"engine": "mock"},
        "system": {"material": "sio2_quartz", "n_atoms": 72},
        "timestamps_unix": {
            "start": 100.0,
            "setup_end": 101.0,
            "warmup_end": 102.0,
            "measure_end": 112.0,
        },
        "steps": {"warmup": 100, "measured": 1000},
        "energy": {"total_start_eV": -10.0, "total_end_eV": -10.0},
    }
    result.update(overrides)
    return result


RECORD = {
    "run_id": "sio2_quartz-n72-r1",
    "precision": "fp32",
    "n_atoms": 72,
    "labels": {"mlpeg.model": "mock", "mlpeg.weights_sha256": MOCK_SHA},
}
CONTAINER = {"mlpeg_model": "mock", "weights_sha256": MOCK_SHA}


def test_choose_supercell_matches_template_ladder():
    """Quartz repeats reproduce the template's cubic ladder."""
    lengths = quartz().cell.lengths()
    spec = json.loads(SPEC_TEMPLATE.read_text())
    sizes = spec["systems"][0]["sizes"]
    for size in sizes:
        assert list(choose_supercell(lengths, 9, size["n_atoms"])) == size["supercell"]


def test_choose_supercell_near_cubic():
    """Repeats keep box edges within 15% for an anisotropic cell."""
    lengths = np.array([3.0, 4.0, 9.0])
    for target in (1_000, 64_000):
        repeats = choose_supercell(lengths, 2, target)
        edges = np.array(repeats) * lengths
        assert edges.max() / edges.min() <= 1.15


def test_stage(campaign: Path):
    """Staging writes correct sizes, checksums, momenta and flags."""
    spec = json.loads((campaign / "in" / "spec.json").read_text())
    assert [s["material"] for s in spec["systems"]] == ["sio2_quartz"]
    system = spec["systems"][0]
    assert system["elements"] == ["O", "Si"]
    assert system["contains_hydrogen"] is False
    n_base = len(quartz())
    assert system["base_cell_atoms"] == n_base
    sizes = system["sizes"]
    assert [s["n_atoms"] for s in sizes] == sorted(s["n_atoms"] for s in sizes)
    assert [s["validate"] for s in sizes] == [True] + [False] * (len(sizes) - 1)
    # Every size is far below 16,000 atoms, so the largest is the reference
    assert [s["reference"] for s in sizes] == [False] * (len(sizes) - 1) + [True]
    for size in sizes:
        path = campaign / "in" / size["file"]
        assert sha256_file(path) == size["sha256"]
        atoms = read(path)
        assert len(atoms) == size["n_atoms"] == n_base * np.prod(size["supercell"])
        assert "momenta" in atoms.arrays
        assert "masses" in atoms.arrays
        assert np.allclose(atoms.get_momenta().sum(axis=0), 0, atol=1e-6)
        assert atoms.get_temperature() > 0
    assert spec["campaign"]["id"].startswith("campaign-")
    assert spec["campaign"]["hardware_id"] is None


def test_stage_warns_for_missing_materials(tmp_path, base_dir, template):
    """Each missing base cell gives a warning, not an error."""
    with pytest.warns(UserWarning) as record:
        stage_campaign(base_dir, tmp_path / "campaign", template=template)
    messages = " ".join(str(w.message) for w in record)
    for material in ("nacl_rocksalt", "h2o_liquid", "uio66"):
        assert f"No base cell for '{material}'" in messages


def test_stage_is_deterministic(tmp_path, base_dir, template):
    """Staging twice gives byte-identical files."""
    with pytest.warns(UserWarning):
        stage_campaign(base_dir, tmp_path / "a", template=template)
        stage_campaign(base_dir, tmp_path / "b", template=template)
    files_a = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*"))
    files_b = sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*"))
    assert files_a == files_b
    for rel in files_a:
        if (tmp_path / "a" / rel).is_file():
            assert (tmp_path / "a" / rel).read_bytes() == (
                tmp_path / "b" / rel
            ).read_bytes()


def test_plan_runs_sizes_ascend(campaign: Path):
    """Sizes ascend within each model, precision and material."""
    spec = json.loads((campaign / "in" / "spec.json").read_text())
    runs = plan_runs(spec, {"a": ["fp32", "fp64"], "b": ["fp32"]}, repeats=3, seed=1)
    n_sizes = len(spec["systems"][0]["sizes"])
    assert len(runs) == 3 * 3 * n_sizes
    groups: dict[tuple, list[int]] = {}
    for run in runs:
        groups.setdefault((run.model, run.precision), []).append(run.n_atoms)
    for sizes in groups.values():
        assert sizes == sorted(sizes)


def test_build_invocation_matches_contract():
    """The Apptainer command follows the contract exactly."""
    command = build_invocation(
        Path("m.sif"), Path("/c/in"), Path("/c/out/r"), "x-n1-r1", "fp32"
    )
    assert command == [
        "apptainer", "run", "--nv", "--containall", "--cleanenv", "--net",
        "--network", "none", "--bind", "/c/in:/mlpeg/in:ro",
        "--bind", "/c/out/r:/mlpeg/out",
        "--env", "CUDA_VISIBLE_DEVICES=0,OMP_NUM_THREADS=8",
        "m.sif", "--spec", "/mlpeg/in/spec.json", "--run-id", "x-n1-r1",
        "--precision", "fp32", "--out", "/mlpeg/out",
    ]  # fmt: skip


def test_container_labels():
    """Definitions of contributed containers carry the required labels."""
    for name in ("_mock", "nep89_gpumd"):
        labels = parse_def_labels(CONTRIB_ROOT / name / "container.def")
        assert check_labels(labels) == []
        expected = load_container_models()[name]
        assert labels["mlpeg.weights_sha256"] == expected["weights_sha256"]
        assert labels["mlpeg.model"] == expected["mlpeg_model"]
    assert check_labels({"mlpeg.model": "x"}) == [
        "missing label mlpeg.precisions",
        "missing label mlpeg.weights_sha256",
        "missing label mlpeg.contract_version",
    ]


def test_oom_skips_larger_sizes(campaign: Path):
    """After an oom, larger sizes of that material are skipped."""
    spec = json.loads((campaign / "in" / "spec.json").read_text())
    sizes = [s["n_atoms"] for s in spec["systems"][0]["sizes"]]
    records = run_campaign(
        campaign,
        models=["_mock"],
        runtime="local",
        precisions=["fp32"],
        repeats=2,
        container_env=[f"MLPEG_MOCK_FAIL=oom:*-n{sizes[1]}-r2"],
    )
    status = {r["run_id"]: r["status"] for r in records}
    assert status[f"sio2_quartz-n{sizes[0]}-r1"] == "ok"
    assert status[f"sio2_quartz-n{sizes[1]}-r1"] == "ok"
    assert status[f"sio2_quartz-n{sizes[1]}-r2"] == "oom"
    for n_atoms in sizes[2:]:
        for repeat in (1, 2):
            assert status[f"sio2_quartz-n{n_atoms}-r{repeat}"] == "skipped"
    assert all(r["precision"] == "fp32" for r in records)


def test_check_schema(tmp_path: Path):
    """Schema check accepts valid results and rejects invalid ones."""
    assert check_schema(tmp_path)[1].reason == "inference.json missing"
    (tmp_path / "inference.json").write_text(json.dumps(make_result()))
    result, check = check_schema(tmp_path)
    assert check.ok
    assert result.model.name == "mock"
    (tmp_path / "inference.json").write_text(json.dumps(make_result(energy=None)))
    assert not check_schema(tmp_path)[1].ok
    (tmp_path / "inference.json").write_text(json.dumps(make_result(status="great")))
    assert not check_schema(tmp_path)[1].ok
    oom = make_result(status="oom", steps=None, energy=None)
    oom["timestamps_unix"] = {"start": 1.0}
    (tmp_path / "inference.json").write_text(json.dumps(oom))
    assert check_schema(tmp_path)[1].ok


def test_check_identity():
    """Identity check needs matching weights, model, precision and run."""
    result = InferenceResult.model_validate(make_result())
    assert check_identity(result, RECORD, CONTAINER).ok
    bad_hash = InferenceResult.model_validate(
        make_result(model={"name": "mock", "weights_sha256": "0", "precision": "fp32"})
    )
    assert "weights_sha256" in check_identity(bad_hash, RECORD, CONTAINER).reason
    assert not check_identity(result, RECORD, CONTAINER | {"mlpeg_model": "x"}).ok
    assert not check_identity(result, RECORD | {"precision": "fp64"}, CONTAINER).ok
    assert not check_identity(result, RECORD | {"n_atoms": 9}, CONTAINER).ok


def test_check_elements():
    """unsupported_elements must agree with the model's elements."""
    ok = InferenceResult.model_validate(make_result())
    unsupported = InferenceResult.model_validate(
        make_result(status="unsupported_elements", steps=None, energy=None)
    )
    assert check_elements(ok, ["O", "Si"], {"O", "Si"}).ok
    assert check_elements(ok, ["O", "Si"], None).ok
    assert not check_elements(ok, ["O", "Si"], {"O"}).ok
    assert check_elements(unsupported, ["O", "Si"], {"O"}).ok
    assert not check_elements(unsupported, ["O", "Si"], None).ok


def test_supported_elements_from_dataset():
    """NEP89 elements are read from the element coverage data."""
    elements = get_supported_elements({"elements_from_dataset": "NEP89"})
    assert {"Si", "O", "Na", "Cl", "H", "C", "Zr"} <= elements
    assert get_supported_elements({"elements": None}) is None


def test_check_drift():
    """Drift above 1 meV/atom fails."""
    assert check_drift(InferenceResult.model_validate(make_result()), 72).ok
    drift = make_result(energy={"total_start_eV": -10.0, "total_end_eV": -9.9})
    assert not check_drift(InferenceResult.model_validate(drift), 72).ok


@pytest.mark.parametrize(
    ("stamps", "steps", "wall", "reason"),
    [
        (None, 1000, 20.0, ""),
        ({"warmup_end": 113.0}, 1000, 20.0, "not monotonic"),
        (None, 1000, 5.0, "exceeds wall time"),
        (None, 10, 20.0, "10 measured steps"),
    ],
)
def test_check_timing(stamps, steps, wall, reason):
    """
    Timing must be monotonic, within the wall time, with enough steps.

    Parameters
    ----------
    stamps
        Timestamps to override.
    steps
        Number of measured steps.
    wall
        External wall time.
    reason
        Expected failure reason, or empty if the check should pass.
    """
    result = make_result(steps={"warmup": 100, "measured": steps})
    result["timestamps_unix"].update(stamps or {})
    check = check_timing(InferenceResult.model_validate(result), wall)
    assert check.ok == (not reason)
    assert reason in check.reason


class ConstantForceCalculator(Calculator):
    """Calculator with zero energy and constant forces."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, force: float):
        """
        Set the force.

        Parameters
        ----------
        force
            Value of every force component, in eV/Å.
        """
        super().__init__()
        self.force = force

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """
        Calculate constant forces.

        Parameters
        ----------
        atoms
            Structure.
        properties
            Properties to calculate.
        system_changes
            Changes since the last calculation.
        """
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.full((len(self.atoms), 3), self.force),
        }


def test_check_static(tmp_path: Path):
    """Static accuracy uses precision-dependent tolerances."""
    structure = quartz().repeat(2)
    path = tmp_path / "validation.extxyz"
    assert not check_static(path, structure, MockCalculator(), "fp32").ok

    frame = structure.copy()
    frame.calc = ConstantForceCalculator(0.005)
    frame.get_forces()
    write(path, frame)
    # 5 meV/Å force RMSE: within fp32's 10 meV/Å, outside fp64's 1 meV/Å
    assert check_static(path, structure, MockCalculator(), "fp32").ok
    check = check_static(path, structure, MockCalculator(), "fp64")
    assert "force RMSE" in check.reason

    moved = structure.copy()
    moved.positions += 0.1
    moved.calc = ConstantForceCalculator(0.0)
    moved.get_forces()
    write(path, moved)
    check = check_static(path, structure, MockCalculator(), "fp32")
    assert "positions differ" in check.reason


def test_derived_quantities():
    """Speed metrics follow from the timestamps."""
    derived = derived_quantities(
        {"start": 0.0, "setup_end": 2.0, "warmup_end": 3.0, "measure_end": 13.0},
        measured_steps=1000,
        n_atoms=10_000,
        timestep_fs=0.5,
    )
    assert derived["time_per_step_s"] == pytest.approx(0.01)
    assert derived["us_per_atom_step"] == pytest.approx(1.0)
    assert derived["atom_steps_per_s"] == pytest.approx(1e6)
    # 100 steps/s * 0.5 fs * 86400 s = 4.32 ns/day
    assert derived["ns_per_day"] == pytest.approx(4.32)
    assert derived["setup_time_s"] == pytest.approx(2.0)
    assert derived["warmup_time_s"] == pytest.approx(1.0)


def test_end_to_end_local(campaign: Path, tmp_path: Path):
    """Run the mock locally with planted failures, then analyse."""
    spec = json.loads((campaign / "in" / "spec.json").read_text())
    small, medium, *_ = [s["n_atoms"] for s in spec["systems"][0]["sizes"]]
    faults = {
        f"drift:*-n{small}-r1": ("ok", False, "NVE drift"),
        f"badforces:*-n{small}-r2": ("ok", False, "force RMSE"),
        f"badhash:*-n{small}-r3": ("ok", False, "weights_sha256 mismatch"),
        f"fewsteps:*-n{medium}-r1": ("ok", False, "measured steps"),
        f"badtime:*-n{medium}-r2": ("ok", False, "not monotonic"),
        f"unsupported_elements:*-n{medium}-r3": (
            "unsupported_elements",
            False,
            "reported unsupported_elements",
        ),
    }
    env = "MLPEG_MOCK_FAIL=" + ";".join(faults)
    run_campaign(
        campaign, models=["_mock"], runtime="local", precisions=["fp32"],
        container_env=[env, "MLPEG_MOCK_PHASE_S=0.01"],
    )  # fmt: skip

    results = analyse_campaign(
        campaign, app_dir=tmp_path / "app", reference_getter=lambda _: MockCalculator()
    )
    runs = results["runs"].set_index("run_id")
    for fault, (status, valid, reason) in faults.items():
        run_id = f"sio2_quartz-n{fault.split('-n')[1].split('-r')[0]}"
        run_id += f"-r{fault[-1]}"
        assert runs.loc[run_id, "status"] == status
        assert runs.loc[run_id, "valid"] == valid
        assert reason in runs.loc[run_id, "reasons"]
    # Every run without a planted fault is valid and has derived speeds
    planted = {
        f"sio2_quartz-n{f.split('-n')[1].split('-r')[0]}-r{f[-1]}" for f in faults
    }
    clean = runs.drop(index=list(planted))
    assert (clean["status"] == "ok").all()
    assert (clean["valid"] == True).all()  # noqa: E712
    assert (clean["us_per_atom_step"] > 0).all()

    for name in ("results.csv", "summary.csv", "figure_scaling.html"):
        assert (campaign / "analysis" / name).is_file()
    for name in (
        "figure_scaling.json",
        "figure_memory.json",
        "info.json",
        "optimized_inference_speed_metrics_table.json",
    ):
        assert (tmp_path / "app" / name).is_file()
    # Every repeat of the two smallest sizes has a planted fault
    assert set(results["summary"]["n_atoms"]) == set(runs["n_atoms"]) - {small, medium}
    record = json.loads(
        next((campaign / "out").glob("_mock/fp32/*/mlpeg_record.json")).read_text()
    )
    assert record["hardware"]["hardware_id"]
    assert record["memory"]["gpu_method"] in (
        "per_process",
        "device_minus_baseline",
        "unavailable",
    )
    hardware_id = json.loads((campaign / "in" / "spec.json").read_text())["campaign"]
    assert hardware_id["hardware_id"] == record["hardware"]["hardware_id"]


def test_error_and_timeout(campaign: Path):
    """Non-zero exits, missing results and timeouts are recorded."""
    spec = json.loads((campaign / "in" / "spec.json").read_text())
    small = spec["systems"][0]["sizes"][0]["n_atoms"]
    records = run_campaign(
        campaign, models=["_mock"], runtime="local", precisions=["fp32"],
        repeats=3, timeout=2,
        container_env=[
            f"MLPEG_MOCK_FAIL=error:*-n{small}-r1;nojson:*-n{small}-r2;"
            f"hang:*-n{small}-r3",
        ],
    )  # fmt: skip
    status = {r["run_id"]: (r["status"], r["error"]) for r in records}
    assert status[f"sio2_quartz-n{small}-r1"] == ("error", "exit code 1")
    assert status[f"sio2_quartz-n{small}-r2"][0] == "error"
    assert "missing" in status[f"sio2_quartz-n{small}-r2"][1]
    assert status[f"sio2_quartz-n{small}-r3"][0] == "timeout"


def test_runscript_rejects_unknown_arguments(campaign: Path, tmp_path: Path):
    """The mock exits non-zero on unexpected arguments or precisions."""
    import subprocess
    import sys

    runscript = CONTRIB_ROOT / "_mock" / "runscript.py"
    base = [sys.executable, str(runscript), "--spec", str(campaign / "in/spec.json")]
    base += ["--run-id", "sio2_quartz-n72-r1", "--out", str(tmp_path)]
    assert subprocess.run([*base, "--precision", "fp16"], check=False).returncode
    assert subprocess.run(
        [*base, "--precision", "fp32", "--extra", "1"], check=False
    ).returncode


@pytest.mark.skipif(not NEP89_FILE.is_file(), reason="NEP89 weights not downloaded")
def test_nep89_reference():
    """The calorine reference uses the same nep.txt as the GPUMD container."""
    pytest.importorskip("calorine")
    from ml_peg.analysis.molecular_dynamics.optimized_inference_speed.validate import (
        get_reference_calculator,
    )

    assert (
        sha256_file(NEP89_FILE)
        == load_container_models()["nep89_gpumd"]["weights_sha256"]
    )
    atoms = quartz().repeat(2)
    atoms.calc = get_reference_calculator("nep89")
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (len(atoms), 3)


@pytest.mark.skipif(
    not (shutil.which("apptainer") and os.environ.get("MLPEG_RUN_APPTAINER_TESTS")),
    reason="Needs Apptainer and MLPEG_RUN_APPTAINER_TESTS=1",
)
def test_apptainer_mock(campaign: Path, tmp_path: Path):
    """Build the mock image and run it with the contract invocation."""
    from ml_peg.calcs.molecular_dynamics.optimized_inference_speed.build import (
        build_container,
    )

    record = build_container("_mock", tmp_path / "cache")
    assert record["labels"]["mlpeg.model"] == "mock"
    spec = json.loads((campaign / "in" / "spec.json").read_text())
    second = spec["systems"][0]["sizes"][1]["n_atoms"]
    records = run_campaign(
        campaign, models=["_mock"], cache_dir=tmp_path / "cache",
        precisions=["fp32"], repeats=1,
        container_env=[f"MLPEG_MOCK_FAIL=oom:*-n{second}-*"],
    )  # fmt: skip
    assert records[0]["status"] == "ok"
    assert records[0]["sif_sha256"] == record["sha256"]
    results = analyse_campaign(
        campaign, app_dir=tmp_path / "app", reference_getter=lambda _: MockCalculator()
    )
    assert results["runs"].set_index("run_id").loc[records[0]["run_id"], "valid"]
